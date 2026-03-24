#include "llaisys_tensor.hpp"

#include "llaisys/models/qwen2.h"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#ifdef ENABLE_NVIDIA_API
#include <nvtx3/nvToolsExt.h>
#endif

namespace {
struct Qwen2Workspace {
    size_t capacity = 0;
    size_t token_capacity = 0;
    llaisys::tensor_t index;
    llaisys::tensor_t x;
    llaisys::tensor_t attn_norm;
    llaisys::tensor_t q2d;
    llaisys::tensor_t k2d;
    llaisys::tensor_t v2d;
    llaisys::tensor_t q_rope;
    llaisys::tensor_t k_rope;
    llaisys::tensor_t attn;
    llaisys::tensor_t attn_out;
    llaisys::tensor_t x_attn;
    llaisys::tensor_t mlp_norm;
    llaisys::tensor_t gate;
    llaisys::tensor_t up;
    llaisys::tensor_t hidden;
    llaisys::tensor_t mlp_out;
    llaisys::tensor_t normed;
    llaisys::tensor_t logits;
    llaisys::tensor_t max_idx;
    llaisys::tensor_t max_val;
    llaisys::tensor_t max_idx_host;
    llaisys::tensor_t generated_ids;
};

struct NvtxRange {
    explicit NvtxRange(const char *name, bool enabled) : enabled_(enabled) {
#ifdef ENABLE_NVIDIA_API
        if (enabled_) {
            nvtxRangePushA(name);
        }
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#ifdef ENABLE_NVIDIA_API
        if (enabled_) {
            nvtxRangePop();
        }
#endif
    }

private:
    bool enabled_;
};
} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int ndevice;
    int *device_ids;
    LlaisysQwen2Weights weights;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;
    size_t cache_len = 0;
    llaisys::tensor_t pos_ids_cache;
    bool use_kv_cache = true;
    Qwen2Workspace workspace;
};

static llaisys::tensor_t unwrap(llaisysTensor_t tensor) {
    if (!tensor) {
        return nullptr;
    }
    return reinterpret_cast<LlaisysTensor *>(tensor)->tensor;
}

static int pick_device_id(const LlaisysQwen2Model *model) {
    if (!model || model->ndevice <= 0 || model->device_ids == nullptr) {
        return 0;
    }
    return model->device_ids[0];
}

static llaisysTensor_t *alloc_layer_ptrs(size_t nlayer) {
    auto *ptrs = new llaisysTensor_t[nlayer];
    for (size_t i = 0; i < nlayer; ++i) {
        ptrs[i] = nullptr;
    }
    return ptrs;
}

static void ensure_workspace(LlaisysQwen2Model *model, size_t cur_len, int device_id) {
    if (model->workspace.capacity >= cur_len) {
        return;
    }

    // 中间张量按见过的最大 cur_len 一次性分配，后续推理只做 slice/view 复用。
    // 这样可以显著减少 decode 阶段的 cudaMalloc/cudaFree 与临时对象开销。
    model->workspace.capacity = cur_len;
    const size_t cap = model->workspace.capacity;
    const auto dtype = model->meta.dtype;
    const auto device = model->device;

    model->workspace.index = llaisys::Tensor::create({cap}, LLAISYS_DTYPE_I64, device, device_id);
    model->workspace.x = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.attn_norm = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.q2d = llaisys::Tensor::create({cap, model->meta.nh * model->meta.dh}, dtype, device, device_id);
    model->workspace.k2d = llaisys::Tensor::create({cap, model->meta.nkvh * model->meta.dh}, dtype, device, device_id);
    model->workspace.v2d = llaisys::Tensor::create({cap, model->meta.nkvh * model->meta.dh}, dtype, device, device_id);
    model->workspace.q_rope = llaisys::Tensor::create({cap, model->meta.nh, model->meta.dh}, dtype, device, device_id);
    model->workspace.k_rope = llaisys::Tensor::create({cap, model->meta.nkvh, model->meta.dh}, dtype, device, device_id);
    model->workspace.attn = llaisys::Tensor::create({cap, model->meta.nh, model->meta.dh}, dtype, device, device_id);
    model->workspace.attn_out = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.x_attn = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.mlp_norm = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.gate = llaisys::Tensor::create({cap, model->meta.di}, dtype, device, device_id);
    model->workspace.up = llaisys::Tensor::create({cap, model->meta.di}, dtype, device, device_id);
    model->workspace.hidden = llaisys::Tensor::create({cap, model->meta.di}, dtype, device, device_id);
    model->workspace.mlp_out = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    model->workspace.normed = llaisys::Tensor::create({cap, model->meta.hs}, dtype, device, device_id);
    // 最终只需要最后一个 token 的 logits，因此 workspace 固定为 [1, voc] 即可。
    model->workspace.logits = llaisys::Tensor::create({1, model->meta.voc}, dtype, device, device_id);
    model->workspace.max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    model->workspace.max_val = llaisys::Tensor::create({1}, dtype, device, device_id);
}

static void ensure_token_workspace(LlaisysQwen2Model *model, size_t capacity, int device_id) {
    if (model->workspace.token_capacity >= capacity) {
        return;
    }
    model->workspace.token_capacity = capacity;
    model->workspace.generated_ids = llaisys::Tensor::create(
        {capacity}, LLAISYS_DTYPE_I64, model->device, device_id);
}

static int64_t copy_token_to_host(LlaisysQwen2Model *model, llaisys::tensor_t token, int device_id) {
    if (token->deviceType() == LLAISYS_DEVICE_CPU) {
        return reinterpret_cast<const int64_t *>(token->data())[0];
    }

    if (!model->workspace.max_idx_host) {
        // 复用一个常驻的 pinned host buffer，避免每个 decode step 都重新申请/释放 host staging。
        llaisys::core::context().setDevice(model->device, device_id);
        model->workspace.max_idx_host = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    }

    llaisys::core::context().setDevice(token->deviceType(), token->deviceId());
    llaisys::core::context().runtime().api()->memcpy_sync(
        model->workspace.max_idx_host->data(),
        token->data(),
        sizeof(int64_t),
        LLAISYS_MEMCPY_D2H);
    return reinterpret_cast<const int64_t *>(model->workspace.max_idx_host->data())[0];
}

static llaisys::tensor_t qwen2_forward(
    LlaisysQwen2Model *model,
    llaisys::tensor_t index,
    size_t ntoken,
    size_t cur_len,
    size_t pos_start,
    bool use_cache,
    int device_id) {
    const bool enable_nvtx = model->device == LLAISYS_DEVICE_NVIDIA;
    ensure_workspace(model, cur_len, device_id);

    NvtxRange infer_range(use_cache ? "llaisys_decode" : "llaisys_prefill", enable_nvtx);

    auto x = model->workspace.x->slice(0, 0, cur_len);
    {
        NvtxRange range("embedding", enable_nvtx);
        llaisys::ops::embedding(x, index, unwrap(model->weights.in_embed));
    }

    if (!model->pos_ids_cache) {
        model->pos_ids_cache = llaisys::Tensor::create({model->meta.maxseq}, LLAISYS_DTYPE_I64, model->device, device_id);
        std::vector<int64_t> full_pos(model->meta.maxseq);
        for (size_t i = 0; i < model->meta.maxseq; ++i) {
            full_pos[i] = static_cast<int64_t>(i);
        }
        model->pos_ids_cache->load(full_pos.data());
    }
    auto pos_ids = model->pos_ids_cache->slice(0, pos_start, pos_start + cur_len);

    auto x_cur = x;
    auto attn_norm = model->workspace.attn_norm->slice(0, 0, cur_len);
    auto q2d = model->workspace.q2d->slice(0, 0, cur_len);
    auto k2d = model->workspace.k2d->slice(0, 0, cur_len);
    auto v2d = model->workspace.v2d->slice(0, 0, cur_len);
    auto q_rope = model->workspace.q_rope->slice(0, 0, cur_len);
    auto k_rope = model->workspace.k_rope->slice(0, 0, cur_len);
    auto attn = model->workspace.attn->slice(0, 0, cur_len);
    auto attn_out = model->workspace.attn_out->slice(0, 0, cur_len);
    auto x_attn = model->workspace.x_attn->slice(0, 0, cur_len);
    auto mlp_norm = model->workspace.mlp_norm->slice(0, 0, cur_len);
    auto gate = model->workspace.gate->slice(0, 0, cur_len);
    auto up = model->workspace.up->slice(0, 0, cur_len);
    auto hidden = model->workspace.hidden->slice(0, 0, cur_len);
    auto mlp_out = model->workspace.mlp_out->slice(0, 0, cur_len);

    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        {
            NvtxRange range("layer_attn", enable_nvtx);
            llaisys::ops::rms_norm(attn_norm, x_cur, unwrap(model->weights.attn_norm_w[layer]), model->meta.epsilon);
            llaisys::ops::linear(q2d, attn_norm, unwrap(model->weights.attn_q_w[layer]),
                                 model->weights.attn_q_b ? unwrap(model->weights.attn_q_b[layer]) : nullptr);
            llaisys::ops::linear(k2d, attn_norm, unwrap(model->weights.attn_k_w[layer]),
                                 model->weights.attn_k_b ? unwrap(model->weights.attn_k_b[layer]) : nullptr);
            llaisys::ops::linear(v2d, attn_norm, unwrap(model->weights.attn_v_w[layer]),
                                 model->weights.attn_v_b ? unwrap(model->weights.attn_v_b[layer]) : nullptr);

            auto q = q2d->view({cur_len, model->meta.nh, model->meta.dh});
            auto k = k2d->view({cur_len, model->meta.nkvh, model->meta.dh});
            auto v = v2d->view({cur_len, model->meta.nkvh, model->meta.dh});

            llaisys::ops::rope(q_rope, q, pos_ids, model->meta.theta);
            llaisys::ops::rope(k_rope, k, pos_ids, model->meta.theta);

            if (!model->k_cache[layer]) {
                model->k_cache[layer] = llaisys::Tensor::create(
                    {model->meta.maxseq, model->meta.nkvh, model->meta.dh},
                    model->meta.dtype, model->device, device_id);
                model->v_cache[layer] = llaisys::Tensor::create(
                    {model->meta.maxseq, model->meta.nkvh, model->meta.dh},
                    model->meta.dtype, model->device, device_id);
            }

            if (use_cache) {
                auto k_dst = model->k_cache[layer]->slice(0, pos_start, pos_start + cur_len);
                auto v_dst = model->v_cache[layer]->slice(0, pos_start, pos_start + cur_len);
                llaisys::ops::rearrange(k_dst, k_rope);
                llaisys::ops::rearrange(v_dst, v);
            } else {
                auto k_dst = model->k_cache[layer]->slice(0, 0, ntoken);
                auto v_dst = model->v_cache[layer]->slice(0, 0, ntoken);
                llaisys::ops::rearrange(k_dst, k_rope);
                llaisys::ops::rearrange(v_dst, v);
            }

            const size_t total_len = use_cache ? (model->cache_len + cur_len) : ntoken;
            auto k_all = model->k_cache[layer]->slice(0, 0, total_len);
            auto v_all = model->v_cache[layer]->slice(0, 0, total_len);

            const float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));
            llaisys::ops::self_attention(attn, q_rope, k_all, v_all, scale);

            auto attn2d = attn->view({cur_len, model->meta.hs});
            llaisys::ops::linear(attn_out, attn2d, unwrap(model->weights.attn_o_w[layer]), nullptr);
            llaisys::ops::add(x_attn, x_cur, attn_out);
        }

        {
            NvtxRange range("layer_mlp", enable_nvtx);
            llaisys::ops::rms_norm(mlp_norm, x_attn, unwrap(model->weights.mlp_norm_w[layer]), model->meta.epsilon);
            llaisys::ops::linear(gate, mlp_norm, unwrap(model->weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear(up, mlp_norm, unwrap(model->weights.mlp_up_w[layer]), nullptr);
            llaisys::ops::swiglu(hidden, gate, up);
            llaisys::ops::linear(mlp_out, hidden, unwrap(model->weights.mlp_down_w[layer]), nullptr);
            llaisys::ops::add(x_cur, x_attn, mlp_out);
        }
    }

    if (!use_cache) {
        model->cache_len = ntoken;
    } else {
        model->cache_len += cur_len;
    }

    auto normed = model->workspace.normed->slice(0, 0, cur_len);
    {
        NvtxRange range("lm_head", enable_nvtx);
        llaisys::ops::rms_norm(normed, x_cur, unwrap(model->weights.out_norm_w), model->meta.epsilon);
        auto last_normed = normed->slice(0, cur_len - 1, cur_len);
        auto logits = model->workspace.logits;
        llaisys::ops::linear(logits, last_normed, unwrap(model->weights.out_embed), nullptr);

        auto logits_view = logits->view({model->meta.voc});
        auto max_idx = model->workspace.max_idx;
        auto max_val = model->workspace.max_val;
        llaisys::ops::argmax(max_idx, max_val, logits_view);
        return max_idx;
    }
}

__C {
__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    std::cout << "[LLAISYS] qwen2 create: nlayer=" << meta->nlayer
              << " dtype=" << meta->dtype
              << " device=" << device
              << " ndevice=" << ndevice << std::endl;

    auto *model = new LlaisysQwen2Model{};
    model->meta = *meta;
    model->device = device;
    model->ndevice = ndevice;
    model->device_ids = device_ids;

    model->weights.in_embed = nullptr;
    model->weights.out_embed = nullptr;
    model->weights.out_norm_w = nullptr;

    size_t nlayer = meta->nlayer;
    model->weights.attn_norm_w = alloc_layer_ptrs(nlayer);
    model->weights.attn_q_w = alloc_layer_ptrs(nlayer);
    model->weights.attn_q_b = alloc_layer_ptrs(nlayer);
    model->weights.attn_k_w = alloc_layer_ptrs(nlayer);
    model->weights.attn_k_b = alloc_layer_ptrs(nlayer);
    model->weights.attn_v_w = alloc_layer_ptrs(nlayer);
    model->weights.attn_v_b = alloc_layer_ptrs(nlayer);
    model->weights.attn_o_w = alloc_layer_ptrs(nlayer);
    model->weights.mlp_norm_w = alloc_layer_ptrs(nlayer);
    model->weights.mlp_gate_w = alloc_layer_ptrs(nlayer);
    model->weights.mlp_up_w = alloc_layer_ptrs(nlayer);
    model->weights.mlp_down_w = alloc_layer_ptrs(nlayer);
    model->k_cache.resize(nlayer);
    model->v_cache.resize(nlayer);
    model->cache_len = 0;
    model->pos_ids_cache = nullptr;

    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    std::cout << "[LLAISYS] qwen2 destroy" << std::endl;

    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    std::cout << "[LLAISYS] qwen2 weights" << std::endl;
    return model ? &model->weights : nullptr;
}

__export void llaisysQwen2ModelSetKVCache(struct LlaisysQwen2Model *model, uint8_t enabled) {
    if (!model) {
        return;
    }
    model->use_kv_cache = enabled != 0;
    if (!model->use_kv_cache) {
        model->cache_len = 0;
    }
}

__export void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    model->cache_len = 0;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || ntoken == 0 || token_ids == nullptr) {
        return -1;
    }
    if (!model->weights.in_embed || !model->weights.out_embed || !model->weights.out_norm_w) {
        std::cerr << "[LLAISYS] qwen2 infer: missing weights" << std::endl;
        return -1;
    }

    const int device_id = pick_device_id(model);
    bool use_cache = model->use_kv_cache && (model->cache_len > 0 && ntoken == model->cache_len + 1);
    if (!use_cache) {
        model->cache_len = 0;
    }

    const size_t cur_len = use_cache ? 1 : ntoken;
    const size_t pos_start = use_cache ? model->cache_len : 0;
    ensure_workspace(model, cur_len, device_id);
    auto index = model->workspace.index->slice(0, 0, cur_len);
    index->load(token_ids + (ntoken - cur_len));

    auto max_idx = qwen2_forward(model, index, ntoken, cur_len, pos_start, use_cache, device_id);
    return copy_token_to_host(model, max_idx, device_id);
}

__export size_t llaisysQwen2ModelGenerate(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken,
    size_t max_new_tokens,
    size_t capacity) {
    if (!model || !token_ids || ntoken == 0 || capacity < ntoken) {
        return 0;
    }

    // generate 是“完整一次生成”的高层接口，decode loop 保持在 backend 内部执行。
    // 生成序列常驻 device，只在结束后一次性拷回 host，尽量避免逐 token 的 host/device 往返。
    llaisysQwen2ModelResetCache(model);

    const int device_id = pick_device_id(model);
    ensure_token_workspace(model, capacity, device_id);

    auto generated_ids = model->workspace.generated_ids;
    auto prompt_ids = generated_ids->slice(0, 0, ntoken);
    prompt_ids->load(token_ids);

    size_t total = ntoken;
    for (size_t step = 0; step < max_new_tokens && total < capacity; ++step) {
        const bool use_cache = step > 0;
        const size_t cur_len = use_cache ? 1 : ntoken;
        const size_t pos_start = use_cache ? (total - 1) : 0;
        auto index = use_cache ? generated_ids->slice(0, total - 1, total) : prompt_ids;
        auto next_idx = qwen2_forward(model, index, total, cur_len, pos_start, use_cache, device_id);

        auto dst = generated_ids->slice(0, total, total + 1);
        llaisys::core::context().setDevice(next_idx->deviceType(), next_idx->deviceId());
        llaisys::core::context().runtime().api()->memcpy_sync(
            dst->data(),
            next_idx->data(),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2D);
        ++total;
    }

    auto out = generated_ids->slice(0, 0, total);
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    llaisys::core::context().runtime().api()->memcpy_sync(
        token_ids,
        out->data(),
        total * sizeof(int64_t),
        LLAISYS_MEMCPY_D2H);

    if (model->meta.end_token >= 0) {
        for (size_t i = ntoken; i < total; ++i) {
            if (token_ids[i] == model->meta.end_token) {
                total = i + 1;
                break;
            }
        }
    }
    return total;
}
}
