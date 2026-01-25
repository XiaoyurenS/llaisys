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

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int ndevice;
    int *device_ids;
    LlaisysQwen2Weights weights;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;
    size_t cache_len = 0;
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

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    std::cout << "[LLAISYS] qwen2 infer ntoken=" << ntoken << std::endl;
    if (!model || ntoken == 0 || token_ids == nullptr) {
        return -1;
    }
    if (!model->weights.in_embed || !model->weights.out_embed || !model->weights.out_norm_w) {
        std::cerr << "[LLAISYS] qwen2 infer: missing weights" << std::endl;
        return -1;
    }

    int device_id = pick_device_id(model);

    bool use_cache = (model->cache_len > 0 && ntoken == model->cache_len + 1);
    if (!use_cache) {
        model->cache_len = 0;
    }

    size_t cur_len = use_cache ? 1 : ntoken;
    size_t pos_start = use_cache ? model->cache_len : 0;

    auto index = llaisys::Tensor::create({cur_len}, LLAISYS_DTYPE_I64, model->device, device_id);
    index->load(token_ids + (ntoken - cur_len));
    auto x = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
    llaisys::ops::embedding(x, index, unwrap(model->weights.in_embed));

    std::vector<int64_t> pos_buf(cur_len);
    for (size_t i = 0; i < cur_len; ++i) {
        pos_buf[i] = static_cast<int64_t>(pos_start + i);
    }
    auto pos_ids = llaisys::Tensor::create({cur_len}, LLAISYS_DTYPE_I64, model->device, device_id);
    pos_ids->load(pos_buf.data());

    auto x_cur = x;
    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        if (!model->weights.attn_norm_w[layer] || !model->weights.attn_q_w[layer]
            || !model->weights.attn_k_w[layer] || !model->weights.attn_v_w[layer]
            || !model->weights.attn_o_w[layer] || !model->weights.mlp_norm_w[layer]
            || !model->weights.mlp_gate_w[layer] || !model->weights.mlp_up_w[layer]
            || !model->weights.mlp_down_w[layer]) {
            std::cerr << "[LLAISYS] qwen2 infer: missing weights at layer " << layer << std::endl;
            return -1;
        }

        auto attn_norm = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::rms_norm(attn_norm, x_cur, unwrap(model->weights.attn_norm_w[layer]), model->meta.epsilon);

        auto q2d = llaisys::Tensor::create({cur_len, model->meta.nh * model->meta.dh}, model->meta.dtype, model->device, device_id);
        auto k2d = llaisys::Tensor::create({cur_len, model->meta.nkvh * model->meta.dh}, model->meta.dtype, model->device, device_id);
        auto v2d = llaisys::Tensor::create({cur_len, model->meta.nkvh * model->meta.dh}, model->meta.dtype, model->device, device_id);

        llaisys::ops::linear(q2d, attn_norm, unwrap(model->weights.attn_q_w[layer]),
                             model->weights.attn_q_b ? unwrap(model->weights.attn_q_b[layer]) : nullptr);
        llaisys::ops::linear(k2d, attn_norm, unwrap(model->weights.attn_k_w[layer]),
                             model->weights.attn_k_b ? unwrap(model->weights.attn_k_b[layer]) : nullptr);
        llaisys::ops::linear(v2d, attn_norm, unwrap(model->weights.attn_v_w[layer]),
                             model->weights.attn_v_b ? unwrap(model->weights.attn_v_b[layer]) : nullptr);

        auto q = q2d->view({cur_len, model->meta.nh, model->meta.dh});
        auto k = k2d->view({cur_len, model->meta.nkvh, model->meta.dh});
        auto v = v2d->view({cur_len, model->meta.nkvh, model->meta.dh});

        auto q_rope = llaisys::Tensor::create({cur_len, model->meta.nh, model->meta.dh}, model->meta.dtype, model->device, device_id);
        auto k_rope = llaisys::Tensor::create({cur_len, model->meta.nkvh, model->meta.dh}, model->meta.dtype, model->device, device_id);
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

        size_t total_len = use_cache ? (model->cache_len + cur_len) : ntoken;
        auto k_all = model->k_cache[layer]->slice(0, 0, total_len);
        auto v_all = model->v_cache[layer]->slice(0, 0, total_len);

        float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));
        auto attn = llaisys::Tensor::create({cur_len, model->meta.nh, model->meta.dh}, model->meta.dtype, model->device, device_id);
        llaisys::ops::self_attention(attn, q_rope, k_all, v_all, scale);

        auto attn2d = attn->view({cur_len, model->meta.hs});
        auto attn_out = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::linear(attn_out, attn2d, unwrap(model->weights.attn_o_w[layer]), nullptr);

        auto x_attn = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::add(x_attn, x_cur, attn_out);

        auto mlp_norm = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::rms_norm(mlp_norm, x_attn, unwrap(model->weights.mlp_norm_w[layer]), model->meta.epsilon);

        auto gate = llaisys::Tensor::create({cur_len, model->meta.di}, model->meta.dtype, model->device, device_id);
        auto up = llaisys::Tensor::create({cur_len, model->meta.di}, model->meta.dtype, model->device, device_id);
        llaisys::ops::linear(gate, mlp_norm, unwrap(model->weights.mlp_gate_w[layer]), nullptr);
        llaisys::ops::linear(up, mlp_norm, unwrap(model->weights.mlp_up_w[layer]), nullptr);

        auto hidden = llaisys::Tensor::create({cur_len, model->meta.di}, model->meta.dtype, model->device, device_id);
        llaisys::ops::swiglu(hidden, gate, up);

        auto mlp_out = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::linear(mlp_out, hidden, unwrap(model->weights.mlp_down_w[layer]), nullptr);

        auto x_out = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::add(x_out, x_attn, mlp_out);
        x_cur = x_out;
    }

    if (!use_cache) {
        model->cache_len = ntoken;
    } else {
        model->cache_len += cur_len;
    }

    auto normed = llaisys::Tensor::create({cur_len, model->meta.hs}, model->meta.dtype, model->device, device_id);
    llaisys::ops::rms_norm(normed, x_cur, unwrap(model->weights.out_norm_w), model->meta.epsilon);

    auto logits = llaisys::Tensor::create({cur_len, model->meta.voc}, model->meta.dtype, model->device, device_id);
    llaisys::ops::linear(logits, normed, unwrap(model->weights.out_embed), nullptr);

    llaisys::tensor_t last_logits = logits;
    if (cur_len > 1) {
        last_logits = logits->slice(0, cur_len - 1, cur_len);
    }
    auto logits_view = last_logits->view({model->meta.voc});
    auto max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, model->device, device_id);
    auto max_val = llaisys::Tensor::create({1}, model->meta.dtype, model->device, device_id);
    llaisys::ops::argmax(max_idx, max_val, logits_view);

    llaisys::tensor_t max_idx_host = max_idx;
    if (max_idx->deviceType() != LLAISYS_DEVICE_CPU) {
        max_idx_host = max_idx->to(LLAISYS_DEVICE_CPU);
    }
    auto *idx_ptr = reinterpret_cast<const int64_t *>(max_idx_host->data());
    return idx_ptr[0];
}
}
