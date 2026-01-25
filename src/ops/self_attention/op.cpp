#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <type_traits>
#include <vector>

namespace llaisys::ops {

template <typename T>
void self_attention_(
    T *out, const T *q, const T *k, const T *v,
    size_t seqlen, size_t nhead, size_t d,
    size_t total_len, size_t nkvhead, size_t dv,
    float scale) {
    std::vector<float> scores(total_len);

    int64_t past_len = static_cast<int64_t>(total_len) - static_cast<int64_t>(seqlen);
    if (past_len < 0) {
        past_len = 0;
    }

    for (size_t i = 0; i < seqlen; ++i) {
        size_t kv_repeat = nhead / nkvhead;
        for (size_t h = 0; h < nhead; ++h) {
            size_t kvh = h / kv_repeat;

            size_t max_t = static_cast<size_t>(past_len) + i;
            if (max_t >= total_len) {
                max_t = total_len - 1;
            }

            // Compute attention scores for t in [0, max_t]
            for (size_t t = 0; t <= max_t; ++t) {
                float acc = 0.0f;
                size_t q_base = (i * nhead + h) * d;
                size_t k_base = (t * nkvhead + kvh) * d;
                for (size_t j = 0; j < d; ++j) {
                    float qv;
                    float kv;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        qv = llaisys::utils::cast<float>(q[q_base + j]);
                        kv = llaisys::utils::cast<float>(k[k_base + j]);
                    } else {
                        qv = static_cast<float>(q[q_base + j]);
                        kv = static_cast<float>(k[k_base + j]);
                    }
                    acc += qv * kv;
                }
                scores[t] = acc * scale;
            }

            // Softmax over [0..max_t]
            float max_score = -INFINITY;
            for (size_t t = 0; t <= max_t; ++t) {
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }
            float sum = 0.0f;
            for (size_t t = 0; t <= max_t; ++t) {
                float e = std::exp(scores[t] - max_score);
                scores[t] = e;
                sum += e;
            }
            float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);

            // Weighted sum of V
            size_t out_base = (i * nhead + h) * dv;
            for (size_t j = 0; j < dv; ++j) {
                float acc = 0.0f;
                for (size_t t = 0; t <= max_t; ++t) {
                    size_t v_base = (t * nkvhead + kvh) * dv;
                    float vv;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        vv = llaisys::utils::cast<float>(v[v_base + j]);
                    } else {
                        vv = static_cast<float>(v[v_base + j]);
                    }
                    acc += scores[t] * inv_sum * vv;
                }

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[out_base + j] = llaisys::utils::cast<T>(acc);
                } else {
                    out[out_base + j] = static_cast<T>(acc);
                }
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    CHECK_ARGUMENT(q->ndim() == 3, "self_attention: q must be 3D");
    CHECK_ARGUMENT(k->ndim() == 3, "self_attention: k must be 3D");
    CHECK_ARGUMENT(v->ndim() == 3, "self_attention: v must be 3D");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "self_attention: attn_val must be 3D");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    CHECK_ARGUMENT(k->shape()[2] == d, "self_attention: k dim2 must match q dim2");
    CHECK_ARGUMENT(v->shape()[0] == total_len, "self_attention: v dim0 must match k dim0");
    CHECK_ARGUMENT(v->shape()[1] == nkvhead, "self_attention: v dim1 must match k dim1");
    CHECK_ARGUMENT(attn_val->shape()[0] == seqlen, "self_attention: attn_val dim0 must match q dim0");
    CHECK_ARGUMENT(attn_val->shape()[1] == nhead, "self_attention: attn_val dim1 must match q dim1");
    CHECK_ARGUMENT(attn_val->shape()[2] == dv, "self_attention: attn_val dim2 must match v dim2");
    CHECK_ARGUMENT(nhead % nkvhead == 0, "self_attention: nhead must be divisible by nkvhead");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val->data()),
                               reinterpret_cast<const float *>(q->data()),
                               reinterpret_cast<const float *>(k->data()),
                               reinterpret_cast<const float *>(v->data()),
                               seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
                               reinterpret_cast<const llaisys::fp16_t *>(q->data()),
                               reinterpret_cast<const llaisys::fp16_t *>(k->data()),
                               reinterpret_cast<const llaisys::fp16_t *>(v->data()),
                               seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
                               reinterpret_cast<const llaisys::bf16_t *>(q->data()),
                               reinterpret_cast<const llaisys::bf16_t *>(k->data()),
                               reinterpret_cast<const llaisys::bf16_t *>(v->data()),
                               seqlen, nhead, d, total_len, nkvhead, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
