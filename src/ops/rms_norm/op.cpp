#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <type_traits>

namespace llaisys::ops {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t m, size_t d, float eps) {
    for (size_t i = 0; i < m; ++i) {
        const T *in_row = in + i * d;
        T *out_row = out + i * d;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float v;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(in_row[j]);
            } else {
                v = static_cast<float>(in_row[j]);
            }
            sum_sq += v * v;
        }
        float mean = sum_sq / static_cast<float>(d);
        float denom = std::sqrt(mean + eps);

        for (size_t j = 0; j < d; ++j) {
            float v;
            float w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(in_row[j]);
                w = llaisys::utils::cast<float>(weight[j]);
                out_row[j] = llaisys::utils::cast<T>(v * w / denom);
            } else {
                v = static_cast<float>(in_row[j]);
                w = static_cast<float>(weight[j]);
                out_row[j] = static_cast<T>(v * w / denom);
            }
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2, "rms_norm: in must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "rms_norm: out must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm: weight must be 1D");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1],
                   "rms_norm: out shape must match in shape");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "rms_norm: weight size must match in dim1");

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous.");

    size_t m = in->shape()[0];
    size_t d = in->shape()[1];

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out->data()),
                         reinterpret_cast<const float *>(in->data()),
                         reinterpret_cast<const float *>(weight->data()),
                         m, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                         reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                         reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                         m, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                         reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                         reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                         m, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
