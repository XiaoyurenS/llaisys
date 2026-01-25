#include "op.hpp"

#include "../../utils.hpp"

#include <type_traits>

namespace llaisys::ops {
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            if (bias) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    acc = llaisys::utils::cast<float>(bias[j]);
                } else {
                    acc = static_cast<float>(bias[j]);
                }
            }
            const T *in_row = in + i * k;
            const T *w_row = weight + j * k;
            for (size_t t = 0; t < k; ++t) {
                float a;
                float b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in_row[t]);
                    b = llaisys::utils::cast<float>(w_row[t]);
                } else {
                    a = static_cast<float>(in_row[t]);
                    b = static_cast<float>(w_row[t]);
                }
                acc += a * b;
            }

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * n + j] = llaisys::utils::cast<T>(acc);
            } else {
                out[i * n + j] = static_cast<T>(acc);
            }
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1, "linear: bias must be 1D");
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2, "linear: in must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 2, "linear: weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "linear: out must be 2D");

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    CHECK_ARGUMENT(weight->shape()[1] == k, "linear: weight dim1 must match in dim1");
    CHECK_ARGUMENT(out->shape()[0] == m, "linear: out dim0 must match in dim0");
    CHECK_ARGUMENT(out->shape()[1] == n, "linear: out dim1 must match weight dim0");
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == n, "linear: bias size must match weight dim0");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: tensors must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out->data()),
                       reinterpret_cast<const float *>(in->data()),
                       reinterpret_cast<const float *>(weight->data()),
                       bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                       m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr,
                       m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr,
                       m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
