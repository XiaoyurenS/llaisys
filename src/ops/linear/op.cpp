#include "op.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <array>
#include <type_traits>

namespace llaisys::ops {

#ifdef LLAISYS_USE_OPENMP
#include <omp.h>
#endif

namespace {
template <typename T>
inline float to_float(T val) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
inline T from_float(float val) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<T>(val);
    } else {
        return static_cast<T>(val);
    }
}
} // namespace

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    constexpr size_t kBlock = 64;
    constexpr size_t nBlock = 32;

#ifdef LLAISYS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m); ++i) {
        const T *in_row = in + static_cast<size_t>(i) * k;
        T *out_row = out + static_cast<size_t>(i) * n;

        for (size_t j0 = 0; j0 < n; j0 += nBlock) {
            const size_t j_end = std::min(j0 + nBlock, n);
            std::array<float, nBlock> acc{};

            for (size_t jj = 0; jj < j_end - j0; ++jj) {
                acc[jj] = bias ? to_float(bias[j0 + jj]) : 0.0f;
            }

            for (size_t t0 = 0; t0 < k; t0 += kBlock) {
                const size_t t_end = std::min(t0 + kBlock, k);
                for (size_t t = t0; t < t_end; ++t) {
                    const float a = to_float(in_row[t]);
                    for (size_t j = j0; j < j_end; ++j) {
                        acc[j - j0] += a * to_float(weight[j * k + t]);
                    }
                }
            }

            for (size_t j = j0; j < j_end; ++j) {
                out_row[j] = from_float<T>(acc[j - j0]);
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
