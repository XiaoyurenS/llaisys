#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <type_traits>

namespace llaisys::ops {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half = d / 2;
    for (size_t i = 0; i < seqlen; ++i) {
        float pos = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < nhead; ++h) {
            size_t base = (i * nhead + h) * d;
            for (size_t j = 0; j < half; ++j) {
                float inv_freq = std::pow(theta, (2.0f * j) / static_cast<float>(d));
                float phi = pos / inv_freq;
                float c = std::cos(phi);
                float s = std::sin(phi);

                float a;
                float b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[base + j]);
                    b = llaisys::utils::cast<float>(in[base + j + half]);
                    out[base + j] = llaisys::utils::cast<T>(a * c - b * s);
                    out[base + j + half] = llaisys::utils::cast<T>(b * c + a * s);
                } else {
                    a = static_cast<float>(in[base + j]);
                    b = static_cast<float>(in[base + j + half]);
                    out[base + j] = static_cast<T>(a * c - b * s);
                    out[base + j + half] = static_cast<T>(b * c + a * s);
                }
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");
    CHECK_ARGUMENT(in->ndim() == 3, "rope: in must be 3D");
    CHECK_ARGUMENT(out->ndim() == 3, "rope: out must be 3D");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rope: out shape must match in shape");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "rope: pos_ids length must match seqlen");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    CHECK_ARGUMENT(d % 2 == 0, "rope: last dimension must be even");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out->data()),
                     reinterpret_cast<const float *>(in->data()),
                     reinterpret_cast<const int64_t *>(pos_ids->data()),
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                     reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                     reinterpret_cast<const int64_t *>(pos_ids->data()),
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                     reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                     reinterpret_cast<const int64_t *>(pos_ids->data()),
                     seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
