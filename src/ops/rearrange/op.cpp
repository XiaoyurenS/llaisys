#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include <cstring>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.cuh"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    size_t elem_size = out->elementSize();
    size_t ndim = out->ndim();
    size_t total = out->numel();

    if (total == 0) {
        return;
    }

    if (out->isContiguous() && in->isContiguous()) {
        if (out->deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(out->data(), in->data(), total * elem_size);
            return;
        }

        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
            return nvidia::rearrange_contiguous(
                out->data(),
                in->data(),
                total * elem_size,
                llaisys::core::context().runtime().stream());
#else
            EXCEPTION_UNSUPPORTED_DEVICE;
#endif
        }
        EXCEPTION_UNSUPPORTED_DEVICE;
        return;
    }

    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU,
                   "rearrange: non-contiguous NVIDIA path is not implemented yet");

    std::vector<size_t> idx(ndim, 0);
    const auto &shape = out->shape();
    const auto &out_strides = out->strides();
    const auto &in_strides = in->strides();

    for (size_t n = 0; n < total; ++n) {
        size_t out_offset = 0;
        size_t in_offset = 0;
        for (size_t i = 0; i < ndim; ++i) {
            out_offset += idx[i] * static_cast<size_t>(out_strides[i]);
            in_offset += idx[i] * static_cast<size_t>(in_strides[i]);
        }

        std::memcpy(
            out->data() + out_offset * elem_size,
            in->data() + in_offset * elem_size,
            elem_size);

        for (size_t d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
} // namespace llaisys::ops
