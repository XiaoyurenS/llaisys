#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <type_traits>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

namespace llaisys::ops {

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g;
        float u;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g = llaisys::utils::cast<float>(gate[i]);
            u = llaisys::utils::cast<float>(up[i]);
        } else {
            g = static_cast<float>(gate[i]);
            u = static_cast<float>(up[i]);
        }

        float sig = 1.0f / (1.0f + std::exp(-g));
        float val = u * (g * sig);

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(val);
        } else {
            out[i] = static_cast<T>(val);
        }
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");

    size_t numel = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_(reinterpret_cast<float *>(out->data()),
                           reinterpret_cast<const float *>(gate->data()),
                           reinterpret_cast<const float *>(up->data()),
                           numel);
        case LLAISYS_DTYPE_F16:
            return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                           numel);
        case LLAISYS_DTYPE_BF16:
            return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                           numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            numel,
            llaisys::core::context().runtime().stream());
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
