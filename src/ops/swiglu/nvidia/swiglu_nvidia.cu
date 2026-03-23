#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {
inline void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(llaisys::fp16_t v) { return __half2float(__ushort_as_half(v._v)); }
__device__ inline float to_float(llaisys::bf16_t v) { return __bfloat162float(__ushort_as_bfloat16(v._v)); }

template <typename T>
__device__ inline T from_float(float v);
template <>
__device__ inline float from_float<float>(float v) { return v; }
template <>
__device__ inline llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    llaisys::fp16_t out{}; out._v = __half_as_ushort(__float2half(v)); return out;
}
template <>
__device__ inline llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    llaisys::bf16_t out{}; out._v = __bfloat16_as_ushort(__float2bfloat16(v)); return out;
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    const float g = to_float(gate[idx]);
    const float u = to_float(up[idx]);
    const float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = from_float<T>(u * (g * sig));
}

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up, size_t numel, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    swiglu_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, numel);
    check_cuda(cudaGetLastError(), "CUDA swiglu kernel launch failed");
}
} // namespace

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel,
            cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            numel,
            cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            numel,
            cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
