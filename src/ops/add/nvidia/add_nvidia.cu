#include "add_nvidia.cuh"

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

__device__ inline float to_float(float v) {
    return v;
}

__device__ inline float to_float(llaisys::fp16_t v) {
    __half raw = __ushort_as_half(v._v);
    return __half2float(raw);
}

__device__ inline float to_float(llaisys::bf16_t v) {
    __nv_bfloat16 raw = __ushort_as_bfloat16(v._v);
    return __bfloat162float(raw);
}

template <typename T>
__device__ inline T from_float(float v);

template <>
__device__ inline float from_float<float>(float v) {
    return v;
}

template <>
__device__ inline llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    llaisys::fp16_t out{};
    out._v = __half_as_ushort(__float2half(v));
    return out;
}

template <>
__device__ inline llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    llaisys::bf16_t out{};
    out._v = __bfloat16_as_ushort(__float2bfloat16(v));
    return out;
}

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    // 统一在 float 中完成计算，兼容 fp32/fp16/bf16 三种类型。
    const float av = to_float(a[idx]);
    const float bv = to_float(b[idx]);
    c[idx] = from_float<T>(av + bv);
}
} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    check_cuda(cudaGetLastError(), "CUDA add kernel launch failed");
}
} // namespace llaisys::ops::nvidia
