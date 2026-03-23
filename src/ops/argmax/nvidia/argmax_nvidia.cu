#include "argmax_nvidia.cuh"

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
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    __shared__ float best_vals[256];
    __shared__ int64_t best_indices[256];

    const size_t tid = static_cast<size_t>(threadIdx.x);
    float local_best = -1.0e30f;
    int64_t local_idx = 0;

    for (size_t i = tid; i < n; i += blockDim.x) {
        const float v = to_float(vals[i]);
        if (v > local_best) {
            local_best = v;
            local_idx = static_cast<int64_t>(i);
        }
    }

    best_vals[tid] = local_best;
    best_indices[tid] = local_idx;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && best_vals[tid + stride] > best_vals[tid]) {
            best_vals[tid] = best_vals[tid + stride];
            best_indices[tid] = best_indices[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = best_indices[0];
        max_val[0] = from_float<T>(best_vals[0]);
    }
}

template <typename T>
void launch_argmax(int64_t *max_idx, T *max_val, const T *vals, size_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    argmax_kernel<<<1, threads, 0, stream>>>(max_idx, max_val, vals, n);
    check_cuda(cudaGetLastError(), "CUDA argmax kernel launch failed");
}
} // namespace

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t n,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax(idx_ptr, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), n, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_argmax(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), n, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), n, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
