#include "rms_norm_nvidia.cuh"

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
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t d, float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);

    const T *in_row = in + row * d;
    T *out_row = out + row * d;

    __shared__ float partial_sum[256];

    float local_sum = 0.0f;
    for (size_t col = tid; col < d; col += blockDim.x) {
        const float v = to_float(in_row[col]);
        local_sum += v * v;
    }
    partial_sum[tid] = local_sum;
    __syncthreads();

    // 先对每一行做平方和归约，得到 RMSNorm 所需的均方值。
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    const float denom = rsqrtf(partial_sum[0] / static_cast<float>(d) + eps);

    // 第二遍把输入乘以归一化系数和逐维 weight，写回输出。
    for (size_t col = tid; col < d; col += blockDim.x) {
        const float v = to_float(in_row[col]);
        const float w = to_float(weight[col]);
        out_row[col] = from_float<T>(v * denom * w);
    }
}

template <typename T>
void launch_rms_norm(T *out, const T *in, const T *weight, size_t m, size_t d, float eps, cudaStream_t stream) {
    constexpr int threads = 256;
    rms_norm_kernel<<<static_cast<int>(m), threads, 0, stream>>>(out, in, weight, d, eps);
    check_cuda(cudaGetLastError(), "CUDA rms_norm kernel launch failed");
}
} // namespace

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t m,
    size_t d,
    float eps,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rms_norm(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            m,
            d,
            eps,
            cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_rms_norm(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            m,
            d,
            eps,
            cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_rms_norm(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            m,
            d,
            eps,
            cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
