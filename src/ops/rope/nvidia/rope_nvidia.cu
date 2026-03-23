#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = seqlen * nhead * half;
    if (idx >= total) {
        return;
    }

    const size_t j = idx % half;
    const size_t tmp = idx / half;
    const size_t h = tmp % nhead;
    const size_t i = tmp / nhead;

    const size_t base = (i * nhead + h) * d;

    // 每个线程负责一对旋转维度 (j, j + half)，避免线程间写冲突。
    const float pos = static_cast<float>(pos_ids[i]);
    const float inv_freq = powf(theta, (2.0f * static_cast<float>(j)) / static_cast<float>(d));
    const float phi = pos / inv_freq;
    const float c = cosf(phi);
    const float s = sinf(phi);

    const float a = to_float(in[base + j]);
    const float b = to_float(in[base + j + half]);
    out[base + j] = from_float<T>(a * c - b * s);
    out[base + j + half] = from_float<T>(b * c + a * s);
}

template <typename T>
void launch_rope(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta, cudaStream_t stream) {
    const size_t half = d / 2;
    const size_t total = seqlen * nhead * half;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    rope_kernel<<<blocks, threads, 0, stream>>>(out, in, pos_ids, seqlen, nhead, d, theta);
    check_cuda(cudaGetLastError(), "CUDA rope kernel launch failed");
}
} // namespace

void rope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t type,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rope(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids,
            seqlen,
            nhead,
            d,
            theta,
            cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_rope(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids,
            seqlen,
            nhead,
            d,
            theta,
            cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_rope(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids,
            seqlen,
            nhead,
            d,
            theta,
            cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
