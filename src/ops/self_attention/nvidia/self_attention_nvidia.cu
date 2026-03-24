#include "self_attention_nvidia.cuh"

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
__global__ void self_attention_kernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t seqlen,
    size_t nhead,
    size_t d,
    size_t total_len,
    size_t nkvhead,
    size_t dv,
    float scale) {
    const size_t idx = static_cast<size_t>(blockIdx.x);
    const size_t total_queries = seqlen * nhead;
    if (idx >= total_queries) {
        return;
    }
    const size_t tid = static_cast<size_t>(threadIdx.x);

    const size_t i = idx / nhead;
    const size_t h = idx % nhead;
    const size_t kv_repeat = nhead / nkvhead;
    const size_t kvh = h / kv_repeat;

    int64_t past_len = static_cast<int64_t>(total_len) - static_cast<int64_t>(seqlen);
    if (past_len < 0) {
        past_len = 0;
    }

    size_t max_t = static_cast<size_t>(past_len) + i;
    if (max_t >= total_len) {
        max_t = total_len - 1;
    }

    const size_t q_base = (i * nhead + h) * d;
    const size_t out_base = (i * nhead + h) * dv;
    extern __shared__ float scores[];

    // 每个线程负责一部分 t，先并行算出 attention score 并写入 shared memory。
    for (size_t t = tid; t <= max_t; t += blockDim.x) {
        const size_t k_base = (t * nkvhead + kvh) * d;
        float score = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            score += to_float(q[q_base + j]) * to_float(k[k_base + j]);
        }
        scores[t] = score * scale;
    }
    __syncthreads();

    if (tid == 0) {
        // softmax 仍然由单线程完成，但前面最重的 qk 打分已经并行化了。
        float max_score = -1.0e30f;
        for (size_t t = 0; t <= max_t; ++t) {
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        float sum = 0.0f;
        for (size_t t = 0; t <= max_t; ++t) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }

        const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
        for (size_t t = 0; t <= max_t; ++t) {
            scores[t] *= inv_sum;
        }
    }
    __syncthreads();

    // 输出维度再在线程间分摊，每个线程负责若干个 dv 位置。
    for (size_t j = tid; j < dv; j += blockDim.x) {
        float acc = 0.0f;
        for (size_t t = 0; t <= max_t; ++t) {
            const size_t v_base = (t * nkvhead + kvh) * dv;
            acc += scores[t] * to_float(v[v_base + j]);
        }
        out[out_base + j] = from_float<T>(acc);
    }
}

template <typename T>
void launch_self_attention(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t seqlen,
    size_t nhead,
    size_t d,
    size_t total_len,
    size_t nkvhead,
    size_t dv,
    float scale,
    cudaStream_t stream) {
    constexpr int threads = 128;
    const size_t total_queries = seqlen * nhead;
    const int blocks = static_cast<int>(total_queries);
    const size_t shared_bytes = total_len * sizeof(float);
    self_attention_kernel<<<blocks, threads, shared_bytes, stream>>>(
        out, q, k, v, seqlen, nhead, d, total_len, nkvhead, dv, scale);
    check_cuda(cudaGetLastError(), "CUDA self_attention kernel launch failed");
}
} // namespace

void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t seqlen,
    size_t nhead,
    size_t d,
    size_t total_len,
    size_t nkvhead,
    size_t dv,
    float scale,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, nhead, d, total_len, nkvhead, dv, scale, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            seqlen, nhead, d, total_len, nkvhead, dv, scale, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            seqlen, nhead, d, total_len, nkvhead, dv, scale, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
