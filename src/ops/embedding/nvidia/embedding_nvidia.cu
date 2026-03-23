#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

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

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t vocab, size_t dim, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t col = idx % dim;
    const size_t row = idx / dim;
    const int64_t emb_row = index[row];

    if (emb_row < 0 || static_cast<size_t>(emb_row) >= vocab) {
        return;
    }

    out[row * dim + col] = weight[static_cast<size_t>(emb_row) * dim + col];
}

template <typename T>
void launch_embedding(T *out, const int64_t *index, const T *weight, size_t n, size_t vocab, size_t dim, cudaStream_t stream) {
    constexpr int threads = 256;
    const size_t total = n * dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    embedding_kernel<<<blocks, threads, 0, stream>>>(out, index, weight, vocab, dim, total);
    check_cuda(cudaGetLastError(), "CUDA embedding kernel launch failed");
}
} // namespace

void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t n,
    size_t vocab,
    size_t dim,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding(
            reinterpret_cast<float *>(out),
            index,
            reinterpret_cast<const float *>(weight),
            n,
            vocab,
            dim,
            cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launch_embedding(
            reinterpret_cast<llaisys::fp16_t *>(out),
            index,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            n,
            vocab,
            dim,
            cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding(
            reinterpret_cast<llaisys::bf16_t *>(out),
            index,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            n,
            vocab,
            dim,
            cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
