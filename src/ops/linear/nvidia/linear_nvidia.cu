#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <unordered_map>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {
inline void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" + std::to_string(static_cast<int>(status)));
    }
}

inline cublasHandle_t get_cublas_handle(cudaStream_t stream) {
    thread_local std::unordered_map<int, cublasHandle_t> handles;

    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice failed");

    auto it = handles.find(device_id);
    if (it == handles.end()) {
        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate(&handle), "cublasCreate failed");
        it = handles.emplace(device_id, handle).first;
    }

    // handle 在 device 维度复用，但 stream 每次调用都要重新绑定到当前 runtime 的 stream。
    check_cublas(cublasSetStream(it->second, stream), "cublasSetStream failed");
    return it->second;
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
__global__ void add_bias_kernel(T *out, const T *bias, size_t m, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t numel = m * n;
    if (idx >= numel) {
        return;
    }

    const size_t col = idx % n;
    // GEMM 主体交给 cuBLAS，bias 作为单独的轻量 kernel 叠加，逻辑更清晰也更容易验证。
    out[idx] = from_float<T>(to_float(out[idx]) + to_float(bias[col]));
}

template <typename T>
void launch_bias_add(T *out, const T *bias, size_t m, size_t n, cudaStream_t stream) {
    if (bias == nullptr) {
        return;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((m * n + threads - 1) / threads);
    add_bias_kernel<<<blocks, threads, 0, stream>>>(out, bias, m, n);
    check_cuda(cudaGetLastError(), "CUDA linear bias add kernel launch failed");
}

void linear_f32(
    float *out,
    const float *in,
    const float *weight,
    const float *bias,
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream) {
    cublasHandle_t handle = get_cublas_handle(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 项目里的 tensor 是 row-major，而 cuBLAS 默认按 column-major 解释内存。
    // 这里利用 (X * W^T)^T = W * X^T，把 row-major GEMM 改写成 column-major GEMM。
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            weight,
            static_cast<int>(k),
            in,
            static_cast<int>(k),
            &beta,
            out,
            static_cast<int>(n)),
        "cublasSgemm failed");

    launch_bias_add(out, bias, m, n, stream);
}

template <typename T>
void linear_gemm_ex(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k,
    cudaDataType_t data_type,
    cudaStream_t stream) {
    cublasHandle_t handle = get_cublas_handle(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 半精度/bfloat16 输入直接交给 cuBLAS，计算类型统一提升到 fp32，优先保证正确性。
    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            weight,
            data_type,
            static_cast<int>(k),
            in,
            data_type,
            static_cast<int>(k),
            &beta,
            out,
            data_type,
            static_cast<int>(n),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx failed");

    launch_bias_add(out, bias, m, n, stream);
}
} // namespace

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k,
    llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_f32(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias ? reinterpret_cast<const float *>(bias) : nullptr,
            m,
            n,
            k,
            cuda_stream);
    case LLAISYS_DTYPE_F16:
        return linear_gemm_ex(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
            m,
            n,
            k,
            CUDA_R_16F,
            cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return linear_gemm_ex(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
            m,
            n,
            k,
            CUDA_R_16BF,
            cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
