#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace {
inline void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

inline cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported memcpy kind for CUDA runtime.");
    }
}
} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        cudaGetLastError(); // 清理 CUDA runtime 的最后一次错误状态。
        return 0;
    }
    check_cuda(status, "cudaGetDeviceCount failed");
    return count;
}

void setDevice(int device_id) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
}

void deviceSynchronize() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    // 为每个 Runtime 创建独立 stream，后续异步 memcpy / kernel 都走这个 stream。
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    check_cuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy failed");
}
void streamSynchronize(llaisysStream_t stream) {
    if (stream == nullptr) {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        return;
    }
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize failed");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, size), "cudaMalloc failed");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFree(ptr), "cudaFree failed");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // 使用 pinned host memory，便于后续 H2D / D2H 拷贝加速。
    check_cuda(cudaMallocHost(&ptr, size), "cudaMallocHost failed");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFreeHost(ptr), "cudaFreeHost failed");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check_cuda(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)), "cudaMemcpy failed");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check_cuda(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            to_cuda_memcpy_kind(kind),
            reinterpret_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync failed");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
