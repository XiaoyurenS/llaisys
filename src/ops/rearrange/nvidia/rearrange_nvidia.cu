#include "rearrange_nvidia.cuh"

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
} // namespace

void rearrange_contiguous(
    std::byte *out,
    const std::byte *in,
    size_t bytes,
    llaisysStream_t stream) {
    check_cuda(
        cudaMemcpyAsync(
            out,
            in,
            bytes,
            cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(stream)),
        "CUDA rearrange contiguous memcpy failed");
}
} // namespace llaisys::ops::nvidia
