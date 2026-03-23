// 这个文件本身不做任何计算。
// 它的唯一作用是让最终的 shared library target 含有至少一个 .cu 源文件，
// 从而触发 xmake/nvcc 对整个目标执行 device link。
// 否则来自静态库（例如 src/ops/add/nvidia/*.cu）的 device code 可能只被编译，
// 但不会在最终 .so 中完成 __cudaRegisterLinkedBinary 注册。

__global__ void llaisys_cuda_stub_kernel() {}
