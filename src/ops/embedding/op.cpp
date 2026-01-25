#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be int64");
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding: weight must be 2D");
    CHECK_ARGUMENT(index->ndim() == 1, "embedding: index must be 1D");
    CHECK_ARGUMENT(out->ndim() == 2, "embedding: out must be 2D");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "embedding: out dim0 must match index length");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "embedding: out dim1 must match weight dim1");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous.");

    const auto *idx_ptr = reinterpret_cast<const int64_t *>(index->data());
    size_t n = index->shape()[0];
    size_t dim = weight->shape()[1];
    size_t elem_size = out->elementSize();

    const std::byte *w_ptr = weight->data();
    std::byte *o_ptr = out->data();

    for (size_t i = 0; i < n; ++i) {
        int64_t row = idx_ptr[i];
        CHECK_ARGUMENT(row >= 0 && static_cast<size_t>(row) < weight->shape()[0],
                       "embedding: index out of range");
        const std::byte *src = w_ptr + static_cast<size_t>(row) * dim * elem_size;
        std::byte *dst = o_ptr + i * dim * elem_size;
        std::memcpy(dst, src, dim * elem_size);
    }
}
} // namespace llaisys::ops
