#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t n,
    size_t vocab,
    size_t dim,
    llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
