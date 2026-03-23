#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rearrange_contiguous(
    std::byte *out,
    const std::byte *in,
    size_t bytes,
    llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
