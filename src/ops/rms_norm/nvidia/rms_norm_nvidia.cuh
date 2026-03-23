#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t m,
    size_t d,
    float eps,
    llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
