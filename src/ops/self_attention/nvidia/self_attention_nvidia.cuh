#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
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
    llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
