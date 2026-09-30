/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#ifndef __SimdHvx_h__
#define __SimdHvx_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t* histogram);
    }
#endif
}
#endif
