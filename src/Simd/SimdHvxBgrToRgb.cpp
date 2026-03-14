/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        template <bool align> void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);

            // BGR to RGB: swap B and R channels in each 3-byte pixel
            // HVX doesn't have a convenient deinterleave-3, so use scalar loop
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    size_t off = col * 3;
                    rgb[off + 0] = bgr[off + 2];
                    rgb[off + 1] = bgr[off + 1];
                    rgb[off + 2] = bgr[off + 0];
                }
                bgr += bgrStride;
                rgb += rgbStride;
            }
        }

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* rgb, size_t rgbStride)
        {
            BgrToRgb<false>(bgr, width, height, bgrStride, rgb, rgbStride);
        }
    }
#endif
}
