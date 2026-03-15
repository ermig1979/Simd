/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        // BGR to Gray conversion: gray = (blue*1868 + green*9617 + red*4899 + 8192) >> 14
        // These are the Base:: weights: BLUE=1868, GREEN=9617, RED=4899, ROUND=8192

        SIMD_INLINE HVX_Vector BgrToGray16(HVX_Vector blue, HVX_Vector green, HVX_Vector red)
        {
            // blue, green, red are 16-bit unsigned per element
            // Compute: (blue*1868 + green*9617 + red*4899 + 8192) >> 14
            const int blueWeight = Base::BLUE_TO_GRAY_WEIGHT;
            const int greenWeight = Base::GREEN_TO_GRAY_WEIGHT;
            const int redWeight = Base::RED_TO_GRAY_WEIGHT;

            HVX_VectorPair prod_b = Q6_Wuw_vmpy_VuhRuh(blue, blueWeight);
            HVX_VectorPair prod_g = Q6_Wuw_vmpy_VuhRuh(green, greenWeight);
            HVX_VectorPair prod_r = Q6_Wuw_vmpy_VuhRuh(red, redWeight);

            HVX_Vector round = Q6_V_vsplat_R(Base::BGR_TO_GRAY_ROUND_TERM);

            HVX_Vector sum_lo = Q6_Vw_vadd_VwVw(Q6_V_lo_W(prod_b), Q6_V_lo_W(prod_g));
            sum_lo = Q6_Vw_vadd_VwVw(sum_lo, Q6_V_lo_W(prod_r));
            sum_lo = Q6_Vw_vadd_VwVw(sum_lo, round);

            HVX_Vector sum_hi = Q6_Vw_vadd_VwVw(Q6_V_hi_W(prod_b), Q6_V_hi_W(prod_g));
            sum_hi = Q6_Vw_vadd_VwVw(sum_hi, Q6_V_hi_W(prod_r));
            sum_hi = Q6_Vw_vadd_VwVw(sum_hi, round);

            // Shift right by 14
            sum_lo = Q6_Vuw_vlsr_VuwR(sum_lo, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
            sum_hi = Q6_Vuw_vlsr_VuwR(sum_hi, Base::BGR_TO_GRAY_AVERAGING_SHIFT);

            // Pack 32-bit to 16-bit
            return Q6_Vh_vshuffo_VhVh(sum_hi, sum_lo);
        }

        template <bool align> void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* gray, size_t grayStride)
        {
            assert(width >= A);

            // Process pixel-by-pixel using the base implementation for correctness
            // HVX doesn't have a convenient Load3 de-interleave, so use scalar
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    size_t off = col * 3;
                    uint32_t b = bgr[off + 0];
                    uint32_t g = bgr[off + 1];
                    uint32_t r = bgr[off + 2];
                    gray[col] = (uint8_t)((b * Base::BLUE_TO_GRAY_WEIGHT + g * Base::GREEN_TO_GRAY_WEIGHT +
                        r * Base::RED_TO_GRAY_WEIGHT + Base::BGR_TO_GRAY_ROUND_TERM) >> Base::BGR_TO_GRAY_AVERAGING_SHIFT);
                }
                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* gray, size_t grayStride)
        {
            BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }
    }
#endif
}
