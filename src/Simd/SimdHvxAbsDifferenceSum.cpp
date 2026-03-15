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
        template <bool align> void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = AlignLo(width, A);
            uint64_t totalSum = 0;

            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                {
                    L2Prefetch(a + aStride, aStride, width, 1);
                    L2Prefetch(b + bStride, bStride, width, 1);
                }

                HVX_Vector rowSumH = Q6_V_vsplat_R(0);
                HVX_Vector rowSumH2 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector a_ = Load<align>(a + col);
                    const HVX_Vector b_ = Load<align>(b + col);
                    const HVX_Vector ad = Q6_Vub_vabsdiff_VubVub(a_, b_);
                    HVX_VectorPair sum16 = Q6_Wuh_vunpack_Vub(ad);
                    rowSumH = Q6_Vh_vadd_VhVh(rowSumH, Q6_V_lo_W(sum16));
                    rowSumH2 = Q6_Vh_vadd_VhVh(rowSumH2, Q6_V_hi_W(sum16));
                }
                // Horizontal reduction: 16-bit -> 32-bit -> scalar
                HVX_VectorPair sum32a = Q6_Wuw_vunpack_Vuh(rowSumH);
                HVX_VectorPair sum32b = Q6_Wuw_vunpack_Vuh(rowSumH2);
                HVX_Vector s32 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32a), Q6_V_hi_W(sum32a));
                s32 = Q6_Vw_vadd_VwVw(s32, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32b), Q6_V_hi_W(sum32b)));

                // Extract and sum all 32-bit lanes to scalar
                SIMD_ALIGNED(128) uint32_t buf[A / sizeof(uint32_t)];
                Store<true>((uint8_t*)buf, s32);
                uint64_t rowTotal = 0;
                for (size_t i = 0; i < A / sizeof(uint32_t); ++i)
                    rowTotal += buf[i];
                // Scalar tail to avoid double-counting overlapping bytes
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += a[col] > b[col] ? a[col] - b[col] : b[col] - a[col];
                totalSum += rowTotal;

                a += aStride;
                b += bStride;
            }
            *sum = totalSum;
        }

        void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif
}
