/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        template <bool align> void GetStatistic(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint8_t* min, uint8_t* max, uint8_t* average)
        {
            assert(width * height && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);

            HVX_Vector _min = Q6_V_vsplat_R(0xFFFFFFFF);
            HVX_Vector _max = Q6_V_vsplat_R(0x00000000);
            uint64_t fullSum = 0;
            uint8_t tailMin = UCHAR_MAX, tailMax = 0;

            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                HVX_Vector rowSumH = Q6_V_vsplat_R(0);
                HVX_Vector rowSumH2 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    _min = Q6_Vub_vmin_VubVub(_min, _src);
                    _max = Q6_Vub_vmax_VubVub(_max, _src);
                    HVX_VectorPair sum16 = Q6_Wuh_vunpack_Vub(_src);
                    rowSumH = Q6_Vh_vadd_VhVh(rowSumH, Q6_V_lo_W(sum16));
                    rowSumH2 = Q6_Vh_vadd_VhVh(rowSumH2, Q6_V_hi_W(sum16));
                }
                // Widen 16 -> 32 for row sum
                HVX_VectorPair sum32a = Q6_Wuw_vunpack_Vuh(rowSumH);
                HVX_VectorPair sum32b = Q6_Wuw_vunpack_Vuh(rowSumH2);
                HVX_Vector s32 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32a), Q6_V_hi_W(sum32a));
                s32 = Q6_Vw_vadd_VwVw(s32, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32b), Q6_V_hi_W(sum32b)));

                SIMD_ALIGNED(128) uint32_t buf32[A / sizeof(uint32_t)];
                Store<true>((uint8_t*)buf32, s32);
                uint64_t rowTotal = 0;
                for (size_t i = 0; i < A / sizeof(uint32_t); ++i)
                    rowTotal += buf32[i];
                // Scalar tail to avoid double-counting overlapping bytes
                for (size_t col = alignedWidth; col < width; ++col)
                {
                    uint8_t v = src[col];
                    rowTotal += v;
                    if (v < tailMin) tailMin = v;
                    if (v > tailMax) tailMax = v;
                }
                fullSum += rowTotal;

                src += stride;
            }

            // Horizontal vector reduction for min/max (vlalign+vmin/vmax pattern)
            *min = Base::MinU8(HorizontalMinU8(_min), tailMin);
            *max = Base::MaxU8(HorizontalMaxU8(_max), tailMax);
            *average = (uint8_t)((fullSum + width * height / 2) / (width * height));
        }

        void GetStatistic(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint8_t* min, uint8_t* max, uint8_t* average)
        {
            if (Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }
    }
#endif
}
