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
        SIMD_INLINE HVX_Vector FeatureDifference(HVX_Vector value, HVX_Vector lo, HVX_Vector hi)
        {
            return Q6_Vub_vmax_VubVub(Q6_Vub_vsub_VubVub_sat(value, hi), Q6_Vub_vsub_VubVub_sat(lo, value));
        }

        SIMD_INLINE HVX_Vector ShiftedWeightedSquare(HVX_Vector diff, int weight)
        {
            // diff^2 as 16-bit unsigned: even/odd byte products
            HVX_VectorPair sq = Q6_Wuh_vmpy_VubVub(diff, diff);
            HVX_Vector sq_even = Q6_V_lo_W(sq);
            HVX_Vector sq_odd = Q6_V_hi_W(sq);

            // Multiply 16-bit squares by weight -> 32-bit results
            // vmpy_VuhRuh uses Rt.uh[0] for even and Rt.uh[1] for odd halfwords,
            // so pack weight into both halfword lanes.
            int w2 = weight | (weight << 16);
            HVX_VectorPair pe = Q6_Wuw_vmpy_VuhRuh(sq_even, w2);
            HVX_VectorPair po = Q6_Wuw_vmpy_VuhRuh(sq_odd, w2);

            // Shift 32-bit products right by 16 bits
            HVX_Vector pe_lo = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(pe), 16);
            HVX_Vector pe_hi = Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(pe), 16);
            HVX_Vector po_lo = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(po), 16);
            HVX_Vector po_hi = Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(po), 16);

            // Pack 32-bit -> 16-bit with unsigned saturation (interleaved)
            HVX_Vector r_even = Q6_Vuh_vsat_VuwVuw(pe_hi, pe_lo);
            HVX_Vector r_odd = Q6_Vuh_vsat_VuwVuw(po_hi, po_lo);

            // Clamp 16-bit values to 255 before extracting low bytes
            HVX_Vector max255 = Q6_V_vsplat_R(0x00FF00FF);
            r_even = Q6_Vuh_vmin_VuhVuh(r_even, max255);
            r_odd = Q6_Vuh_vmin_VuhVuh(r_odd, max255);

            return Q6_Vb_vshuffe_VbVb(r_odd, r_even);
        }

        template <bool align> void AddFeatureDifference(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride,
            uint16_t weight, uint8_t* difference, size_t differenceStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
                assert(Aligned(difference) && Aligned(differenceStride));
            }

            size_t alignedWidth = AlignLo(width, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _value = Load<align>(value + col);
                    const HVX_Vector _lo = Load<align>(lo + col);
                    const HVX_Vector _hi = Load<align>(hi + col);
                    HVX_Vector _difference = Load<align>(difference + col);

                    const HVX_Vector fd = FeatureDifference(_value, _lo, _hi);
                    const HVX_Vector inc = ShiftedWeightedSquare(fd, weight);
                    Store<align>(difference + col, Q6_Vub_vadd_VubVub_sat(_difference, inc));
                }
                for (size_t col = alignedWidth; col < width; ++col)
                {
                    int v = value[col], l = lo[col], h = hi[col];
                    int fd1 = (v > h) ? (v - h) : 0;
                    int fd2 = (l > v) ? (l - v) : 0;
                    int fd = (fd1 > fd2) ? fd1 : fd2;
                    int inc = (fd * fd * weight) >> 16;
                    int sum = difference[col] + inc;
                    difference[col] = (uint8_t)(sum > 255 ? 255 : sum);
                }
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                difference += differenceStride;
            }
        }

        void AddFeatureDifference(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride,
            uint16_t weight, uint8_t* difference, size_t differenceStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) &&
                Aligned(hi) && Aligned(hiStride) && Aligned(difference) && Aligned(differenceStride))
                AddFeatureDifference<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
            else
                AddFeatureDifference<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
        }
    }
#endif
}
