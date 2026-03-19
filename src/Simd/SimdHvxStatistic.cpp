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

        //-----------------------------------------------------------------------

        SIMD_INLINE uint64_t HorizontalSumU32(HVX_Vector v)
        {
            SIMD_ALIGNED(128) uint32_t buf[A / sizeof(uint32_t)];
            Store<true>((uint8_t*)buf, v);
            uint64_t sum = 0;
            for (size_t i = 0; i < A / sizeof(uint32_t); ++i)
                sum += buf[i];
            return sum;
        }

        SIMD_INLINE HVX_Vector WidenSumU8toU32(HVX_Vector src)
        {
            HVX_VectorPair sum16 = Q6_Wuh_vunpack_Vub(src);
            HVX_VectorPair lo32 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(sum16));
            HVX_VectorPair hi32 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(sum16));
            HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_V_lo_W(lo32), Q6_V_hi_W(lo32));
            return Q6_Vw_vadd_VwVw(s, Q6_Vw_vadd_VwVw(Q6_V_lo_W(hi32), Q6_V_hi_W(hi32)));
        }

        //-----------------------------------------------------------------------

        template <bool align> void GetRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);

            memset(sums, 0, sizeof(uint32_t) * height);
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                HVX_Vector rowSumH = Q6_V_vsplat_R(0);
                HVX_Vector rowSumH2 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    HVX_VectorPair sum16 = Q6_Wuh_vunpack_Vub(_src);
                    rowSumH = Q6_Vh_vadd_VhVh(rowSumH, Q6_V_lo_W(sum16));
                    rowSumH2 = Q6_Vh_vadd_VhVh(rowSumH2, Q6_V_hi_W(sum16));
                }
                HVX_VectorPair sum32a = Q6_Wuw_vunpack_Vuh(rowSumH);
                HVX_VectorPair sum32b = Q6_Wuw_vunpack_Vuh(rowSumH2);
                HVX_Vector s32 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32a), Q6_V_hi_W(sum32a));
                s32 = Q6_Vw_vadd_VwVw(s32, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sum32b), Q6_V_hi_W(sum32b)));

                uint64_t rowTotal = HorizontalSumU32(s32);
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += src[col];
                sums[row] = (uint32_t)rowTotal;

                src += stride;
            }
        }

        void GetRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void AccumulateU8toU32(const HVX_Vector& src, uint32_t* dst)
        {
            HVX_VectorPair u16pair = Q6_Wuh_vunpack_Vub(src);
            HVX_VectorPair u32_0 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(u16pair));
            HVX_VectorPair u32_1 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(u16pair));
            Store<false>((uint8_t*)(dst + 0), Q6_Vw_vadd_VwVw(Load<false>((const uint8_t*)(dst + 0)), Q6_V_lo_W(u32_0)));
            Store<false>((uint8_t*)(dst + A / 4), Q6_Vw_vadd_VwVw(Load<false>((const uint8_t*)(dst + A / 4)), Q6_V_hi_W(u32_0)));
            Store<false>((uint8_t*)(dst + A / 2), Q6_Vw_vadd_VwVw(Load<false>((const uint8_t*)(dst + A / 2)), Q6_V_lo_W(u32_1)));
            Store<false>((uint8_t*)(dst + 3 * A / 4), Q6_Vw_vadd_VwVw(Load<false>((const uint8_t*)(dst + 3 * A / 4)), Q6_V_hi_W(u32_1)));
        }

        template <bool align> void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);

            uint32_t* buf32 = (uint32_t*)Allocate(sizeof(uint32_t) * alignedHiWidth);
            memset(buf32, 0, sizeof(uint32_t) * alignedHiWidth);

            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    AccumulateU8toU32(_src, buf32 + col);
                }
                for (size_t col = alignedWidth; col < width; ++col)
                    buf32[col] += src[col];
                src += stride;
            }
            memcpy(sums, buf32, sizeof(uint32_t) * width);
            Free(buf32);
        }

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);

            memset(sums, 0, sizeof(uint32_t) * height);
            const uint8_t* src0 = src;
            const uint8_t* src1 = src + stride;
            height--;
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src1 + stride, stride, width, 1);

                HVX_Vector rowSum32 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src0 = Load<align>(src0 + col);
                    const HVX_Vector _src1 = Load<align>(src1 + col);
                    HVX_Vector absDiff = Q6_Vub_vabsdiff_VubVub(_src0, _src1);
                    rowSum32 = Q6_Vw_vadd_VwVw(rowSum32, WidenSumU8toU32(absDiff));
                }

                uint64_t rowTotal = HorizontalSumU32(rowSum32);
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += Base::AbsDifferenceU8(src0[col], src1[col]);
                sums[row] = (uint32_t)rowTotal;

                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);

            uint32_t* buf32 = (uint32_t*)Allocate(sizeof(uint32_t) * alignedHiWidth);
            memset(buf32, 0, sizeof(uint32_t) * alignedHiWidth);

            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width + 1, 1);

                for (size_t col = 0; col < alignedLoWidth; col += A)
                {
                    const HVX_Vector _src0 = Load<align>(src + col + 0);
                    const HVX_Vector _src1 = Load<false>(src + col + 1);
                    HVX_Vector absDiff = Q6_Vub_vabsdiff_VubVub(_src0, _src1);
                    AccumulateU8toU32(absDiff, buf32 + col);
                }
                for (size_t col = alignedLoWidth; col < width; ++col)
                    buf32[col] += Base::AbsDifferenceU8(src[col], src[col + 1]);
                src += stride;
            }
            memcpy(sums, buf32, sizeof(uint32_t) * width);
            sums[width] = 0;
            Free(buf32);
        }

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            uint64_t fullSum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                HVX_Vector rowSum32 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    rowSum32 = Q6_Vw_vadd_VwVw(rowSum32, WidenSumU8toU32(_src));
                }
                uint64_t rowTotal = HorizontalSumU32(rowSum32);
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += src[col];
                fullSum += rowTotal;
                src += stride;
            }
            *sum = fullSum;
        }

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------

        template <bool align> void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            uint64_t fullSum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                HVX_Vector rowSum32 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    // Q6_Wuh_vmpy_VubVub: even/odd u8 products -> interleaved u16 pairs
                    HVX_VectorPair prod16 = Q6_Wuh_vmpy_VubVub(_src, _src);
                    // Widen each u16 half to u32 and accumulate
                    HVX_VectorPair lo32 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(prod16));
                    HVX_VectorPair hi32 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(prod16));
                    HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_V_lo_W(lo32), Q6_V_hi_W(lo32));
                    s = Q6_Vw_vadd_VwVw(s, Q6_Vw_vadd_VwVw(Q6_V_lo_W(hi32), Q6_V_hi_W(hi32)));
                    rowSum32 = Q6_Vw_vadd_VwVw(rowSum32, s);
                }
                uint64_t rowTotal = HorizontalSumU32(rowSum32);
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += (uint64_t)src[col] * src[col];
                fullSum += rowTotal;
                src += stride;
            }
            *sum = fullSum;
        }

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------

        template <bool align> void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint64_t* valueSum, uint64_t* squareSum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            uint64_t fullValueSum = 0, fullSquareSum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                    L2Prefetch(src + stride, stride, width, 1);

                HVX_Vector rowValueSum32 = Q6_V_vsplat_R(0);
                HVX_Vector rowSquareSum32 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _src = Load<align>(src + col);
                    // Value sum
                    rowValueSum32 = Q6_Vw_vadd_VwVw(rowValueSum32, WidenSumU8toU32(_src));
                    // Square sum
                    HVX_VectorPair prod16 = Q6_Wuh_vmpy_VubVub(_src, _src);
                    HVX_VectorPair lo32 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(prod16));
                    HVX_VectorPair hi32 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(prod16));
                    HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_V_lo_W(lo32), Q6_V_hi_W(lo32));
                    s = Q6_Vw_vadd_VwVw(s, Q6_Vw_vadd_VwVw(Q6_V_lo_W(hi32), Q6_V_hi_W(hi32)));
                    rowSquareSum32 = Q6_Vw_vadd_VwVw(rowSquareSum32, s);
                }
                uint64_t rowValTotal = HorizontalSumU32(rowValueSum32);
                uint64_t rowSqTotal = HorizontalSumU32(rowSquareSum32);
                for (size_t col = alignedWidth; col < width; ++col)
                {
                    rowValTotal += src[col];
                    rowSqTotal += (uint64_t)src[col] * src[col];
                }
                fullValueSum += rowValTotal;
                fullSquareSum += rowSqTotal;
                src += stride;
            }
            *valueSum = fullValueSum;
            *squareSum = fullSquareSum;
        }

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint64_t* valueSum, uint64_t* squareSum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSum<true>(src, stride, width, height, valueSum, squareSum);
            else
                ValueSquareSum<false>(src, stride, width, height, valueSum, squareSum);
        }

        //-----------------------------------------------------------------------

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height,
            size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(channels >= 1 && channels <= 4);
            if (channels == 1)
                ValueSquareSum(src, stride, width, height, valueSums, squareSums);
            else
                Base::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = AlignLo(width, A);
            uint64_t fullSum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                if (row + 1 < height)
                {
                    L2Prefetch(a + aStride, aStride, width, 1);
                    L2Prefetch(b + bStride, bStride, width, 1);
                }

                HVX_Vector rowSum32 = Q6_V_vsplat_R(0);

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const HVX_Vector _a = Load<align>(a + col);
                    const HVX_Vector _b = Load<align>(b + col);
                    HVX_VectorPair prod16 = Q6_Wuh_vmpy_VubVub(_a, _b);
                    HVX_VectorPair lo32 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(prod16));
                    HVX_VectorPair hi32 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(prod16));
                    HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_V_lo_W(lo32), Q6_V_hi_W(lo32));
                    s = Q6_Vw_vadd_VwVw(s, Q6_Vw_vadd_VwVw(Q6_V_lo_W(hi32), Q6_V_hi_W(hi32)));
                    rowSum32 = Q6_Vw_vadd_VwVw(rowSum32, s);
                }
                uint64_t rowTotal = HorizontalSumU32(rowSum32);
                for (size_t col = alignedWidth; col < width; ++col)
                    rowTotal += (uint64_t)a[col] * b[col];
                fullSum += rowTotal;
                a += aStride;
                b += bStride;
            }
            *sum = fullSum;
        }

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif
}
