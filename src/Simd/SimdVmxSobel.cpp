/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool abs> SIMD_INLINE void SobelDx(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum(
                vec_sub(UnpackLoU8(a[0][2]), UnpackLoU8(a[0][0])),
                vec_sub(UnpackLoU8(a[1][2]), UnpackLoU8(a[1][0])),
                vec_sub(UnpackLoU8(a[2][2]), UnpackLoU8(a[2][0]))));
            hi = ConditionalAbs<abs>(BinomialSum(
                vec_sub(UnpackHiU8(a[0][2]), UnpackHiU8(a[0][0])),
                vec_sub(UnpackHiU8(a[1][2]), UnpackHiU8(a[1][0])),
                vec_sub(UnpackHiU8(a[2][2]), UnpackHiU8(a[2][0]))));
        }

        template<bool align, bool first, bool abs> SIMD_INLINE void SobelDx(v128_u8 a[3][3], Storer<align> & dst)
        {
            v128_u16 lo, hi;
            SobelDx<abs>(a, lo, hi);
            Store<align, first>(dst, lo);
            Store<align, false>(dst, hi);
        }

        template <bool align, bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            v128_u8 a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                Storer<align> _dst(dst);
                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDx<align, true, abs>(a, _dst);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDx<align, false, abs>(a, _dst);
                }
                Flush(_dst);

                {
                    Storer<false> _dst(dst + width - A);
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src1 + width - A, a[1]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    SobelDx<false, true, abs>(a, _dst);
                    Flush(_dst);
                }

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDx<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDx<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDx<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDx<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDxAbsSum(v128_u8 a[3][3], v128_u32 & sum)
        {
            v128_u16 lo, hi;
            SobelDx<true>(a, lo, hi);
            sum = vec_msum(lo, K16_0001, sum);
            sum = vec_msum(hi, K16_0001, sum);
        }

        SIMD_INLINE void SetMask3(v128_u8 a[3], v128_u8 mask)
        {
            a[0] = vec_and(a[0], mask);
            a[1] = vec_and(a[1], mask);
            a[2] = vec_and(a[2], mask);
        }

        SIMD_INLINE void SetMask3x3(v128_u8 a[3][3], v128_u8 mask)
        {
            SetMask3(a[0], mask);
            SetMask3(a[1], mask);
            SetMask3(a[2], mask);
        }

        template <bool align> void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            v128_u8 a[3][3];
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            *sum = 0;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                v128_u32 rowSum = K32_00000000;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDxAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDxAbsSum(a, rowSum);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                SobelDxAbsSum(a, rowSum);

                *sum += ExtractSum(rowSum);
            }
        }

        void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SobelDxAbsSum<true>(src, stride, width, height, sum);
            else
                SobelDxAbsSum<false>(src, stride, width, height, sum);
        }

        template <bool abs> SIMD_INLINE void SobelDy(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum(
                vec_sub(UnpackLoU8(a[2][0]), UnpackLoU8(a[0][0])),
                vec_sub(UnpackLoU8(a[2][1]), UnpackLoU8(a[0][1])),
                vec_sub(UnpackLoU8(a[2][2]), UnpackLoU8(a[0][2]))));
            hi = ConditionalAbs<abs>(BinomialSum(
                vec_sub(UnpackHiU8(a[2][0]), UnpackHiU8(a[0][0])),
                vec_sub(UnpackHiU8(a[2][1]), UnpackHiU8(a[0][1])),
                vec_sub(UnpackHiU8(a[2][2]), UnpackHiU8(a[0][2]))));
        }

        template<bool align, bool first, bool abs> SIMD_INLINE void SobelDy(v128_u8 a[3][3], Storer<align> & dst)
        {
            v128_u16 lo, hi;
            SobelDy<abs>(a, lo, hi);
            Store<align, first>(dst, lo);
            Store<align, false>(dst, hi);
        }

        template <bool align, bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            v128_u8 a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                Storer<align> _dst(dst);
                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDy<align, true, abs>(a, _dst);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDy<align, false, abs>(a, _dst);
                }
                Flush(_dst);

                {
                    Storer<false> _dst(dst + width - A);
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    SobelDy<false, true, abs>(a, _dst);
                    Flush(_dst);
                }

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDy<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDy<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDyAbsSum(v128_u8 a[3][3], v128_u32 & sum)
        {
            v128_u16 lo, hi;
            SobelDy<true>(a, lo, hi);
            sum = vec_msum(lo, K16_0001, sum);
            sum = vec_msum(hi, K16_0001, sum);
        }

        template <bool align> void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            v128_u8 a[3][3];
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            *sum = 0;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                v128_u32 rowSum = K32_00000000;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDyAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDyAbsSum(a, rowSum);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                SobelDyAbsSum(a, rowSum);

                *sum += ExtractSum(rowSum);
            }
        }

        void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SobelDyAbsSum<true>(src, stride, width, height, sum);
            else
                SobelDyAbsSum<false>(src, stride, width, height, sum);
        }

        SIMD_INLINE v128_u16 ContourMetrics(v128_u16 dx, v128_u16 dy)
        {
            return vec_or(vec_sl(vec_add(dx, dy), K16_0001), vec_and(vec_cmplt(dx, dy), K16_0001));
        }

        SIMD_INLINE void ContourMetrics(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            v128_u16 dxLo, dxHi, dyLo, dyHi;
            SobelDx<true>(a, dxLo, dxHi);
            SobelDy<true>(a, dyLo, dyHi);
            lo = ContourMetrics(dxLo, dyLo);
            hi = ContourMetrics(dxHi, dyHi);
        }

        template<bool align, bool first> SIMD_INLINE void ContourMetrics(v128_u8 a[3][3], Storer<align> & dst)
        {
            v128_u16 lo, hi;
            ContourMetrics(a, lo, hi);
            Store<align, first>(dst, lo);
            Store<align, false>(dst, hi);
        }

        template <bool align> void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            v128_u8 a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                Storer<align> _dst(dst);
                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                ContourMetrics<align, true>(a, _dst);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    ContourMetrics<align, false>(a, _dst);
                }
                Flush(_dst);

                {
                    Storer<false> _dst(dst + width - A);
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src1 + width - A, a[1]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    ContourMetrics<false, true>(a, _dst);
                    Flush(_dst);
                }

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourMetrics<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                ContourMetrics<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align, bool first> SIMD_INLINE void ContourMetricsMasked(v128_u8 a[3][3], const uint8_t * mask, const v128_u8 & indexMin, Storer<align> & dst)
        {
            v128_u8 m = GreaterOrEqual(Load<align>(mask), indexMin);
            v128_u16 lo, hi;
            ContourMetrics(a, lo, hi);
            Store<align, first>(dst, vec_and(lo, UnpackLoU8(m, m)));
            Store<align, false>(dst, vec_and(hi, UnpackHiU8(m, m)));
        }

        template <bool align> void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            v128_u8 _indexMin = SetU8(indexMin);
            v128_u8 a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                Storer<align> _dst(dst);
                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                ContourMetricsMasked<align, true>(a, mask, _indexMin, _dst);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    ContourMetricsMasked<align, false>(a, mask + col, _indexMin, _dst);
                }
                Flush(_dst);

                {
                    Storer<false> _dst(dst + width - A);
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src1 + width - A, a[1]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    ContourMetricsMasked<false, true>(a, mask + width - A, _indexMin, _dst);
                    Flush(_dst);
                }

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride) && Aligned(mask) && Aligned(maskStride))
                ContourMetricsMasked<true>(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                ContourMetricsMasked<false>(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE v128_u16 AnchorComponent(const int16_t * src, size_t step, const v128_s16 & current, const v128_s16 & threshold, const v128_u16 & mask)
        {
            v128_s16 last = vec_sr(Load<align>(src - step), K16_0001);
            v128_s16 next = vec_sr(Load<align>(src + step), K16_0001);
            return vec_and(vec_xor((v128_u16)vec_or(vec_cmplt(vec_sub(current, last), threshold), vec_cmplt(vec_sub(current, next), threshold)), K16_FFFF), mask);
        }

        template<bool align> SIMD_INLINE v128_u16 Anchor(const int16_t * src, size_t stride, const v128_s16 & threshold)
        {
            v128_s16 _src = Load<align>(src);
            v128_u16 direction = vec_and((v128_u16)_src, K16_0001);
            v128_s16 magnitude = vec_sr(_src, K16_0001);
            v128_u16 vertical = AnchorComponent<false>(src, 1, magnitude, threshold, (v128_u16)vec_cmpeq(direction, K16_0001));
            v128_u16 horizontal = AnchorComponent<align>(src, stride, magnitude, threshold, (v128_u16)vec_cmpeq(direction, K16_0000));
            return vec_and(vec_xor(vec_cmpeq((v128_u16)magnitude, K16_0000), K16_FFFF), vec_and(vec_or(vertical, horizontal), K16_00FF));
        }

        template<bool align> SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const v128_s16 & threshold, uint8_t * dst)
        {
            v128_u16 lo = Anchor<align>(src, stride, threshold);
            v128_u16 hi = Anchor<align>(src + HA, stride, threshold);
            Store<align>(dst, vec_pack(lo, hi));
        }

        template<bool align, bool first> SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const v128_s16 & threshold, Storer<align> & dst)
        {
            v128_u16 lo = Anchor<align>(src, stride, threshold);
            v128_u16 hi = Anchor<align>(src + HA, stride, threshold);
            Store<align, first>(dst, vec_pack(lo, hi));
        }

        template <bool align> void ContourAnchors(const int16_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride, HA) && Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            v128_s16 _threshold = SetI16(threshold);
            memset(dst, 0, width);
            memset(dst + dstStride*(height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                Anchor<false>(src + 1, srcStride, _threshold, dst + 1);
                if (bodyWidth >= DA)
                {
                    Storer<align> _dst(dst + A);
                    Anchor<align, true>(src + A, srcStride, _threshold, _dst);
                    for (size_t col = DA; col < bodyWidth; col += A)
                        Anchor<align, false>(src + col, srcStride, _threshold, _dst);
                    Flush(_dst);
                }
                Anchor<false>(src + width - A - 1, srcStride, _threshold, dst + width - A - 1);
                dst[width - 1] = 0;
                src += step*srcStride;
                dst += step*dstStride;
            }
        }

        void ContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourAnchors<true>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
            else
                ContourAnchors<false>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
