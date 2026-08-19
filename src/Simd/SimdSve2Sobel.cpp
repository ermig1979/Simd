/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template <bool abs> SIMD_INLINE int16_t SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2);

        template <> SIMD_INLINE int16_t SobelDx<false>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2)
        {
            return (int16_t)((s0[x2] + 2 * s1[x2] + s2[x2]) - (s0[x0] + 2 * s1[x0] + s2[x0]));
        }

        template <> SIMD_INLINE int16_t SobelDx<true>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2)
        {
            return (int16_t)Simd::Abs(SobelDx<false>(s0, s1, s2, x0, x2));
        }

        template <bool abs> SIMD_INLINE svint16_t SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask);

        template <> SIMD_INLINE svint16_t SobelDx<false>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t left = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1), 1)), svld1ub_s16(mask, s2));
            svint16_t right = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 2), 1)), svld1ub_s16(mask, s2 + 2));
            return svsub_s16_x(mask, right, left);
        }

        template <> SIMD_INLINE svint16_t SobelDx<true>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            return svabs_s16_x(mask, SobelDx<false>(s0, s1, s2, mask));
        }

        template <bool abs> SIMD_INLINE void SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, SobelDx<abs>(s0, s1, s2, mask));
        }

        template <bool abs> void SobelDx(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = SobelDx<abs>(src0, src1, src2, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    SobelDx<abs>(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = SobelDx<abs>(src0, src1, src2, width - 2, width - 1);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<false>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE svint16x2_t DxDiff(const uint8_t* src, const svbool_t& mask)
        {
            svuint8_t left = svld1_u8(mask, src - 1);
            svuint8_t right = svld1_u8(mask, src + 1);
            return svcreate2_s16(
                svreinterpret_s16_u16(svsublb_u16(right, left)),
                svreinterpret_s16_u16(svsublt_u16(right, left)));
        }

        SIMD_INLINE void StoreAbsDx(svint16x2_t d0, svint16x2_t d1, svint16x2_t d2, int16_t* dst, const svbool_t& lo, const svbool_t& hi)
        {
            const svbool_t mask16 = svptrue_b16();
            svint16_t even = svabs_s16_x(mask16, svadd_s16_x(mask16, svadd_s16_x(mask16, svget2(d0, 0), svlsl_n_s16_x(mask16, svget2(d1, 0), 1)), svget2(d2, 0)));
            svint16_t odd = svabs_s16_x(mask16, svadd_s16_x(mask16, svadd_s16_x(mask16, svget2(d0, 1), svlsl_n_s16_x(mask16, svget2(d1, 1), 1)), svget2(d2, 1)));
            svst1_s16(lo, dst, svzip1_s16(even, odd));
            svst1_s16(hi, dst + svcnth(), svzip2_s16(even, odd));
        }

        SIMD_INLINE void SobelDxAbs1(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            int16_t* dst, const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            StoreAbsDx(DxDiff(s0, mask), DxDiff(s1, mask), DxDiff(s2, mask), dst, lo, hi);
        }

        SIMD_INLINE void SobelDxAbs2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            int16_t* dst0, int16_t* dst1, const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            svint16x2_t d0 = DxDiff(s0, mask);
            svint16x2_t d1 = DxDiff(s1, mask);
            svint16x2_t d2 = DxDiff(s2, mask);
            svint16x2_t d3 = DxDiff(s3, mask);
            StoreAbsDx(d0, d1, d2, dst0, lo, hi);
            StoreAbsDx(d1, d2, d3, dst1, lo, hi);
        }

        SIMD_INLINE void SobelDxAbs4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            int16_t* dst0, int16_t* dst1, int16_t* dst2, int16_t* dst3,
            const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            svint16x2_t d0 = DxDiff(s0, mask);
            svint16x2_t d1 = DxDiff(s1, mask);
            svint16x2_t d2 = DxDiff(s2, mask);
            svint16x2_t d3 = DxDiff(s3, mask);
            svint16x2_t d4 = DxDiff(s4, mask);
            svint16x2_t d5 = DxDiff(s5, mask);
            StoreAbsDx(d0, d1, d2, dst0, lo, hi);
            StoreAbsDx(d1, d2, d3, dst1, lo, hi);
            StoreAbsDx(d2, d3, d4, dst2, lo, hi);
            StoreAbsDx(d3, d4, d5, dst3, lo, hi);
        }

        SIMD_INLINE void SobelDxAbsEdge(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, size_t width)
        {
            dst[0] = SobelDx<true>(s0, s1, s2, 0, 1);
            dst[width - 1] = SobelDx<true>(s0, s1, s2, width - 2, width - 1);
        }

        void SobelDxAbsBody(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            int16_t* dst, size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbs1(s0 + col, s1 + col, s2 + col, dst + col, all8, all16, all16);
                SobelDxAbs1(s0 + col + A, s1 + col + A, s2 + col + A, dst + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDxAbs1(s0 + col, s1 + col, s2 + col, dst + col, all8, all16, all16);
            if (col < end)
                SobelDxAbs1(s0 + col, s1 + col, s2 + col, dst + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDxAbsBody2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            int16_t* dst0, int16_t* dst1, size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, all8, all16, all16);
                SobelDxAbs2(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, dst0 + col + A, dst1 + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDxAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, all8, all16, all16);
            if (col < end)
                SobelDxAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDxAbsBody4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            int16_t* dst0, int16_t* dst1, int16_t* dst2, int16_t* dst3,
            size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all8, all16, all16);
                SobelDxAbs4(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, s4 + col + A, s5 + col + A,
                    dst0 + col + A, dst1 + col + A, dst2 + col + A, dst3 + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDxAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all8, all16, all16);
            if (col < end)
                SobelDxAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDxAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);
            assert(width > 1);

            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const size_t HA = svcnth();
            const size_t end = width - 1;
            const svbool_t all8 = svptrue_b8();
            const svbool_t all16 = svptrue_b16();
            int16_t* dst16 = (int16_t*)dst;
            size_t dst16Stride = dstStride / sizeof(int16_t);

            size_t row = 0;
            for (; row + 4 <= height; row += 4)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = src2 + srcStride;
                const uint8_t* src4 = src3 + srcStride;
                const uint8_t* src5 = row + 4 < height ? src4 + srcStride : src4;
                int16_t* dst0 = dst16 + dst16Stride * row;
                int16_t* dst1 = dst0 + dst16Stride;
                int16_t* dst2 = dst1 + dst16Stride;
                int16_t* dst3 = dst2 + dst16Stride;

                SobelDxAbsEdge(src0, src1, src2, dst0, width);
                SobelDxAbsEdge(src1, src2, src3, dst1, width);
                SobelDxAbsEdge(src2, src3, src4, dst2, width);
                SobelDxAbsEdge(src3, src4, src5, dst3, width);
                SobelDxAbsBody4(src0, src1, src2, src3, src4, src5, dst0, dst1, dst2, dst3, end, A, A2, HA, all8, all16);
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = row + 2 < height ? src2 + srcStride : src2;
                int16_t* dst0 = dst16 + dst16Stride * row;
                int16_t* dst1 = dst0 + dst16Stride;

                SobelDxAbsEdge(src0, src1, src2, dst0, width);
                SobelDxAbsEdge(src1, src2, src3, dst1, width);
                SobelDxAbsBody2(src0, src1, src2, src3, dst0, dst1, end, A, A2, HA, all8, all16);
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1;
                int16_t* dst0 = dst16 + dst16Stride * row;

                SobelDxAbsEdge(src0, src1, src2, dst0, width);
                SobelDxAbsBody(src0, src1, src2, dst0, end, A, A2, HA, all8, all16);
            }
        }

        SIMD_INLINE void AccumulateAbsDx(svint16x2_t d0, svint16x2_t d1, svint16x2_t d2, svuint32_t& sum)
        {
            const svbool_t mask16 = svptrue_b16();
            const svbool_t mask32 = svptrue_b32();
            svint16_t even = svadd_s16_x(mask16, svadd_s16_x(mask16, svget2(d0, 0), svlsl_n_s16_x(mask16, svget2(d1, 0), 1)), svget2(d2, 0));
            svint16_t odd = svadd_s16_x(mask16, svadd_s16_x(mask16, svget2(d0, 1), svlsl_n_s16_x(mask16, svget2(d1, 1), 1)), svget2(d2, 1));
            svuint16_t absEven = svreinterpret_u16_s16(svabs_s16_x(mask16, even));
            svuint16_t absOdd = svreinterpret_u16_s16(svabs_s16_x(mask16, odd));
            sum = svadd_u32_x(mask32, sum, svaddlb_u32(absEven, absOdd));
            sum = svadd_u32_x(mask32, sum, svaddlt_u32(absEven, absOdd));
        }

        SIMD_INLINE void SobelDxAbsSum1(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            svuint32_t& sum, const svbool_t& mask)
        {
            AccumulateAbsDx(DxDiff(s0, mask), DxDiff(s1, mask), DxDiff(s2, mask), sum);
        }

        SIMD_INLINE void SobelDxAbsSum2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            svuint32_t& sum, const svbool_t& mask)
        {
            svint16x2_t d0 = DxDiff(s0, mask);
            svint16x2_t d1 = DxDiff(s1, mask);
            svint16x2_t d2 = DxDiff(s2, mask);
            svint16x2_t d3 = DxDiff(s3, mask);
            AccumulateAbsDx(d0, d1, d2, sum);
            AccumulateAbsDx(d1, d2, d3, sum);
        }

        SIMD_INLINE void SobelDxAbsSum4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            svuint32_t& sum, const svbool_t& mask)
        {
            svint16x2_t d0 = DxDiff(s0, mask);
            svint16x2_t d1 = DxDiff(s1, mask);
            svint16x2_t d2 = DxDiff(s2, mask);
            svint16x2_t d3 = DxDiff(s3, mask);
            svint16x2_t d4 = DxDiff(s4, mask);
            svint16x2_t d5 = DxDiff(s5, mask);
            AccumulateAbsDx(d0, d1, d2, sum);
            AccumulateAbsDx(d1, d2, d3, sum);
            AccumulateAbsDx(d2, d3, d4, sum);
            AccumulateAbsDx(d3, d4, d5, sum);
        }

        SIMD_INLINE uint64_t EdgeAbsDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2)
        {
            return (uint64_t)SobelDx<true>(s0, s1, s2, x0, x2);
        }

        void SobelDxAbsSumBody(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbsSum1(s0 + col, s1 + col, s2 + col, sum, all);
                SobelDxAbsSum1(s0 + col + A, s1 + col + A, s2 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDxAbsSum1(s0 + col, s1 + col, s2 + col, sum, all);
            if (col < end)
                SobelDxAbsSum1(s0 + col, s1 + col, s2 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDxAbsSumBody2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
                SobelDxAbsSum2(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDxAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
            if (col < end)
                SobelDxAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDxAbsSumBody4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDxAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
                SobelDxAbsSum4(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, s4 + col + A, s5 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDxAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
            if (col < end)
                SobelDxAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDxAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width > 1);

            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const size_t end = width - 1;
            const svbool_t all = svptrue_b8();
            uint64_t fullSum = 0;

            size_t row = 0;
            for (; row + 4 <= height; row += 4)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1 + stride;
                const uint8_t* src3 = src2 + stride;
                const uint8_t* src4 = src3 + stride;
                const uint8_t* src5 = row + 4 < height ? src4 + stride : src4;

                uint64_t edge = EdgeAbsDx(src0, src1, src2, 0, 1);
                edge += EdgeAbsDx(src1, src2, src3, 0, 1);
                edge += EdgeAbsDx(src2, src3, src4, 0, 1);
                edge += EdgeAbsDx(src3, src4, src5, 0, 1);
                edge += EdgeAbsDx(src0, src1, src2, width - 2, width - 1);
                edge += EdgeAbsDx(src1, src2, src3, width - 2, width - 1);
                edge += EdgeAbsDx(src2, src3, src4, width - 2, width - 1);
                edge += EdgeAbsDx(src3, src4, src5, width - 2, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDxAbsSumBody4(src0, src1, src2, src3, src4, src5, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1 + stride;
                const uint8_t* src3 = row + 2 < height ? src2 + stride : src2;

                uint64_t edge = EdgeAbsDx(src0, src1, src2, 0, 1);
                edge += EdgeAbsDx(src1, src2, src3, 0, 1);
                edge += EdgeAbsDx(src0, src1, src2, width - 2, width - 1);
                edge += EdgeAbsDx(src1, src2, src3, width - 2, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDxAbsSumBody2(src0, src1, src2, src3, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1;

                uint64_t edge = EdgeAbsDx(src0, src1, src2, 0, 1);
                edge += EdgeAbsDx(src0, src1, src2, width - 2, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDxAbsSumBody(src0, src1, src2, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            *sum = fullSum;
        }

        template <bool abs> SIMD_INLINE int16_t SobelDy(const uint8_t* s0, const uint8_t* s2, size_t x0, size_t x1, size_t x2);

        template <> SIMD_INLINE int16_t SobelDy<false>(const uint8_t* s0, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return (int16_t)((s2[x0] + 2 * s2[x1] + s2[x2]) - (s0[x0] + 2 * s0[x1] + s0[x2]));
        }

        template <> SIMD_INLINE int16_t SobelDy<true>(const uint8_t* s0, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return (int16_t)Simd::Abs(SobelDy<false>(s0, s2, x0, x1, x2));
        }

        template <bool abs> SIMD_INLINE svint16_t SobelDy(const uint8_t* s0, const uint8_t* s2, const svbool_t& mask);

        template <> SIMD_INLINE svint16_t SobelDy<false>(const uint8_t* s0, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t top = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s0 + 1), 1)), svld1ub_s16(mask, s0 + 2));
            svint16_t bottom = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s2 + 1), 1)), svld1ub_s16(mask, s2 + 2));
            return svsub_s16_x(mask, bottom, top);
        }

        template <> SIMD_INLINE svint16_t SobelDy<true>(const uint8_t* s0, const uint8_t* s2, const svbool_t& mask)
        {
            return svabs_s16_x(mask, SobelDy<false>(s0, s2, mask));
        }

        SIMD_INLINE svint16_t SobelDyAbs(const uint8_t* s0, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t top = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s0 + 1), 1)), svld1ub_s16(mask, s0 + 2));
            svint16_t bottom = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s2 + 1), 1)), svld1ub_s16(mask, s2 + 2));
            return svabs_s16_x(mask, svsub_s16_x(mask, bottom, top));
        }

        template <bool abs> SIMD_INLINE void SobelDy(const uint8_t* s0, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, SobelDy<abs>(s0, s2, mask));
        }

        template <bool abs> void SobelDy(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t* src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = SobelDy<abs>(src0, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    SobelDy<abs>(src0 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = SobelDy<abs>(src0, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDy<false>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE svuint16x2_t DyHoriz(const uint8_t* src, const svbool_t& mask)
        {
            svuint8_t left = svld1_u8(mask, src - 1);
            svuint8_t center = svld1_u8(mask, src);
            svuint8_t right = svld1_u8(mask, src + 1);
            return svcreate2_u16(
                svmlalb_n_u16(svaddlb_u16(left, right), center, 2),
                svmlalt_n_u16(svaddlt_u16(left, right), center, 2));
        }

        SIMD_INLINE void StoreAbsDy(svuint16x2_t top, svuint16x2_t bot, int16_t* dst, const svbool_t& lo, const svbool_t& hi)
        {
            const svbool_t mask16 = svptrue_b16();
            svint16_t even = svabs_s16_x(mask16, svsub_s16_x(mask16, svreinterpret_s16_u16(svget2(bot, 0)), svreinterpret_s16_u16(svget2(top, 0))));
            svint16_t odd = svabs_s16_x(mask16, svsub_s16_x(mask16, svreinterpret_s16_u16(svget2(bot, 1)), svreinterpret_s16_u16(svget2(top, 1))));
            svst1_s16(lo, dst, svzip1_s16(even, odd));
            svst1_s16(hi, dst + svcnth(), svzip2_s16(even, odd));
        }

        SIMD_INLINE void SobelDyAbs1(const uint8_t* s0, const uint8_t* s2,
            int16_t* dst, const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            StoreAbsDy(DyHoriz(s0, mask), DyHoriz(s2, mask), dst, lo, hi);
        }

        SIMD_INLINE void SobelDyAbs2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            int16_t* dst0, int16_t* dst1, const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            svuint16x2_t h0 = DyHoriz(s0, mask);
            svuint16x2_t h1 = DyHoriz(s1, mask);
            svuint16x2_t h2 = DyHoriz(s2, mask);
            svuint16x2_t h3 = DyHoriz(s3, mask);
            StoreAbsDy(h0, h2, dst0, lo, hi);
            StoreAbsDy(h1, h3, dst1, lo, hi);
        }

        SIMD_INLINE void SobelDyAbs4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            int16_t* dst0, int16_t* dst1, int16_t* dst2, int16_t* dst3,
            const svbool_t& mask, const svbool_t& lo, const svbool_t& hi)
        {
            svuint16x2_t h0 = DyHoriz(s0, mask);
            svuint16x2_t h1 = DyHoriz(s1, mask);
            svuint16x2_t h2 = DyHoriz(s2, mask);
            svuint16x2_t h3 = DyHoriz(s3, mask);
            svuint16x2_t h4 = DyHoriz(s4, mask);
            svuint16x2_t h5 = DyHoriz(s5, mask);
            StoreAbsDy(h0, h2, dst0, lo, hi);
            StoreAbsDy(h1, h3, dst1, lo, hi);
            StoreAbsDy(h2, h4, dst2, lo, hi);
            StoreAbsDy(h3, h5, dst3, lo, hi);
        }

        SIMD_INLINE void SobelDyAbsEdge(const uint8_t* s0, const uint8_t* s2, int16_t* dst, size_t width)
        {
            dst[0] = SobelDy<true>(s0, s2, 0, 0, 1);
            dst[width - 1] = SobelDy<true>(s0, s2, width - 2, width - 1, width - 1);
        }

        void SobelDyAbsBody(const uint8_t* s0, const uint8_t* s2,
            int16_t* dst, size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbs1(s0 + col, s2 + col, dst + col, all8, all16, all16);
                SobelDyAbs1(s0 + col + A, s2 + col + A, dst + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDyAbs1(s0 + col, s2 + col, dst + col, all8, all16, all16);
            if (col < end)
                SobelDyAbs1(s0 + col, s2 + col, dst + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDyAbsBody2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            int16_t* dst0, int16_t* dst1, size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, all8, all16, all16);
                SobelDyAbs2(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, dst0 + col + A, dst1 + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDyAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, all8, all16, all16);
            if (col < end)
                SobelDyAbs2(s0 + col, s1 + col, s2 + col, s3 + col, dst0 + col, dst1 + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDyAbsBody4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            int16_t* dst0, int16_t* dst1, int16_t* dst2, int16_t* dst3,
            size_t end, size_t A, size_t A2, size_t HA, const svbool_t& all8, const svbool_t& all16)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all8, all16, all16);
                SobelDyAbs4(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, s4 + col + A, s5 + col + A,
                    dst0 + col + A, dst1 + col + A, dst2 + col + A, dst3 + col + A, all8, all16, all16);
            }
            for (; col + A <= end; col += A)
                SobelDyAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all8, all16, all16);
            if (col < end)
                SobelDyAbs4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, svwhilelt_b8(col, end),
                    svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));
        }

        void SobelDyAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);
            assert(width > 1);

            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const size_t HA = svcnth();
            const size_t end = width - 1;
            const svbool_t all8 = svptrue_b8();
            const svbool_t all16 = svptrue_b16();
            int16_t* dst16 = (int16_t*)dst;
            size_t dst16Stride = dstStride / sizeof(int16_t);

            size_t row = 0;
            for (; row + 4 <= height; row += 4)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = src2 + srcStride;
                const uint8_t* src4 = src3 + srcStride;
                const uint8_t* src5 = row + 4 < height ? src4 + srcStride : src4;
                int16_t* dst0 = dst16 + dst16Stride * row;
                int16_t* dst1 = dst0 + dst16Stride;
                int16_t* dst2 = dst1 + dst16Stride;
                int16_t* dst3 = dst2 + dst16Stride;

                SobelDyAbsEdge(src0, src2, dst0, width);
                SobelDyAbsEdge(src1, src3, dst1, width);
                SobelDyAbsEdge(src2, src4, dst2, width);
                SobelDyAbsEdge(src3, src5, dst3, width);
                SobelDyAbsBody4(src0, src1, src2, src3, src4, src5, dst0, dst1, dst2, dst3, end, A, A2, HA, all8, all16);
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = row + 2 < height ? src2 + srcStride : src2;
                int16_t* dst0 = dst16 + dst16Stride * row;
                int16_t* dst1 = dst0 + dst16Stride;

                SobelDyAbsEdge(src0, src2, dst0, width);
                SobelDyAbsEdge(src1, src3, dst1, width);
                SobelDyAbsBody2(src0, src1, src2, src3, dst0, dst1, end, A, A2, HA, all8, all16);
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1;
                int16_t* dst0 = dst16 + dst16Stride * row;

                SobelDyAbsEdge(src0, src2, dst0, width);
                SobelDyAbsBody(src0, src2, dst0, end, A, A2, HA, all8, all16);
            }
        }

        SIMD_INLINE void AccumulateAbsDy(svuint16x2_t top, svuint16x2_t bot, svuint32_t& sum)
        {
            const svbool_t mask16 = svptrue_b16();
            const svbool_t mask32 = svptrue_b32();
            svint16_t even = svsub_s16_x(mask16, svreinterpret_s16_u16(svget2(bot, 0)), svreinterpret_s16_u16(svget2(top, 0)));
            svint16_t odd = svsub_s16_x(mask16, svreinterpret_s16_u16(svget2(bot, 1)), svreinterpret_s16_u16(svget2(top, 1)));
            svuint16_t absEven = svreinterpret_u16_s16(svabs_s16_x(mask16, even));
            svuint16_t absOdd = svreinterpret_u16_s16(svabs_s16_x(mask16, odd));
            sum = svadd_u32_x(mask32, sum, svaddlb_u32(absEven, absOdd));
            sum = svadd_u32_x(mask32, sum, svaddlt_u32(absEven, absOdd));
        }

        SIMD_INLINE void SobelDyAbsSum1(const uint8_t* s0, const uint8_t* s2,
            svuint32_t& sum, const svbool_t& mask)
        {
            AccumulateAbsDy(DyHoriz(s0, mask), DyHoriz(s2, mask), sum);
        }

        SIMD_INLINE void SobelDyAbsSum2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            svuint32_t& sum, const svbool_t& mask)
        {
            svuint16x2_t h0 = DyHoriz(s0, mask);
            svuint16x2_t h1 = DyHoriz(s1, mask);
            svuint16x2_t h2 = DyHoriz(s2, mask);
            svuint16x2_t h3 = DyHoriz(s3, mask);
            AccumulateAbsDy(h0, h2, sum);
            AccumulateAbsDy(h1, h3, sum);
        }

        SIMD_INLINE void SobelDyAbsSum4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            svuint32_t& sum, const svbool_t& mask)
        {
            svuint16x2_t h0 = DyHoriz(s0, mask);
            svuint16x2_t h1 = DyHoriz(s1, mask);
            svuint16x2_t h2 = DyHoriz(s2, mask);
            svuint16x2_t h3 = DyHoriz(s3, mask);
            svuint16x2_t h4 = DyHoriz(s4, mask);
            svuint16x2_t h5 = DyHoriz(s5, mask);
            AccumulateAbsDy(h0, h2, sum);
            AccumulateAbsDy(h1, h3, sum);
            AccumulateAbsDy(h2, h4, sum);
            AccumulateAbsDy(h3, h5, sum);
        }

        SIMD_INLINE uint64_t EdgeAbsDy(const uint8_t* s0, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return (uint64_t)SobelDy<true>(s0, s2, x0, x1, x2);
        }

        void SobelDyAbsSumBody(const uint8_t* s0, const uint8_t* s2,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbsSum1(s0 + col, s2 + col, sum, all);
                SobelDyAbsSum1(s0 + col + A, s2 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDyAbsSum1(s0 + col, s2 + col, sum, all);
            if (col < end)
                SobelDyAbsSum1(s0 + col, s2 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDyAbsSumBody2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
                SobelDyAbsSum2(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDyAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
            if (col < end)
                SobelDyAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDyAbsSumBody4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                SobelDyAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
                SobelDyAbsSum4(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, s4 + col + A, s5 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                SobelDyAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
            if (col < end)
                SobelDyAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, svwhilelt_b8(col, end));
        }

        void SobelDyAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width > 1);

            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const size_t end = width - 1;
            const svbool_t all = svptrue_b8();
            uint64_t fullSum = 0;

            size_t row = 0;
            for (; row + 4 <= height; row += 4)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1 + stride;
                const uint8_t* src3 = src2 + stride;
                const uint8_t* src4 = src3 + stride;
                const uint8_t* src5 = row + 4 < height ? src4 + stride : src4;

                uint64_t edge = EdgeAbsDy(src0, src2, 0, 0, 1);
                edge += EdgeAbsDy(src1, src3, 0, 0, 1);
                edge += EdgeAbsDy(src2, src4, 0, 0, 1);
                edge += EdgeAbsDy(src3, src5, 0, 0, 1);
                edge += EdgeAbsDy(src0, src2, width - 2, width - 1, width - 1);
                edge += EdgeAbsDy(src1, src3, width - 2, width - 1, width - 1);
                edge += EdgeAbsDy(src2, src4, width - 2, width - 1, width - 1);
                edge += EdgeAbsDy(src3, src5, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDyAbsSumBody4(src0, src1, src2, src3, src4, src5, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1 + stride;
                const uint8_t* src3 = row + 2 < height ? src2 + stride : src2;

                uint64_t edge = EdgeAbsDy(src0, src2, 0, 0, 1);
                edge += EdgeAbsDy(src1, src3, 0, 0, 1);
                edge += EdgeAbsDy(src0, src2, width - 2, width - 1, width - 1);
                edge += EdgeAbsDy(src1, src3, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDyAbsSumBody2(src0, src1, src2, src3, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1;

                uint64_t edge = EdgeAbsDy(src0, src2, 0, 0, 1);
                edge += EdgeAbsDy(src0, src2, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                SobelDyAbsSumBody(src0, src2, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            *sum = fullSum;
        }

        SIMD_INLINE int16_t ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            int dx = Simd::Abs((s0[x2] + 2 * s1[x2] + s2[x2]) - (s0[x0] + 2 * s1[x0] + s2[x0]));
            int dy = SobelDy<true>(s0, s2, x0, x1, x2);
            return (int16_t)((dx + dy) * 2 + (dx >= dy ? 0 : 1));
        }

        SIMD_INLINE svint16_t SobelDxAbs(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t left = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1), 1)), svld1ub_s16(mask, s2));
            svint16_t right = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 2), 1)), svld1ub_s16(mask, s2 + 2));
            return svabs_s16_x(mask, svsub_s16_x(mask, right, left));
        }

        SIMD_INLINE svint16_t ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t dx = SobelDxAbs(s0, s1, s2, mask);
            svint16_t dy = SobelDyAbs(s0, s2, mask);
            svint16_t sum = svlsl_n_s16_x(mask, svadd_s16_x(mask, dx, dy), 1);
            return svadd_s16_x(mask, sum, svsel_s16(svcmplt_s16(mask, dx, dy), svdup_n_s16(1), svdup_n_s16(0)));
        }

        SIMD_INLINE void ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, ContourMetrics(s0, s1, s2, mask));
        }

        void ContourMetrics(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = ContourMetrics(src0, src1, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    ContourMetrics(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetrics(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void ContourMetricsMasked(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* mask, uint8_t indexMin, int16_t* dst, const svbool_t& tail)
        {
            svuint16_t _mask = svld1ub_u16(tail, mask);
            svbool_t valid = svcmpge_n_u16(tail, _mask, indexMin);
            svst1_s16(tail, dst, svsel_s16(valid, ContourMetrics(s0, s1, s2, tail), svdup_n_s16(0)));
        }

        void ContourMetricsMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t indexMin, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = mask[0] < indexMin ? 0 : ContourMetrics(src0, src1, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    ContourMetricsMasked(src0 + col - 1, src1 + col - 1, src2 + col - 1,
                        mask + col, indexMin, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = mask[width - 1] < indexMin ? 0 : ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t indexMin, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svint16_t Half(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_s16_u16(svlsr_n_u16_x(mask, svld1_u16(mask, src), 1));
        }

        SIMD_INLINE void Anchor(const uint16_t* src, size_t stride, const svint16_t& threshold, uint8_t* dst, const svbool_t& mask)
        {
            svuint16_t _src = svld1_u16(mask, src);
            svuint16_t direction = svand_n_u16_x(mask, _src, 1);
            svuint16_t magnitude = svlsr_n_u16_x(mask, _src, 1);
            svint16_t value = svsub_s16_x(mask, svreinterpret_s16_u16(magnitude), threshold);

            svbool_t vertical = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 1),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - 1, mask)), svcmpge_s16(mask, value, Half(src + 1, mask))));
            svbool_t horizontal = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 0),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - stride, mask)), svcmpge_s16(mask, value, Half(src + stride, mask))));
            svbool_t anchor = svand_b_z(mask, svcmpgt_n_u16(mask, magnitude, 0), svorr_b_z(mask, vertical, horizontal));

            svst1b_u16(mask, dst, svsel_u16(anchor, svdup_n_u16(0xFF), svdup_n_u16(0)));
        }

        void ContourAnchors(const uint16_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcnth();
            const svint16_t _threshold = svdup_n_s16(threshold);

            memset(dst, 0, width);
            memset(dst + dstStride * (height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                for (size_t col = 1; col < width - 1; col += A)
                    Anchor(src + col, srcStride, _threshold, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = 0;
                src += step * srcStride;
                dst += step * dstStride;
            }
        }

        void ContourAnchors(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            ContourAnchors((const uint16_t*)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif
}
