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
        template <bool abs> SIMD_INLINE int16_t Laplace(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2);

        template <> SIMD_INLINE int16_t Laplace<false>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return 8 * s1[x1] - (s0[x0] + s0[x1] + s0[x2] + s1[x0] + s1[x2] + s2[x0] + s2[x1] + s2[x2]);
        }

        template <> SIMD_INLINE int16_t Laplace<true>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return (int16_t)Simd::Abs(Laplace<false>(s0, s1, s2, x0, x1, x2));
        }

        SIMD_INLINE svint16_t Laplace(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t center = svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 1), 3);
            svint16_t sum0 = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 0), svld1ub_s16(mask, s0 + 1)), svld1ub_s16(mask, s0 + 2));
            svint16_t sum1 = svadd_s16_x(mask, svld1ub_s16(mask, s1 + 0), svld1ub_s16(mask, s1 + 2));
            svint16_t sum2 = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s2 + 0), svld1ub_s16(mask, s2 + 1)), svld1ub_s16(mask, s2 + 2));
            return svsub_s16_x(mask, center, svadd_s16_x(mask, svadd_s16_x(mask, sum0, sum1), sum2));
        }

        template <bool abs> SIMD_INLINE svint16_t ConditionalAbs(const svint16_t& value, const svbool_t& mask)
        {
            return value;
        }

        template <> SIMD_INLINE svint16_t ConditionalAbs<true>(const svint16_t& value, const svbool_t& mask)
        {
            return svabs_s16_x(mask, value);
        }

        template <bool abs> SIMD_INLINE void Laplace(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, ConditionalAbs<abs>(Laplace(s0, s1, s2, mask), mask));
        }

        template <bool abs> void Laplace(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
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

                dst[0] = Laplace<abs>(src0, src1, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    Laplace<abs>(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = Laplace<abs>(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void Laplace(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            Laplace<false>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        void LaplaceAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            Laplace<true>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE svuint8_t Horiz(const uint8_t* src, const svbool_t& mask, svuint16x2_t& horiz)
        {
            svuint8_t left = svld1_u8(mask, src - 1);
            svuint8_t center = svld1_u8(mask, src);
            svuint8_t right = svld1_u8(mask, src + 1);
            const svbool_t mask16 = svptrue_b16();
            horiz = svcreate2_u16(
                svadd_u16_x(mask16, svaddlb_u16(left, right), svmovlb_u16(center)),
                svadd_u16_x(mask16, svaddlt_u16(left, right), svmovlt_u16(center)));
            return center;
        }

        SIMD_INLINE void AccumulateAbsLaplace(const svuint8_t& center, svuint16x2_t a, svuint16x2_t b, svuint16x2_t c, svuint32_t& sum)
        {
            const svbool_t mask16 = svptrue_b16();
            const svbool_t mask32 = svptrue_b32();
            svint16_t evenC = svreinterpret_s16_u16(svmovlb_u16(center));
            svint16_t oddC = svreinterpret_s16_u16(svmovlt_u16(center));
            svint16_t even9 = svadd_s16_x(mask16, svlsl_n_s16_x(mask16, evenC, 3), evenC);
            svint16_t odd9 = svadd_s16_x(mask16, svlsl_n_s16_x(mask16, oddC, 3), oddC);
            svint16_t evenS = svreinterpret_s16_u16(svadd_u16_x(mask16, svadd_u16_x(mask16, svget2(a, 0), svget2(b, 0)), svget2(c, 0)));
            svint16_t oddS = svreinterpret_s16_u16(svadd_u16_x(mask16, svadd_u16_x(mask16, svget2(a, 1), svget2(b, 1)), svget2(c, 1)));
            svuint16_t absEven = svreinterpret_u16_s16(svabs_s16_x(mask16, svsub_s16_x(mask16, even9, evenS)));
            svuint16_t absOdd = svreinterpret_u16_s16(svabs_s16_x(mask16, svsub_s16_x(mask16, odd9, oddS)));
            sum = svadd_u32_x(mask32, sum, svaddlb_u32(absEven, absOdd));
            sum = svadd_u32_x(mask32, sum, svaddlt_u32(absEven, absOdd));
        }

        SIMD_INLINE void LaplaceAbsSum1(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            svuint32_t& sum, const svbool_t& mask)
        {
            svuint16x2_t h0, h1, h2;
            Horiz(s0, mask, h0);
            svuint8_t c1 = Horiz(s1, mask, h1);
            Horiz(s2, mask, h2);
            AccumulateAbsLaplace(c1, h0, h1, h2, sum);
        }

        SIMD_INLINE void LaplaceAbsSum2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            svuint32_t& sum, const svbool_t& mask)
        {
            svuint16x2_t h0, h1, h2, h3;
            Horiz(s0, mask, h0);
            svuint8_t c1 = Horiz(s1, mask, h1);
            svuint8_t c2 = Horiz(s2, mask, h2);
            Horiz(s3, mask, h3);
            AccumulateAbsLaplace(c1, h0, h1, h2, sum);
            AccumulateAbsLaplace(c2, h1, h2, h3, sum);
        }

        SIMD_INLINE void LaplaceAbsSum4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            svuint32_t& sum, const svbool_t& mask)
        {
            svuint16x2_t h0, h1, h2, h3, h4, h5;
            Horiz(s0, mask, h0);
            svuint8_t c1 = Horiz(s1, mask, h1);
            svuint8_t c2 = Horiz(s2, mask, h2);
            svuint8_t c3 = Horiz(s3, mask, h3);
            svuint8_t c4 = Horiz(s4, mask, h4);
            Horiz(s5, mask, h5);
            AccumulateAbsLaplace(c1, h0, h1, h2, sum);
            AccumulateAbsLaplace(c2, h1, h2, h3, sum);
            AccumulateAbsLaplace(c3, h2, h3, h4, sum);
            AccumulateAbsLaplace(c4, h3, h4, h5, sum);
        }

        SIMD_INLINE uint64_t EdgeAbs(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            return (uint64_t)Laplace<true>(s0, s1, s2, x0, x1, x2);
        }

        void LaplaceAbsSumBody(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                LaplaceAbsSum1(s0 + col, s1 + col, s2 + col, sum, all);
                LaplaceAbsSum1(s0 + col + A, s1 + col + A, s2 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                LaplaceAbsSum1(s0 + col, s1 + col, s2 + col, sum, all);
            if (col < end)
                LaplaceAbsSum1(s0 + col, s1 + col, s2 + col, sum, svwhilelt_b8(col, end));
        }

        void LaplaceAbsSumBody2(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                LaplaceAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
                LaplaceAbsSum2(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                LaplaceAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, all);
            if (col < end)
                LaplaceAbsSum2(s0 + col, s1 + col, s2 + col, s3 + col, sum, svwhilelt_b8(col, end));
        }

        void LaplaceAbsSumBody4(
            const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* s3, const uint8_t* s4, const uint8_t* s5,
            size_t end, size_t A, size_t A2, const svbool_t& all, svuint32_t& sum)
        {
            size_t col = 1;
            for (; col + A2 <= end; col += A2)
            {
                LaplaceAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
                LaplaceAbsSum4(s0 + col + A, s1 + col + A, s2 + col + A, s3 + col + A, s4 + col + A, s5 + col + A, sum, all);
            }
            for (; col + A <= end; col += A)
                LaplaceAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, all);
            if (col < end)
                LaplaceAbsSum4(s0 + col, s1 + col, s2 + col, s3 + col, s4 + col, s5 + col, sum, svwhilelt_b8(col, end));
        }

        void LaplaceAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
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

                uint64_t edge = EdgeAbs(src0, src1, src2, 0, 0, 1);
                edge += EdgeAbs(src1, src2, src3, 0, 0, 1);
                edge += EdgeAbs(src2, src3, src4, 0, 0, 1);
                edge += EdgeAbs(src3, src4, src5, 0, 0, 1);
                edge += EdgeAbs(src0, src1, src2, width - 2, width - 1, width - 1);
                edge += EdgeAbs(src1, src2, src3, width - 2, width - 1, width - 1);
                edge += EdgeAbs(src2, src3, src4, width - 2, width - 1, width - 1);
                edge += EdgeAbs(src3, src4, src5, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                LaplaceAbsSumBody4(src0, src1, src2, src3, src4, src5, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1 + stride;
                const uint8_t* src3 = row + 2 < height ? src2 + stride : src2;

                uint64_t edge = EdgeAbs(src0, src1, src2, 0, 0, 1);
                edge += EdgeAbs(src1, src2, src3, 0, 0, 1);
                edge += EdgeAbs(src0, src1, src2, width - 2, width - 1, width - 1);
                edge += EdgeAbs(src1, src2, src3, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                LaplaceAbsSumBody2(src0, src1, src2, src3, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + stride * row;
                const uint8_t* src0 = row ? src1 - stride : src1;
                const uint8_t* src2 = src1;

                uint64_t edge = EdgeAbs(src0, src1, src2, 0, 0, 1);
                edge += EdgeAbs(src0, src1, src2, width - 2, width - 1, width - 1);

                svuint32_t body = svdup_n_u32(0);
                LaplaceAbsSumBody(src0, src1, src2, end, A, A2, all, body);
                fullSum += edge + svaddv_u32(svptrue_b32(), body);
            }

            *sum = fullSum;
        }
    }
#endif
}
