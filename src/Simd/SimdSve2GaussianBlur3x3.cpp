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
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint16_t BinomialEven(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            return svmlalb_n_u16(svaddlb_u16(left, right), center, 2);
        }

        SIMD_INLINE svuint16_t BinomialOdd(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            return svmlalt_n_u16(svaddlt_u16(left, right), center, 2);
        }

        SIMD_INLINE svuint16_t Binomial16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c)
        {
            const svbool_t mask = svptrue_b16();
            return svmla_n_u16_x(mask, svadd_u16_x(mask, a, c), b, 2);
        }

        SIMD_INLINE svuint8_t PackEvenOdd(const svuint16_t& even, const svuint16_t& odd)
        {
            return svqxtnt_u16(svqxtnb_u16(even), odd);
        }

        SIMD_INLINE svuint8_t Vert3(svuint16x2_t a, svuint16x2_t b, svuint16x2_t c)
        {
            const svbool_t mask = svptrue_b16();
            return PackEvenOdd(
                svrshr_n_u16_x(mask, Binomial16(svget2(a, 0), svget2(b, 0), svget2(c, 0)), 4),
                svrshr_n_u16_x(mask, Binomial16(svget2(a, 1), svget2(b, 1), svget2(c, 1)), 4));
        }

        template <size_t step> SIMD_INLINE svuint16x2_t Horiz(const uint8_t* src, const svbool_t& mask)
        {
            svuint8_t left = svld1_u8(mask, src - step);
            svuint8_t center = svld1_u8(mask, src);
            svuint8_t right = svld1_u8(mask, src + step);
            return svcreate2_u16(BinomialEven(left, center, right), BinomialOdd(left, center, right));
        }

        template <size_t step> SIMD_INLINE void Blur1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            uint8_t* dst, const svbool_t& mask)
        {
            svst1_u8(mask, dst, Vert3(Horiz<step>(src0, mask), Horiz<step>(src1, mask), Horiz<step>(src2, mask)));
        }

        template <size_t step> SIMD_INLINE void Blur2(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
            uint8_t* dst0, uint8_t* dst1, const svbool_t& mask)
        {
            svuint16x2_t h0 = Horiz<step>(src0, mask);
            svuint16x2_t h1 = Horiz<step>(src1, mask);
            svuint16x2_t h2 = Horiz<step>(src2, mask);
            svuint16x2_t h3 = Horiz<step>(src3, mask);
            svst1_u8(mask, dst0, Vert3(h0, h1, h2));
            svst1_u8(mask, dst1, Vert3(h1, h2, h3));
        }

        template <size_t step> SIMD_INLINE void Blur4(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, uint8_t* dst2, uint8_t* dst3, const svbool_t& mask)
        {
            svuint16x2_t h0 = Horiz<step>(src0, mask);
            svuint16x2_t h1 = Horiz<step>(src1, mask);
            svuint16x2_t h2 = Horiz<step>(src2, mask);
            svuint16x2_t h3 = Horiz<step>(src3, mask);
            svuint16x2_t h4 = Horiz<step>(src4, mask);
            svuint16x2_t h5 = Horiz<step>(src5, mask);
            svst1_u8(mask, dst0, Vert3(h0, h1, h2));
            svst1_u8(mask, dst1, Vert3(h1, h2, h3));
            svst1_u8(mask, dst2, Vert3(h2, h3, h4));
            svst1_u8(mask, dst3, Vert3(h3, h4, h5));
        }

        SIMD_INLINE void Edge(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            uint8_t* dst, size_t x0, size_t x1, size_t x2)
        {
            dst[x1] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, x0, x1, x2);
        }

        template <size_t step> void BlurBody(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            uint8_t* dst, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = step;
            for (; col + A2 <= end; col += A2)
            {
                Blur1<step>(src0 + col, src1 + col, src2 + col, dst + col, all);
                Blur1<step>(src0 + col + A, src1 + col + A, src2 + col + A, dst + col + A, all);
            }
            for (; col < end; col += A)
                Blur1<step>(src0 + col, src1 + col, src2 + col, dst + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void BlurBody2(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
            uint8_t* dst0, uint8_t* dst1, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = step;
            for (; col + A2 <= end; col += A2)
            {
                Blur2<step>(src0 + col, src1 + col, src2 + col, src3 + col, dst0 + col, dst1 + col, all);
                Blur2<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, dst0 + col + A, dst1 + col + A, all);
            }
            for (; col < end; col += A)
                Blur2<step>(src0 + col, src1 + col, src2 + col, src3 + col, dst0 + col, dst1 + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void BlurBody4(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, uint8_t* dst2, uint8_t* dst3,
            size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = step;
            for (; col + A2 <= end; col += A2)
            {
                Blur4<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all);
                Blur4<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, src5 + col + A,
                    dst0 + col + A, dst1 + col + A, dst2 + col + A, dst3 + col + A, all);
            }
            for (; col < end; col += A)
                Blur4<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            const size_t size = width * step;
            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const svbool_t all = svptrue_b8();

            if (width == 1)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    const uint8_t* src1 = src + srcStride * row;
                    const uint8_t* src0 = row ? src1 - srcStride : src1;
                    const uint8_t* src2 = row + 1 < height ? src1 + srcStride : src1;
                    for (size_t col = 0; col < step; ++col)
                        dst[col] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, col, col, col);
                    dst += dstStride;
                }
                return;
            }

            const size_t end = size - step;
            size_t row = 0;
            for (; row + 4 <= height; row += 4)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = src2 + srcStride;
                const uint8_t* src4 = src3 + srcStride;
                const uint8_t* src5 = row + 4 < height ? src4 + srcStride : src4;
                uint8_t* dst0 = dst + dstStride * row;
                uint8_t* dst1 = dst0 + dstStride;
                uint8_t* dst2 = dst1 + dstStride;
                uint8_t* dst3 = dst2 + dstStride;

                for (size_t x = 0; x < step; ++x)
                {
                    Edge(src0, src1, src2, dst0, x, x, x + step);
                    Edge(src1, src2, src3, dst1, x, x, x + step);
                    Edge(src2, src3, src4, dst2, x, x, x + step);
                    Edge(src3, src4, src5, dst3, x, x, x + step);
                }
                BlurBody4<step>(src0, src1, src2, src3, src4, src5, dst0, dst1, dst2, dst3, end, A, A2, all);
                for (size_t x = end; x < size; ++x)
                {
                    Edge(src0, src1, src2, dst0, x - step, x, x);
                    Edge(src1, src2, src3, dst1, x - step, x, x);
                    Edge(src2, src3, src4, dst2, x - step, x, x);
                    Edge(src3, src4, src5, dst3, x - step, x, x);
                }
            }

            if (row + 2 <= height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src3 = row + 2 < height ? src2 + srcStride : src2;
                uint8_t* dst0 = dst + dstStride * row;
                uint8_t* dst1 = dst0 + dstStride;

                for (size_t x = 0; x < step; ++x)
                {
                    Edge(src0, src1, src2, dst0, x, x, x + step);
                    Edge(src1, src2, src3, dst1, x, x, x + step);
                }
                BlurBody2<step>(src0, src1, src2, src3, dst0, dst1, end, A, A2, all);
                for (size_t x = end; x < size; ++x)
                {
                    Edge(src0, src1, src2, dst0, x - step, x, x);
                    Edge(src1, src2, src3, dst1, x - step, x, x);
                }
                row += 2;
            }

            if (row < height)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = src1;
                uint8_t* dst0 = dst + dstStride * row;

                for (size_t x = 0; x < step; ++x)
                    Edge(src0, src1, src2, dst0, x, x, x + step);
                BlurBody<step>(src0, src1, src2, dst0, end, A, A2, all);
                for (size_t x = end; x < size; ++x)
                    Edge(src0, src1, src2, dst0, x - step, x, x);
            }
        }

        void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
