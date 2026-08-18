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

        SIMD_INLINE svuint8_t VertPack(const svuint16_t& e0, const svuint16_t& o0,
            const svuint16_t& e1, const svuint16_t& o1, const svuint16_t& e2, const svuint16_t& o2)
        {
            const svbool_t mask = svptrue_b16();
            return PackEvenOdd(
                svrshr_n_u16_x(mask, Binomial16(e0, e1, e2), 4),
                svrshr_n_u16_x(mask, Binomial16(o0, o1, o2), 4));
        }

        template <size_t step> SIMD_INLINE svuint16x2_t Horiz(const uint8_t* src, const svbool_t& mask)
        {
            svuint8_t left = svld1_u8(mask, src - step);
            svuint8_t center = svld1_u8(mask, src);
            svuint8_t right = svld1_u8(mask, src + step);
            return svcreate2_u16(BinomialEven(left, center, right), BinomialOdd(left, center, right));
        }

        template <size_t step> void BlurCol(const uint8_t* src, size_t srcStride, size_t height,
            uint8_t* dst, size_t dstStride, size_t col, const svbool_t& mask)
        {
            svuint16x2_t h1 = Horiz<step>(src + col, mask);
            svuint16x2_t h0 = h1;
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src2 = row + 1 < height ? src + srcStride : src;
                svuint16x2_t h2 = Horiz<step>(src2 + col, mask);
                svst1_u8(mask, dst + col, VertPack(svget2(h0, 0), svget2(h0, 1),
                    svget2(h1, 0), svget2(h1, 1), svget2(h2, 0), svget2(h2, 1)));
                h0 = h1;
                h1 = h2;
                src += srcStride;
                dst += dstStride;
            }
        }

        template <size_t step> void BlurCol2(const uint8_t* src, size_t srcStride, size_t height,
            uint8_t* dst, size_t dstStride, size_t col)
        {
            const size_t A = svcntb();
            const svbool_t mask = svptrue_b8();
            svuint16x2_t a1 = Horiz<step>(src + col, mask);
            svuint16x2_t b1 = Horiz<step>(src + col + A, mask);
            svuint16x2_t a0 = a1;
            svuint16x2_t b0 = b1;
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src2 = row + 1 < height ? src + srcStride : src;
                svuint16x2_t a2 = Horiz<step>(src2 + col, mask);
                svuint16x2_t b2 = Horiz<step>(src2 + col + A, mask);
                svst1_u8(mask, dst + col, VertPack(svget2(a0, 0), svget2(a0, 1),
                    svget2(a1, 0), svget2(a1, 1), svget2(a2, 0), svget2(a2, 1)));
                svst1_u8(mask, dst + col + A, VertPack(svget2(b0, 0), svget2(b0, 1),
                    svget2(b1, 0), svget2(b1, 1), svget2(b2, 0), svget2(b2, 1)));
                a0 = a1;
                a1 = a2;
                b0 = b1;
                b1 = b2;
                src += srcStride;
                dst += dstStride;
            }
        }

        template <size_t step> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            const size_t size = width * step;
            const size_t A = svcntb();
            const size_t A2 = A * 2;

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
            size_t col = step;
            for (; col + A2 <= end; col += A2)
                BlurCol2<step>(src, srcStride, height, dst, dstStride, col);
            for (; col < end; col += A)
                BlurCol<step>(src, srcStride, height, dst, dstStride, col, svwhilelt_b8(col, end));

            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = row + 1 < height ? src1 + srcStride : src1;
                uint8_t* dstRow = dst + dstStride * row;
                for (size_t x = 0; x < step; ++x)
                    dstRow[x] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, x, x, x + step);
                for (size_t x = end; x < size; ++x)
                    dstRow[x] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, x - step, x, x);
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
