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
        SIMD_INLINE svuint16_t BinomialSumLo(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpklo_u16(left), svunpklo_u16(right)), svlsl_n_u16_x(mask, svunpklo_u16(center), 1));
        }

        SIMD_INLINE svuint16_t BinomialSumHi(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpkhi_u16(left), svunpkhi_u16(right)), svlsl_n_u16_x(mask, svunpkhi_u16(center), 1));
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, a, c), svlsl_n_u16_x(mask, b, 1));
        }

        SIMD_INLINE svuint16_t DivideBy16(const svuint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 8), 4);
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE svuint8_t GaussianBlur3x3(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            size_t step, const svbool_t& mask8)
        {
            svuint8_t left0 = svld1_u8(mask8, src0 - step);
            svuint8_t center0 = svld1_u8(mask8, src0);
            svuint8_t right0 = svld1_u8(mask8, src0 + step);
            svuint8_t left1 = svld1_u8(mask8, src1 - step);
            svuint8_t center1 = svld1_u8(mask8, src1);
            svuint8_t right1 = svld1_u8(mask8, src1 + step);
            svuint8_t left2 = svld1_u8(mask8, src2 - step);
            svuint8_t center2 = svld1_u8(mask8, src2);
            svuint8_t right2 = svld1_u8(mask8, src2 + step);
            svuint16_t lo = BinomialSum16(BinomialSumLo(left0, center0, right0),
                BinomialSumLo(left1, center1, right1), BinomialSumLo(left2, center2, right2));
            svuint16_t hi = BinomialSum16(BinomialSumHi(left0, center0, right0),
                BinomialSumHi(left1, center1, right1), BinomialSumHi(left2, center2, right2));
            return PackU16ToU8(DivideBy16(lo), DivideBy16(hi));
        }

        template <size_t step> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            const size_t size = width * step;
            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const svbool_t all = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = row + 1 < height ? src1 + srcStride : src1;
                for (size_t col = 0; col < step; ++col)
                    dst[col] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, col, col, col + step);

                if (width > 1)
                {
                    const size_t end = size - step;
                    size_t col = step;
                    for (; col + A2 <= end; col += A2)
                    {
                        svst1_u8(all, dst + col, GaussianBlur3x3(src0 + col, src1 + col, src2 + col, step, all));
                        svst1_u8(all, dst + col + A, GaussianBlur3x3(src0 + col + A, src1 + col + A, src2 + col + A, step, all));
                    }
                    for (; col < end; col += A)
                    {
                        svbool_t mask = svwhilelt_b8(col, end);
                        svst1_u8(mask, dst + col, GaussianBlur3x3(src0 + col, src1 + col, src2 + col, step, mask));
                    }

                    for (col = end; col < size; ++col)
                        dst[col] = (uint8_t)Base::GaussianBlur3x3<true>(src0, src1, src2, col - step, col, col);
                }
                dst += dstStride;
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
