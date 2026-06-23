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
#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE uint8_t DivideBy9(size_t value)
        {
            return uint8_t(((value + 5) * Base::DIVISION_BY_9_FACTOR) >> Base::DIVISION_BY_9_SHIFT);
        }

        SIMD_INLINE svuint16_t DivideBy9(const svuint16_t& value)
        {
            const svbool_t mask = svptrue_b32();
            svuint16_t value5 = svadd_n_u16_x(svptrue_b16(), value, 5);
            svuint32_t lo = svlsr_n_u32_x(mask, svmul_n_u32_x(mask, svunpklo_u32(value5),
                Base::DIVISION_BY_9_FACTOR), Base::DIVISION_BY_9_SHIFT);
            svuint32_t hi = svlsr_n_u32_x(mask, svmul_n_u32_x(mask, svunpkhi_u32(value5),
                Base::DIVISION_BY_9_FACTOR), Base::DIVISION_BY_9_SHIFT);
            return svuzp1_u16(svreinterpret_u16_u32(lo), svreinterpret_u16_u32(hi));
        }

        SIMD_INLINE svuint16_t RowSumLo(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpklo_u16(left), svunpklo_u16(center)), svunpklo_u16(right));
        }

        SIMD_INLINE svuint16_t RowSumHi(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpkhi_u16(left), svunpkhi_u16(center)), svunpkhi_u16(right));
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE void MeanFilter3x3(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            size_t step, uint8_t* dst, const svbool_t& mask8)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint8_t left0 = svld1_u8(mask8, src0 - step);
            svuint8_t center0 = svld1_u8(mask8, src0);
            svuint8_t right0 = svld1_u8(mask8, src0 + step);
            svuint8_t left1 = svld1_u8(mask8, src1 - step);
            svuint8_t center1 = svld1_u8(mask8, src1);
            svuint8_t right1 = svld1_u8(mask8, src1 + step);
            svuint8_t left2 = svld1_u8(mask8, src2 - step);
            svuint8_t center2 = svld1_u8(mask8, src2);
            svuint8_t right2 = svld1_u8(mask8, src2 + step);
            svuint16_t lo = svadd_u16_x(mask16, svadd_u16_x(mask16, RowSumLo(left0, center0, right0),
                RowSumLo(left1, center1, right1)), RowSumLo(left2, center2, right2));
            svuint16_t hi = svadd_u16_x(mask16, svadd_u16_x(mask16, RowSumHi(left0, center0, right0),
                RowSumHi(left1, center1, right1)), RowSumHi(left2, center2, right2));
            svst1_u8(mask8, dst, PackU16ToU8(DivideBy9(lo), DivideBy9(hi)));
        }

        template <size_t step> void MeanFilter3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * step;
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src1 = src + srcStride * row;
                const uint8_t* src0 = row ? src1 - srcStride : src1;
                const uint8_t* src2 = row + 1 < height ? src1 + srcStride : src1;
                if (width == 1)
                {
                    for (size_t col = 0; col < step; ++col)
                        dst[col] = DivideBy9((src0[col] + src1[col] + src2[col]) * 3);
                }
                else
                {
                    for (size_t col = 0; col < step; ++col)
                        dst[col] = DivideBy9(src0[col] + src0[col] + src0[col + step] +
                            src1[col] + src1[col] + src1[col + step] +
                            src2[col] + src2[col] + src2[col + step]);

                    const size_t end = size - step, A = svcntb();
                    size_t col = step;
                    for (; col < end; col += A)
                        MeanFilter3x3(src0 + col, src1 + col, src2 + col, step, dst + col, svwhilelt_b8(col, end));

                    for (col = end; col < size; ++col)
                        dst[col] = DivideBy9(src0[col - step] + src0[col] + src0[col] +
                            src1[col - step] + src1[col] + src1[col] +
                            src2[col - step] + src2[col] + src2[col]);
                }
                dst += dstStride;
            }
        }

        void MeanFilter3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MeanFilter3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MeanFilter3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MeanFilter3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MeanFilter3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
