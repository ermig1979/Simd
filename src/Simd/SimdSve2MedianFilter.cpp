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
        SIMD_INLINE uint8_t Median5(const uint8_t* y[3], size_t x0, size_t x1, size_t x2)
        {
            int a[5];
            a[0] = y[0][x1];
            a[1] = y[1][x0];
            a[2] = y[1][x1];
            a[3] = y[1][x2];
            a[4] = y[2][x1];
            Base::SortU8(a[2], a[3]);
            Base::SortU8(a[1], a[2]);
            Base::SortU8(a[2], a[3]);
            a[4] = Base::MaxU8(a[1], a[4]);
            a[0] = Base::MinU8(a[0], a[3]);
            Base::SortU8(a[2], a[0]);
            a[2] = Base::MaxU8(a[4], a[2]);
            a[2] = Base::MinU8(a[2], a[0]);
            return (uint8_t)a[2];
        }

        SIMD_INLINE void SortU8(svuint8_t& a, svuint8_t& b, const svbool_t& mask)
        {
            svuint8_t lo = svmin_u8_x(mask, a, b);
            svuint8_t hi = svmax_u8_x(mask, a, b);
            a = lo;
            b = hi;
        }

        SIMD_INLINE svuint8_t Median5(svuint8_t a0, svuint8_t a1, svuint8_t a2, svuint8_t a3, svuint8_t a4, const svbool_t& mask)
        {
            SortU8(a2, a3, mask);
            SortU8(a1, a2, mask);
            SortU8(a2, a3, mask);
            a4 = svmax_u8_x(mask, a1, a4);
            a0 = svmin_u8_x(mask, a0, a3);
            SortU8(a2, a0, mask);
            a2 = svmax_u8_x(mask, a4, a2);
            a2 = svmin_u8_x(mask, a2, a0);
            return a2;
        }

        template <size_t step> SIMD_INLINE svuint8_t MedianFilterRhomb3x3(const uint8_t* y[3], size_t offset, const svbool_t& mask)
        {
            svuint8_t a0 = svld1_u8(mask, y[0] + offset);
            svuint8_t a1 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a2 = svld1_u8(mask, y[1] + offset);
            svuint8_t a3 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a4 = svld1_u8(mask, y[2] + offset);
            return Median5(a0, a1, a2, a3, a4, mask);
        }

        template <size_t step> void MedianFilterRhomb3x3(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(width > 1 && step * (width - 1) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t end = size - step;
            const uint8_t* y[3];

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[1] = src + srcStride * row;
                y[0] = row ? y[1] - srcStride : y[1];
                y[2] = row + 1 < height ? y[1] + srcStride : y[1];

                for (size_t col = 0; col < step; ++col)
                    dst[col] = Median5(y, col, col, col + step);

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, MedianFilterRhomb3x3<step>(y, col, mask));
                }

                for (size_t col = end; col < size; ++col)
                    dst[col] = Median5(y, col - step, col, col);
            }
        }

        void MedianFilterRhomb3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MedianFilterRhomb3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterRhomb3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterRhomb3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterRhomb3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
