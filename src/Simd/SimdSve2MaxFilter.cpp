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
        SIMD_INLINE uint8_t Max9(const uint8_t* y[3], size_t x[3], int threshold)
        {
            uint8_t a[9];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
            a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
            a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];

            uint8_t max = a[0];
            for (int i = 1; i < 9; ++i)
                max = max > a[i] ? max : a[i];

            if (1 >= threshold)
                return max;

            int num = 0;
            for (int i = 0; i < 9; ++i)
            {
                if (a[i] == max)
                    ++num;
            }
            return num >= threshold ? max : a[4];
        }

        template <size_t step> SIMD_INLINE void LoadSquare3x3(const uint8_t* y[3], size_t offset, svuint8_t a[9], const svbool_t& mask)
        {
            a[0] = svld1_u8(mask, y[0] + offset - step);
            a[1] = svld1_u8(mask, y[0] + offset);
            a[2] = svld1_u8(mask, y[0] + offset + step);
            a[3] = svld1_u8(mask, y[1] + offset - step);
            a[4] = svld1_u8(mask, y[1] + offset);
            a[5] = svld1_u8(mask, y[1] + offset + step);
            a[6] = svld1_u8(mask, y[2] + offset - step);
            a[7] = svld1_u8(mask, y[2] + offset);
            a[8] = svld1_u8(mask, y[2] + offset + step);
        }

        SIMD_INLINE svuint8_t Max9(svuint8_t a[9], int threshold, const svbool_t& mask)
        {
            svuint8_t max = a[0];
            for (int i = 1; i < 9; ++i)
                max = svmax_u8_x(mask, max, a[i]);

            if (1 >= threshold)
                return max;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            for (int i = 0; i < 9; ++i)
                count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a[i]), one, zero));

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), max, a[4]);
        }

        template <size_t step> void MaxFilterSquare3x3(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(width > 2 && step * (width - 2) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t end = size - step;
            const uint8_t* y[3];
            size_t x[3];
            svuint8_t a[9];

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[1] = src + srcStride * row;
                y[0] = row ? y[1] - srcStride : y[1];
                y[2] = row + 1 < height ? y[1] + srcStride : y[1];

                for (size_t col = 0; col < step; ++col)
                {
                    x[0] = col;
                    x[1] = col;
                    x[2] = col + step;
                    dst[col] = Max9(y, x, threshold);
                }

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    LoadSquare3x3<step>(y, col, a, mask);
                    svst1_u8(mask, dst + col, Max9(a, threshold, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - step;
                    x[1] = col;
                    x[2] = col;
                    dst[col] = Max9(y, x, threshold);
                }
            }
        }

        void MaxFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare3x3<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare3x3<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare3x3<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare3x3<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }
    }
#endif
}
