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
        SIMD_INLINE uint8_t Midpoint9(const uint8_t* y[3], size_t x[3])
        {
            uint8_t a[9];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
            a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
            a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];

            uint8_t min = a[0], max = a[0];
            for (int i = 1; i < 9; ++i)
            {
                min = min < a[i] ? min : a[i];
                max = max > a[i] ? max : a[i];
            }
            return uint8_t((min + max + 1) >> 1);
        }

        template <size_t step> SIMD_INLINE void LoadSquare3x3(const uint8_t* y[3], size_t offset,
            svuint8_t& a0, svuint8_t& a1, svuint8_t& a2, svuint8_t& a3, svuint8_t& a4, svuint8_t& a5, svuint8_t& a6, svuint8_t& a7, svuint8_t& a8,
            const svbool_t& mask)
        {
            a0 = svld1_u8(mask, y[0] + offset - step);
            a1 = svld1_u8(mask, y[0] + offset);
            a2 = svld1_u8(mask, y[0] + offset + step);
            a3 = svld1_u8(mask, y[1] + offset - step);
            a4 = svld1_u8(mask, y[1] + offset);
            a5 = svld1_u8(mask, y[1] + offset + step);
            a6 = svld1_u8(mask, y[2] + offset - step);
            a7 = svld1_u8(mask, y[2] + offset);
            a8 = svld1_u8(mask, y[2] + offset + step);
        }

        SIMD_INLINE svuint8_t Midpoint9(const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2, const svuint8_t& a3, const svuint8_t& a4,
            const svuint8_t& a5, const svuint8_t& a6, const svuint8_t& a7, const svuint8_t& a8, const svbool_t& mask)
        {
            svuint8_t min = a0, max = a0;
            min = svmin_u8_x(mask, min, a1);
            max = svmax_u8_x(mask, max, a1);
            min = svmin_u8_x(mask, min, a2);
            max = svmax_u8_x(mask, max, a2);
            min = svmin_u8_x(mask, min, a3);
            max = svmax_u8_x(mask, max, a3);
            min = svmin_u8_x(mask, min, a4);
            max = svmax_u8_x(mask, max, a4);
            min = svmin_u8_x(mask, min, a5);
            max = svmax_u8_x(mask, max, a5);
            min = svmin_u8_x(mask, min, a6);
            max = svmax_u8_x(mask, max, a6);
            min = svmin_u8_x(mask, min, a7);
            max = svmax_u8_x(mask, max, a7);
            min = svmin_u8_x(mask, min, a8);
            max = svmax_u8_x(mask, max, a8);
            return svrhadd_u8_x(mask, min, max);
        }

        template <size_t step> void MidpointFilterSquare3x3(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(width > 2 && step * (width - 2) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t end = size - step;
            const uint8_t* y[3];
            size_t x[3];
            svuint8_t a0, a1, a2, a3, a4, a5, a6, a7, a8;

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
                    dst[col] = Midpoint9(y, x);
                }

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    LoadSquare3x3<step>(y, col, a0, a1, a2, a3, a4, a5, a6, a7, a8, mask);
                    svst1_u8(mask, dst + col, Midpoint9(a0, a1, a2, a3, a4, a5, a6, a7, a8, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - step;
                    x[1] = col;
                    x[2] = col;
                    dst[col] = Midpoint9(y, x);
                }
            }
        }

        void MidpointFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MidpointFilterSquare3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MidpointFilterSquare3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MidpointFilterSquare3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MidpointFilterSquare3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        SIMD_INLINE uint8_t Midpoint25(const uint8_t* y[5], size_t x[5])
        {
            uint8_t a[25];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]]; a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]]; a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]]; a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]]; a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]]; a[24] = y[4][x[4]];

            uint8_t min = a[0], max = a[0];
            for (int i = 1; i < 25; ++i)
            {
                min = min < a[i] ? min : a[i];
                max = max > a[i] ? max : a[i];
            }
            return uint8_t((min + max + 1) >> 1);
        }

        SIMD_INLINE void UpdateMinMax(const svuint8_t& value, svuint8_t& min, svuint8_t& max, const svbool_t& mask)
        {
            min = svmin_u8_x(mask, min, value);
            max = svmax_u8_x(mask, max, value);
        }

        template <size_t step> SIMD_INLINE svuint8_t Midpoint25(const uint8_t* y[5], size_t offset, const svbool_t& mask)
        {
            svuint8_t a00 = svld1_u8(mask, y[0] + offset - 2 * step);
            svuint8_t a01 = svld1_u8(mask, y[0] + offset - step);
            svuint8_t a02 = svld1_u8(mask, y[0] + offset);
            svuint8_t a03 = svld1_u8(mask, y[0] + offset + step);
            svuint8_t a04 = svld1_u8(mask, y[0] + offset + 2 * step);
            svuint8_t a10 = svld1_u8(mask, y[1] + offset - 2 * step);
            svuint8_t a11 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a12 = svld1_u8(mask, y[1] + offset);
            svuint8_t a13 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a14 = svld1_u8(mask, y[1] + offset + 2 * step);
            svuint8_t a20 = svld1_u8(mask, y[2] + offset - 2 * step);
            svuint8_t a21 = svld1_u8(mask, y[2] + offset - step);
            svuint8_t a22 = svld1_u8(mask, y[2] + offset);
            svuint8_t a23 = svld1_u8(mask, y[2] + offset + step);
            svuint8_t a24 = svld1_u8(mask, y[2] + offset + 2 * step);
            svuint8_t a30 = svld1_u8(mask, y[3] + offset - 2 * step);
            svuint8_t a31 = svld1_u8(mask, y[3] + offset - step);
            svuint8_t a32 = svld1_u8(mask, y[3] + offset);
            svuint8_t a33 = svld1_u8(mask, y[3] + offset + step);
            svuint8_t a34 = svld1_u8(mask, y[3] + offset + 2 * step);
            svuint8_t a40 = svld1_u8(mask, y[4] + offset - 2 * step);
            svuint8_t a41 = svld1_u8(mask, y[4] + offset - step);
            svuint8_t a42 = svld1_u8(mask, y[4] + offset);
            svuint8_t a43 = svld1_u8(mask, y[4] + offset + step);
            svuint8_t a44 = svld1_u8(mask, y[4] + offset + 2 * step);
            svuint8_t min = a00, max = a00;
            UpdateMinMax(a01, min, max, mask);
            UpdateMinMax(a02, min, max, mask);
            UpdateMinMax(a03, min, max, mask);
            UpdateMinMax(a04, min, max, mask);
            UpdateMinMax(a10, min, max, mask);
            UpdateMinMax(a11, min, max, mask);
            UpdateMinMax(a12, min, max, mask);
            UpdateMinMax(a13, min, max, mask);
            UpdateMinMax(a14, min, max, mask);
            UpdateMinMax(a20, min, max, mask);
            UpdateMinMax(a21, min, max, mask);
            UpdateMinMax(a22, min, max, mask);
            UpdateMinMax(a23, min, max, mask);
            UpdateMinMax(a24, min, max, mask);
            UpdateMinMax(a30, min, max, mask);
            UpdateMinMax(a31, min, max, mask);
            UpdateMinMax(a32, min, max, mask);
            UpdateMinMax(a33, min, max, mask);
            UpdateMinMax(a34, min, max, mask);
            UpdateMinMax(a40, min, max, mask);
            UpdateMinMax(a41, min, max, mask);
            UpdateMinMax(a42, min, max, mask);
            UpdateMinMax(a43, min, max, mask);
            UpdateMinMax(a44, min, max, mask);
            return svrhadd_u8_x(mask, min, max);
        }

        template <size_t step> void MidpointFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(width > 4 && step * (width - 4) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t body = 2 * step;
            const size_t end = size - body;
            const uint8_t* y[5];
            size_t x[5];

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[2] = src + srcStride * row;
                y[1] = row ? y[2] - srcStride : y[2];
                y[0] = row > 1 ? y[2] - 2 * srcStride : y[1];
                y[3] = row + 1 < height ? y[2] + srcStride : y[2];
                y[4] = row + 2 < height ? y[2] + 2 * srcStride : y[3];

                for (size_t col = 0; col < 2 * body; ++col)
                {
                    if (col < body)
                    {
                        x[0] = col < step ? col : col - step;
                        x[1] = x[0];
                        x[2] = col;
                        x[3] = x[2] + step;
                        x[4] = x[3] + step;
                    }
                    else
                    {
                        x[0] = size - 6 * step + col;
                        x[1] = x[0] + step;
                        x[2] = x[1] + step;
                        x[3] = col < 3 * step ? x[2] + step : x[2];
                        x[4] = x[3];
                    }
                    dst[x[2]] = Midpoint25(y, x);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, Midpoint25<step>(y, col, mask));
                }
            }
        }

        void MidpointFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MidpointFilterSquare5x5<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MidpointFilterSquare5x5<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MidpointFilterSquare5x5<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MidpointFilterSquare5x5<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
