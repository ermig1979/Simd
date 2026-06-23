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

        SIMD_INLINE svuint8_t Max9(const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2, const svuint8_t& a3, const svuint8_t& a4,
            const svuint8_t& a5, const svuint8_t& a6, const svuint8_t& a7, const svuint8_t& a8, int threshold, const svbool_t& mask)
        {
            svuint8_t max = a0;
            max = svmax_u8_x(mask, max, a1);
            max = svmax_u8_x(mask, max, a2);
            max = svmax_u8_x(mask, max, a3);
            max = svmax_u8_x(mask, max, a4);
            max = svmax_u8_x(mask, max, a5);
            max = svmax_u8_x(mask, max, a6);
            max = svmax_u8_x(mask, max, a7);
            max = svmax_u8_x(mask, max, a8);

            if (1 >= threshold)
                return max;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a0), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a1), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a2), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a3), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a4), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a5), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a6), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a7), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a8), one, zero));

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), max, a4);
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
                    dst[col] = Max9(y, x, threshold);
                }

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    LoadSquare3x3<step>(y, col, a0, a1, a2, a3, a4, a5, a6, a7, a8, mask);
                    svst1_u8(mask, dst + col, Max9(a0, a1, a2, a3, a4, a5, a6, a7, a8, threshold, mask));
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

        SIMD_INLINE uint8_t Max25(const uint8_t* y[5], size_t x[5], int threshold)
        {
            uint8_t a[25];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]]; a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]]; a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]]; a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]]; a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]]; a[24] = y[4][x[4]];

            uint8_t max = a[0];
            for (int i = 1; i < 25; ++i)
                max = max > a[i] ? max : a[i];

            if (1 >= threshold)
                return max;

            int num = 0;
            for (int i = 0; i < 25; ++i)
            {
                if (a[i] == max)
                    ++num;
            }
            return num >= threshold ? max : a[12];
        }

        SIMD_INLINE void UpdateMax(const svuint8_t& value, svuint8_t& max, const svbool_t& mask)
        {
            max = svmax_u8_x(mask, max, value);
        }

        SIMD_INLINE void UpdateCount(const svuint8_t& value, const svuint8_t& max, svuint8_t& count, const svuint8_t& one, const svuint8_t& zero, const svbool_t& mask)
        {
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, value), one, zero));
        }

        template <size_t step> SIMD_INLINE svuint8_t Max25(const uint8_t* y[5], size_t offset, int threshold, const svbool_t& mask)
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
            svuint8_t max = a00;
            UpdateMax(a01, max, mask);
            UpdateMax(a02, max, mask);
            UpdateMax(a03, max, mask);
            UpdateMax(a04, max, mask);
            UpdateMax(a10, max, mask);
            UpdateMax(a11, max, mask);
            UpdateMax(a12, max, mask);
            UpdateMax(a13, max, mask);
            UpdateMax(a14, max, mask);
            UpdateMax(a20, max, mask);
            UpdateMax(a21, max, mask);
            UpdateMax(a22, max, mask);
            UpdateMax(a23, max, mask);
            UpdateMax(a24, max, mask);
            UpdateMax(a30, max, mask);
            UpdateMax(a31, max, mask);
            UpdateMax(a32, max, mask);
            UpdateMax(a33, max, mask);
            UpdateMax(a34, max, mask);
            UpdateMax(a40, max, mask);
            UpdateMax(a41, max, mask);
            UpdateMax(a42, max, mask);
            UpdateMax(a43, max, mask);
            UpdateMax(a44, max, mask);

            if (1 >= threshold)
                return max;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            UpdateCount(a00, max, count, one, zero, mask);
            UpdateCount(a01, max, count, one, zero, mask);
            UpdateCount(a02, max, count, one, zero, mask);
            UpdateCount(a03, max, count, one, zero, mask);
            UpdateCount(a04, max, count, one, zero, mask);
            UpdateCount(a10, max, count, one, zero, mask);
            UpdateCount(a11, max, count, one, zero, mask);
            UpdateCount(a12, max, count, one, zero, mask);
            UpdateCount(a13, max, count, one, zero, mask);
            UpdateCount(a14, max, count, one, zero, mask);
            UpdateCount(a20, max, count, one, zero, mask);
            UpdateCount(a21, max, count, one, zero, mask);
            UpdateCount(a22, max, count, one, zero, mask);
            UpdateCount(a23, max, count, one, zero, mask);
            UpdateCount(a24, max, count, one, zero, mask);
            UpdateCount(a30, max, count, one, zero, mask);
            UpdateCount(a31, max, count, one, zero, mask);
            UpdateCount(a32, max, count, one, zero, mask);
            UpdateCount(a33, max, count, one, zero, mask);
            UpdateCount(a34, max, count, one, zero, mask);
            UpdateCount(a40, max, count, one, zero, mask);
            UpdateCount(a41, max, count, one, zero, mask);
            UpdateCount(a42, max, count, one, zero, mask);
            UpdateCount(a43, max, count, one, zero, mask);
            UpdateCount(a44, max, count, one, zero, mask);

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), max, a22);
        }

        template <size_t step> void MaxFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(width > 4 && step * (width - 4) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t body = 2 * step;
            const size_t end = size - 2 * step;
            const uint8_t* y[5];
            size_t x[5];

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[2] = src + srcStride * row;
                y[1] = row ? y[2] - srcStride : y[2];
                y[0] = row > 1 ? y[2] - 2 * srcStride : y[1];
                y[3] = row + 1 < height ? y[2] + srcStride : y[2];
                y[4] = row + 2 < height ? y[2] + 2 * srcStride : y[3];

                for (size_t col = 0; col < body; ++col)
                {
                    x[0] = col < step ? col : col - step;
                    x[1] = x[0];
                    x[2] = col;
                    x[3] = col + step;
                    x[4] = col + 2 * step;
                    dst[col] = Max25(y, x, threshold);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, Max25<step>(y, col, threshold, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - 2 * step;
                    x[1] = col - step;
                    x[2] = col;
                    x[3] = col + step < size ? col + step : col;
                    x[4] = col + 2 * step < size ? col + 2 * step : x[3];
                    dst[col] = Max25(y, x, threshold);
                }
            }
        }

        void MaxFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare5x5<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare5x5<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare5x5<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare5x5<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }
    }
#endif
}
