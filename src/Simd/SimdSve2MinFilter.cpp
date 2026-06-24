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
        SIMD_INLINE uint8_t Min9(const uint8_t* y[3], size_t x[3], int threshold)
        {
            uint8_t a[9];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
            a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
            a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];

            uint8_t min = a[0];
            for (int i = 1; i < 9; ++i)
                min = min < a[i] ? min : a[i];

            if (1 >= threshold)
                return min;

            int num = 0;
            for (int i = 0; i < 9; ++i)
            {
                if (a[i] == min)
                    ++num;
            }
            return num >= threshold ? min : a[4];
        }

        template <size_t step> SIMD_INLINE void LoadMinSquare3x3(const uint8_t* y[3], size_t offset,
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

        SIMD_INLINE svuint8_t Min9(const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2, const svuint8_t& a3, const svuint8_t& a4,
            const svuint8_t& a5, const svuint8_t& a6, const svuint8_t& a7, const svuint8_t& a8, int threshold, const svbool_t& mask)
        {
            svuint8_t min = a0;
            min = svmin_u8_x(mask, min, a1);
            min = svmin_u8_x(mask, min, a2);
            min = svmin_u8_x(mask, min, a3);
            min = svmin_u8_x(mask, min, a4);
            min = svmin_u8_x(mask, min, a5);
            min = svmin_u8_x(mask, min, a6);
            min = svmin_u8_x(mask, min, a7);
            min = svmin_u8_x(mask, min, a8);

            if (1 >= threshold)
                return min;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a0), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a1), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a2), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a3), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a4), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a5), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a6), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a7), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, a8), one, zero));

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), min, a4);
        }

        template <size_t step> void MinFilterSquare3x3(
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
                    dst[col] = Min9(y, x, threshold);
                }

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    LoadMinSquare3x3<step>(y, col, a0, a1, a2, a3, a4, a5, a6, a7, a8, mask);
                    svst1_u8(mask, dst + col, Min9(a0, a1, a2, a3, a4, a5, a6, a7, a8, threshold, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - step;
                    x[1] = col;
                    x[2] = col;
                    dst[col] = Min9(y, x, threshold);
                }
            }
        }

        void MinFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MinFilterSquare3x3<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MinFilterSquare3x3<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MinFilterSquare3x3<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MinFilterSquare3x3<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }

        SIMD_INLINE uint8_t Min25(const uint8_t* y[5], size_t x[5], int threshold)
        {
            uint8_t a[25];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]]; a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]]; a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]]; a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]]; a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]]; a[24] = y[4][x[4]];

            uint8_t min = a[0];
            for (int i = 1; i < 25; ++i)
                min = min < a[i] ? min : a[i];

            if (1 >= threshold)
                return min;

            int num = 0;
            for (int i = 0; i < 25; ++i)
            {
                if (a[i] == min)
                    ++num;
            }
            return num >= threshold ? min : a[12];
        }

        SIMD_INLINE void UpdateMin(const svuint8_t& value, svuint8_t& min, const svbool_t& mask)
        {
            min = svmin_u8_x(mask, min, value);
        }

        SIMD_INLINE void UpdateCount(const svuint8_t& value, const svuint8_t& min, svuint8_t& count, const svuint8_t& one, const svuint8_t& zero, const svbool_t& mask)
        {
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, min, value), one, zero));
        }

        template <size_t step> SIMD_INLINE svuint8_t Min25(const uint8_t* y[5], size_t offset, int threshold, const svbool_t& mask)
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
            svuint8_t min = a00;
            UpdateMin(a01, min, mask);
            UpdateMin(a02, min, mask);
            UpdateMin(a03, min, mask);
            UpdateMin(a04, min, mask);
            UpdateMin(a10, min, mask);
            UpdateMin(a11, min, mask);
            UpdateMin(a12, min, mask);
            UpdateMin(a13, min, mask);
            UpdateMin(a14, min, mask);
            UpdateMin(a20, min, mask);
            UpdateMin(a21, min, mask);
            UpdateMin(a22, min, mask);
            UpdateMin(a23, min, mask);
            UpdateMin(a24, min, mask);
            UpdateMin(a30, min, mask);
            UpdateMin(a31, min, mask);
            UpdateMin(a32, min, mask);
            UpdateMin(a33, min, mask);
            UpdateMin(a34, min, mask);
            UpdateMin(a40, min, mask);
            UpdateMin(a41, min, mask);
            UpdateMin(a42, min, mask);
            UpdateMin(a43, min, mask);
            UpdateMin(a44, min, mask);

            if (1 >= threshold)
                return min;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            UpdateCount(a00, min, count, one, zero, mask);
            UpdateCount(a01, min, count, one, zero, mask);
            UpdateCount(a02, min, count, one, zero, mask);
            UpdateCount(a03, min, count, one, zero, mask);
            UpdateCount(a04, min, count, one, zero, mask);
            UpdateCount(a10, min, count, one, zero, mask);
            UpdateCount(a11, min, count, one, zero, mask);
            UpdateCount(a12, min, count, one, zero, mask);
            UpdateCount(a13, min, count, one, zero, mask);
            UpdateCount(a14, min, count, one, zero, mask);
            UpdateCount(a20, min, count, one, zero, mask);
            UpdateCount(a21, min, count, one, zero, mask);
            UpdateCount(a22, min, count, one, zero, mask);
            UpdateCount(a23, min, count, one, zero, mask);
            UpdateCount(a24, min, count, one, zero, mask);
            UpdateCount(a30, min, count, one, zero, mask);
            UpdateCount(a31, min, count, one, zero, mask);
            UpdateCount(a32, min, count, one, zero, mask);
            UpdateCount(a33, min, count, one, zero, mask);
            UpdateCount(a34, min, count, one, zero, mask);
            UpdateCount(a40, min, count, one, zero, mask);
            UpdateCount(a41, min, count, one, zero, mask);
            UpdateCount(a42, min, count, one, zero, mask);
            UpdateCount(a43, min, count, one, zero, mask);
            UpdateCount(a44, min, count, one, zero, mask);

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), min, a22);
        }

        template <size_t step> void MinFilterSquare5x5(
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
                    dst[x[2]] = Min25(y, x, threshold);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, Min25<step>(y, col, threshold, mask));
                }
            }
        }

        void MinFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MinFilterSquare5x5<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MinFilterSquare5x5<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MinFilterSquare5x5<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MinFilterSquare5x5<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }
    }
#endif
}
