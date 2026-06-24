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

        SIMD_INLINE uint8_t Median9(const uint8_t* y[3], size_t x0, size_t x1, size_t x2)
        {
            int a[9];
            a[0] = y[0][x0]; a[1] = y[0][x1]; a[2] = y[0][x2];
            a[3] = y[1][x0]; a[4] = y[1][x1]; a[5] = y[1][x2];
            a[6] = y[2][x0]; a[7] = y[2][x1]; a[8] = y[2][x2];

            Base::SortU8(a[1], a[2]); Base::SortU8(a[4], a[5]); Base::SortU8(a[7], a[8]);
            Base::SortU8(a[0], a[1]); Base::SortU8(a[3], a[4]); Base::SortU8(a[6], a[7]);
            Base::SortU8(a[1], a[2]); Base::SortU8(a[4], a[5]); Base::SortU8(a[7], a[8]);
            a[3] = Base::MaxU8(a[0], a[3]);
            a[5] = Base::MinU8(a[5], a[8]);
            Base::SortU8(a[4], a[7]);
            a[6] = Base::MaxU8(a[3], a[6]);
            a[4] = Base::MaxU8(a[1], a[4]);
            a[2] = Base::MinU8(a[2], a[5]);
            a[4] = Base::MinU8(a[4], a[7]);
            Base::SortU8(a[4], a[2]);
            a[4] = Base::MaxU8(a[6], a[4]);
            a[4] = Base::MinU8(a[4], a[2]);

            return (uint8_t)a[4];
        }

        SIMD_INLINE svuint8_t Median9(svuint8_t a0, svuint8_t a1, svuint8_t a2, svuint8_t a3, svuint8_t a4,
            svuint8_t a5, svuint8_t a6, svuint8_t a7, svuint8_t a8, const svbool_t& mask)
        {
            SortU8(a1, a2, mask); SortU8(a4, a5, mask); SortU8(a7, a8, mask);
            SortU8(a0, a1, mask); SortU8(a3, a4, mask); SortU8(a6, a7, mask);
            SortU8(a1, a2, mask); SortU8(a4, a5, mask); SortU8(a7, a8, mask);
            a3 = svmax_u8_x(mask, a0, a3);
            a5 = svmin_u8_x(mask, a5, a8);
            SortU8(a4, a7, mask);
            a6 = svmax_u8_x(mask, a3, a6);
            a4 = svmax_u8_x(mask, a1, a4);
            a2 = svmin_u8_x(mask, a2, a5);
            a4 = svmin_u8_x(mask, a4, a7);
            SortU8(a4, a2, mask);
            a4 = svmax_u8_x(mask, a6, a4);
            return svmin_u8_x(mask, a4, a2);
        }

        template <size_t step> SIMD_INLINE svuint8_t MedianFilterSquare3x3(const uint8_t* y[3], size_t offset, const svbool_t& mask)
        {
            svuint8_t a0 = svld1_u8(mask, y[0] + offset - step);
            svuint8_t a1 = svld1_u8(mask, y[0] + offset);
            svuint8_t a2 = svld1_u8(mask, y[0] + offset + step);
            svuint8_t a3 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a4 = svld1_u8(mask, y[1] + offset);
            svuint8_t a5 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a6 = svld1_u8(mask, y[2] + offset - step);
            svuint8_t a7 = svld1_u8(mask, y[2] + offset);
            svuint8_t a8 = svld1_u8(mask, y[2] + offset + step);
            return Median9(a0, a1, a2, a3, a4, a5, a6, a7, a8, mask);
        }

        template <size_t step> void MedianFilterSquare3x3(
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
                    dst[col] = Median9(y, col, col, col + step);

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, MedianFilterSquare3x3<step>(y, col, mask));
                }

                for (size_t col = end; col < size; ++col)
                    dst[col] = Median9(y, col - step, col, col);
            }
        }

        void MedianFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MedianFilterSquare3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterSquare3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterSquare3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterSquare3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        SIMD_INLINE uint8_t Median13(const uint8_t* y[5], size_t x[5])
        {
            int a[13];
            a[0] = y[0][x[2]];
            a[1] = y[1][x[1]]; a[2] = y[1][x[2]]; a[3] = y[1][x[3]];
            a[4] = y[2][x[0]]; a[5] = y[2][x[1]]; a[6] = y[2][x[2]]; a[7] = y[2][x[3]]; a[8] = y[2][x[4]];
            a[9] = y[3][x[1]]; a[10] = y[3][x[2]]; a[11] = y[3][x[3]];
            a[12] = y[4][x[2]];

            Base::SortU8(a[0], a[1]); Base::SortU8(a[3], a[4]); Base::SortU8(a[2], a[4]);
            Base::SortU8(a[2], a[3]); Base::SortU8(a[6], a[7]); Base::SortU8(a[5], a[7]);
            Base::SortU8(a[5], a[6]); Base::SortU8(a[9], a[10]); Base::SortU8(a[8], a[10]);
            Base::SortU8(a[8], a[9]); Base::SortU8(a[11], a[12]); Base::SortU8(a[5], a[8]);
            Base::SortU8(a[2], a[8]); Base::SortU8(a[2], a[5]); Base::SortU8(a[6], a[9]);
            Base::SortU8(a[3], a[9]); Base::SortU8(a[3], a[6]); Base::SortU8(a[7], a[10]);
            Base::SortU8(a[4], a[10]); Base::SortU8(a[4], a[7]); Base::SortU8(a[3], a[12]);
            Base::SortU8(a[0], a[9]);
            a[1] = Base::MinU8(a[1], a[10]);
            a[1] = Base::MinU8(a[1], a[7]);
            a[1] = Base::MinU8(a[1], a[9]);
            a[11] = Base::MaxU8(a[5], a[11]);
            a[11] = Base::MaxU8(a[3], a[11]);
            a[11] = Base::MaxU8(a[2], a[11]);
            Base::SortU8(a[0], a[6]); Base::SortU8(a[1], a[8]); Base::SortU8(a[6], a[8]);
            a[4] = Base::MinU8(a[4], a[8]);
            Base::SortU8(a[0], a[1]); Base::SortU8(a[4], a[6]); Base::SortU8(a[0], a[4]);
            a[11] = Base::MaxU8(a[0], a[11]);
            Base::SortU8(a[6], a[11]);
            a[1] = Base::MinU8(a[1], a[11]);
            Base::SortU8(a[1], a[4]); Base::SortU8(a[6], a[12]);
            a[6] = Base::MaxU8(a[1], a[6]);
            a[4] = Base::MinU8(a[4], a[12]);
            a[6] = Base::MaxU8(a[4], a[6]);

            return (uint8_t)a[6];
        }

        SIMD_INLINE svuint8_t Median13(svuint8_t a0, svuint8_t a1, svuint8_t a2, svuint8_t a3, svuint8_t a4, svuint8_t a5, svuint8_t a6,
            svuint8_t a7, svuint8_t a8, svuint8_t a9, svuint8_t a10, svuint8_t a11, svuint8_t a12, const svbool_t& mask)
        {
            SortU8(a0, a1, mask); SortU8(a3, a4, mask); SortU8(a2, a4, mask);
            SortU8(a2, a3, mask); SortU8(a6, a7, mask); SortU8(a5, a7, mask);
            SortU8(a5, a6, mask); SortU8(a9, a10, mask); SortU8(a8, a10, mask);
            SortU8(a8, a9, mask); SortU8(a11, a12, mask); SortU8(a5, a8, mask);
            SortU8(a2, a8, mask); SortU8(a2, a5, mask); SortU8(a6, a9, mask);
            SortU8(a3, a9, mask); SortU8(a3, a6, mask); SortU8(a7, a10, mask);
            SortU8(a4, a10, mask); SortU8(a4, a7, mask); SortU8(a3, a12, mask);
            SortU8(a0, a9, mask);
            a1 = svmin_u8_x(mask, a1, a10);
            a1 = svmin_u8_x(mask, a1, a7);
            a1 = svmin_u8_x(mask, a1, a9);
            a11 = svmax_u8_x(mask, a5, a11);
            a11 = svmax_u8_x(mask, a3, a11);
            a11 = svmax_u8_x(mask, a2, a11);
            SortU8(a0, a6, mask); SortU8(a1, a8, mask); SortU8(a6, a8, mask);
            a4 = svmin_u8_x(mask, a4, a8);
            SortU8(a0, a1, mask); SortU8(a4, a6, mask); SortU8(a0, a4, mask);
            a11 = svmax_u8_x(mask, a0, a11);
            SortU8(a6, a11, mask);
            a1 = svmin_u8_x(mask, a1, a11);
            SortU8(a1, a4, mask); SortU8(a6, a12, mask);
            a6 = svmax_u8_x(mask, a1, a6);
            a4 = svmin_u8_x(mask, a4, a12);
            return svmax_u8_x(mask, a4, a6);
        }

        template <size_t step> SIMD_INLINE svuint8_t MedianFilterRhomb5x5(const uint8_t* y[5], size_t offset, const svbool_t& mask)
        {
            svuint8_t a0 = svld1_u8(mask, y[0] + offset);
            svuint8_t a1 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a2 = svld1_u8(mask, y[1] + offset);
            svuint8_t a3 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a4 = svld1_u8(mask, y[2] + offset - 2 * step);
            svuint8_t a5 = svld1_u8(mask, y[2] + offset - step);
            svuint8_t a6 = svld1_u8(mask, y[2] + offset);
            svuint8_t a7 = svld1_u8(mask, y[2] + offset + step);
            svuint8_t a8 = svld1_u8(mask, y[2] + offset + 2 * step);
            svuint8_t a9 = svld1_u8(mask, y[3] + offset - step);
            svuint8_t a10 = svld1_u8(mask, y[3] + offset);
            svuint8_t a11 = svld1_u8(mask, y[3] + offset + step);
            svuint8_t a12 = svld1_u8(mask, y[4] + offset);
            return Median13(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, mask);
        }

        template <size_t step> void MedianFilterRhomb5x5(
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

                for (size_t col = 0; col < body; ++col)
                {
                    x[0] = col < step ? col : col - step;
                    x[1] = x[0];
                    x[2] = col;
                    x[3] = col + step;
                    x[4] = col + 2 * step;
                    dst[col] = Median13(y, x);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, MedianFilterRhomb5x5<step>(y, col, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - 2 * step;
                    x[1] = col - step;
                    x[2] = col;
                    x[3] = col + step < size ? col + step : col;
                    x[4] = col + 2 * step < size ? col + 2 * step : x[3];
                    dst[col] = Median13(y, x);
                }
            }
        }

        void MedianFilterRhomb5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MedianFilterRhomb5x5<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterRhomb5x5<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterRhomb5x5<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterRhomb5x5<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        SIMD_INLINE uint8_t Median25(const uint8_t* y[5], size_t x[5])
        {
            int a[25];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]]; a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]]; a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]]; a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]]; a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]]; a[24] = y[4][x[4]];

            Base::SortU8(a[0], a[1]); Base::SortU8(a[3], a[4]); Base::SortU8(a[2], a[4]);
            Base::SortU8(a[2], a[3]); Base::SortU8(a[6], a[7]); Base::SortU8(a[5], a[7]);
            Base::SortU8(a[5], a[6]); Base::SortU8(a[9], a[10]); Base::SortU8(a[8], a[10]);
            Base::SortU8(a[8], a[9]); Base::SortU8(a[12], a[13]); Base::SortU8(a[11], a[13]);
            Base::SortU8(a[11], a[12]); Base::SortU8(a[15], a[16]); Base::SortU8(a[14], a[16]);
            Base::SortU8(a[14], a[15]); Base::SortU8(a[18], a[19]); Base::SortU8(a[17], a[19]);
            Base::SortU8(a[17], a[18]); Base::SortU8(a[21], a[22]); Base::SortU8(a[20], a[22]);
            Base::SortU8(a[20], a[21]); Base::SortU8(a[23], a[24]); Base::SortU8(a[2], a[5]);
            Base::SortU8(a[3], a[6]); Base::SortU8(a[0], a[6]); Base::SortU8(a[0], a[3]);
            Base::SortU8(a[4], a[7]); Base::SortU8(a[1], a[7]); Base::SortU8(a[1], a[4]);
            Base::SortU8(a[11], a[14]); Base::SortU8(a[8], a[14]); Base::SortU8(a[8], a[11]);
            Base::SortU8(a[12], a[15]); Base::SortU8(a[9], a[15]); Base::SortU8(a[9], a[12]);
            Base::SortU8(a[13], a[16]); Base::SortU8(a[10], a[16]); Base::SortU8(a[10], a[13]);
            Base::SortU8(a[20], a[23]); Base::SortU8(a[17], a[23]); Base::SortU8(a[17], a[20]);
            Base::SortU8(a[21], a[24]); Base::SortU8(a[18], a[24]); Base::SortU8(a[18], a[21]);
            Base::SortU8(a[19], a[22]); Base::SortU8(a[9], a[18]); Base::SortU8(a[0], a[18]);
            a[17] = Base::MaxU8(a[8], a[17]);
            a[9] = Base::MaxU8(a[0], a[9]);
            Base::SortU8(a[10], a[19]); Base::SortU8(a[1], a[19]); Base::SortU8(a[1], a[10]);
            Base::SortU8(a[11], a[20]); Base::SortU8(a[2], a[20]); Base::SortU8(a[12], a[21]);
            a[11] = Base::MaxU8(a[2], a[11]);
            Base::SortU8(a[3], a[21]); Base::SortU8(a[3], a[12]); Base::SortU8(a[13], a[22]);
            a[4] = Base::MinU8(a[4], a[22]);
            Base::SortU8(a[4], a[13]); Base::SortU8(a[14], a[23]);
            Base::SortU8(a[5], a[23]); Base::SortU8(a[5], a[14]); Base::SortU8(a[15], a[24]);
            a[6] = Base::MinU8(a[6], a[24]);
            Base::SortU8(a[6], a[15]);
            a[7] = Base::MinU8(a[7], a[16]);
            a[7] = Base::MinU8(a[7], a[19]);
            a[13] = Base::MinU8(a[13], a[21]);
            a[15] = Base::MinU8(a[15], a[23]);
            a[7] = Base::MinU8(a[7], a[13]);
            a[7] = Base::MinU8(a[7], a[15]);
            a[9] = Base::MaxU8(a[1], a[9]);
            a[11] = Base::MaxU8(a[3], a[11]);
            a[17] = Base::MaxU8(a[5], a[17]);
            a[17] = Base::MaxU8(a[11], a[17]);
            a[17] = Base::MaxU8(a[9], a[17]);
            Base::SortU8(a[4], a[10]);
            Base::SortU8(a[6], a[12]); Base::SortU8(a[7], a[14]); Base::SortU8(a[4], a[6]);
            a[7] = Base::MaxU8(a[4], a[7]);
            Base::SortU8(a[12], a[14]);
            a[10] = Base::MinU8(a[10], a[14]);
            Base::SortU8(a[6], a[7]); Base::SortU8(a[10], a[12]); Base::SortU8(a[6], a[10]);
            a[17] = Base::MaxU8(a[6], a[17]);
            Base::SortU8(a[12], a[17]);
            a[7] = Base::MinU8(a[7], a[17]);
            Base::SortU8(a[7], a[10]); Base::SortU8(a[12], a[18]);
            a[12] = Base::MaxU8(a[7], a[12]);
            a[10] = Base::MinU8(a[10], a[18]);
            Base::SortU8(a[12], a[20]);
            a[10] = Base::MinU8(a[10], a[20]);
            a[12] = Base::MaxU8(a[10], a[12]);

            return (uint8_t)a[12];
        }

        SIMD_INLINE svuint8_t Median25(svuint8_t a0, svuint8_t a1, svuint8_t a2, svuint8_t a3, svuint8_t a4,
            svuint8_t a5, svuint8_t a6, svuint8_t a7, svuint8_t a8, svuint8_t a9,
            svuint8_t a10, svuint8_t a11, svuint8_t a12, svuint8_t a13, svuint8_t a14,
            svuint8_t a15, svuint8_t a16, svuint8_t a17, svuint8_t a18, svuint8_t a19,
            svuint8_t a20, svuint8_t a21, svuint8_t a22, svuint8_t a23, svuint8_t a24, const svbool_t& mask)
        {
            SortU8(a0, a1, mask); SortU8(a3, a4, mask); SortU8(a2, a4, mask);
            SortU8(a2, a3, mask); SortU8(a6, a7, mask); SortU8(a5, a7, mask);
            SortU8(a5, a6, mask); SortU8(a9, a10, mask); SortU8(a8, a10, mask);
            SortU8(a8, a9, mask); SortU8(a12, a13, mask); SortU8(a11, a13, mask);
            SortU8(a11, a12, mask); SortU8(a15, a16, mask); SortU8(a14, a16, mask);
            SortU8(a14, a15, mask); SortU8(a18, a19, mask); SortU8(a17, a19, mask);
            SortU8(a17, a18, mask); SortU8(a21, a22, mask); SortU8(a20, a22, mask);
            SortU8(a20, a21, mask); SortU8(a23, a24, mask); SortU8(a2, a5, mask);
            SortU8(a3, a6, mask); SortU8(a0, a6, mask); SortU8(a0, a3, mask);
            SortU8(a4, a7, mask); SortU8(a1, a7, mask); SortU8(a1, a4, mask);
            SortU8(a11, a14, mask); SortU8(a8, a14, mask); SortU8(a8, a11, mask);
            SortU8(a12, a15, mask); SortU8(a9, a15, mask); SortU8(a9, a12, mask);
            SortU8(a13, a16, mask); SortU8(a10, a16, mask); SortU8(a10, a13, mask);
            SortU8(a20, a23, mask); SortU8(a17, a23, mask); SortU8(a17, a20, mask);
            SortU8(a21, a24, mask); SortU8(a18, a24, mask); SortU8(a18, a21, mask);
            SortU8(a19, a22, mask); SortU8(a9, a18, mask); SortU8(a0, a18, mask);
            a17 = svmax_u8_x(mask, a8, a17);
            a9 = svmax_u8_x(mask, a0, a9);
            SortU8(a10, a19, mask); SortU8(a1, a19, mask); SortU8(a1, a10, mask);
            SortU8(a11, a20, mask); SortU8(a2, a20, mask); SortU8(a12, a21, mask);
            a11 = svmax_u8_x(mask, a2, a11);
            SortU8(a3, a21, mask); SortU8(a3, a12, mask); SortU8(a13, a22, mask);
            a4 = svmin_u8_x(mask, a4, a22);
            SortU8(a4, a13, mask); SortU8(a14, a23, mask);
            SortU8(a5, a23, mask); SortU8(a5, a14, mask); SortU8(a15, a24, mask);
            a6 = svmin_u8_x(mask, a6, a24);
            SortU8(a6, a15, mask);
            a7 = svmin_u8_x(mask, a7, a16);
            a7 = svmin_u8_x(mask, a7, a19);
            a13 = svmin_u8_x(mask, a13, a21);
            a15 = svmin_u8_x(mask, a15, a23);
            a7 = svmin_u8_x(mask, a7, a13);
            a7 = svmin_u8_x(mask, a7, a15);
            a9 = svmax_u8_x(mask, a1, a9);
            a11 = svmax_u8_x(mask, a3, a11);
            a17 = svmax_u8_x(mask, a5, a17);
            a17 = svmax_u8_x(mask, a11, a17);
            a17 = svmax_u8_x(mask, a9, a17);
            SortU8(a4, a10, mask);
            SortU8(a6, a12, mask); SortU8(a7, a14, mask); SortU8(a4, a6, mask);
            a7 = svmax_u8_x(mask, a4, a7);
            SortU8(a12, a14, mask);
            a10 = svmin_u8_x(mask, a10, a14);
            SortU8(a6, a7, mask); SortU8(a10, a12, mask); SortU8(a6, a10, mask);
            a17 = svmax_u8_x(mask, a6, a17);
            SortU8(a12, a17, mask);
            a7 = svmin_u8_x(mask, a7, a17);
            SortU8(a7, a10, mask); SortU8(a12, a18, mask);
            a12 = svmax_u8_x(mask, a7, a12);
            a10 = svmin_u8_x(mask, a10, a18);
            SortU8(a12, a20, mask);
            a10 = svmin_u8_x(mask, a10, a20);
            return svmax_u8_x(mask, a10, a12);
        }

        SIMD_INLINE void Sort5(svuint8_t& a0, svuint8_t& a1, svuint8_t& a2, svuint8_t& a3, svuint8_t& a4, const svbool_t& mask)
        {
            SortU8(a0, a3, mask); SortU8(a1, a4, mask);
            SortU8(a0, a2, mask); SortU8(a1, a3, mask);
            SortU8(a0, a1, mask); SortU8(a2, a4, mask);
            SortU8(a1, a2, mask); SortU8(a3, a4, mask);
            SortU8(a2, a3, mask);
        }

        SIMD_INLINE void PartialSort20(svuint8_t& a0, svuint8_t& a1, svuint8_t& a2, svuint8_t& a3, svuint8_t& a4,
            svuint8_t& a5, svuint8_t& a6, svuint8_t& a7, svuint8_t& a8, svuint8_t& a9,
            svuint8_t& a10, svuint8_t& a11, svuint8_t& a12, svuint8_t& a13, svuint8_t& a14,
            svuint8_t& a15, svuint8_t& a16, svuint8_t& a17, svuint8_t& a18, svuint8_t& a19, const svbool_t& mask)
        {
            SortU8(a0, a3, mask); SortU8(a1, a7, mask); SortU8(a2, a5, mask);
            SortU8(a4, a8, mask); SortU8(a6, a9, mask); SortU8(a10, a13, mask);
            SortU8(a11, a15, mask); SortU8(a12, a18, mask); SortU8(a14, a17, mask);
            SortU8(a16, a19, mask); SortU8(a0, a14, mask); SortU8(a1, a11, mask);
            SortU8(a2, a16, mask); SortU8(a3, a17, mask); SortU8(a4, a12, mask);
            SortU8(a5, a19, mask); SortU8(a6, a10, mask); SortU8(a7, a15, mask);
            SortU8(a8, a18, mask); SortU8(a9, a13, mask); SortU8(a0, a4, mask);
            SortU8(a1, a2, mask); SortU8(a3, a8, mask); SortU8(a5, a7, mask);
            SortU8(a11, a16, mask); SortU8(a12, a14, mask); SortU8(a15, a19, mask);
            SortU8(a17, a18, mask); SortU8(a1, a6, mask); SortU8(a2, a12, mask);
            SortU8(a3, a5, mask); SortU8(a4, a11, mask); SortU8(a7, a17, mask);
            SortU8(a8, a15, mask); SortU8(a13, a18, mask); SortU8(a14, a16, mask);
            a1 = svmax_u8_x(mask, a0, a1); SortU8(a2, a6, mask);
            SortU8(a7, a10, mask); SortU8(a9, a12, mask); SortU8(a13, a17, mask);
            a18 = svmin_u8_x(mask, a18, a19); SortU8(a1, a6, mask);
            SortU8(a5, a9, mask); SortU8(a7, a11, mask); SortU8(a8, a12, mask);
            SortU8(a10, a14, mask); SortU8(a13, a18, mask); SortU8(a3, a5, mask);
            SortU8(a4, a7, mask); SortU8(a8, a10, mask); SortU8(a9, a11, mask);
            SortU8(a12, a15, mask); SortU8(a14, a16, mask);
            a3 = svmax_u8_x(mask, a1, a3); a4 = svmax_u8_x(mask, a2, a4);
            SortU8(a5, a7, mask); SortU8(a6, a10, mask); SortU8(a9, a13, mask);
            SortU8(a12, a14, mask); a15 = svmin_u8_x(mask, a15, a17);
            a16 = svmin_u8_x(mask, a16, a18); a4 = svmax_u8_x(mask, a3, a4);
            SortU8(a6, a7, mask); SortU8(a8, a9, mask); SortU8(a10, a11, mask);
            SortU8(a12, a13, mask); a15 = svmin_u8_x(mask, a15, a16);
            a6 = svmax_u8_x(mask, a4, a6); a8 = svmax_u8_x(mask, a5, a8);
            SortU8(a7, a9, mask); SortU8(a10, a12, mask);
            a11 = svmin_u8_x(mask, a11, a14); a13 = svmin_u8_x(mask, a13, a15);
            a8 = svmax_u8_x(mask, a6, a8); SortU8(a7, a10, mask);
            SortU8(a9, a12, mask); a11 = svmin_u8_x(mask, a11, a13);
            SortU8(a7, a8, mask); SortU8(a9, a10, mask); SortU8(a11, a12, mask);
        }

        SIMD_INLINE void Sort25x2(svuint8_t& x0, svuint8_t& x1, svuint8_t& x2, svuint8_t& x3, svuint8_t& x4,
            svuint8_t& y0, svuint8_t& y1, svuint8_t& y2, svuint8_t& y3, svuint8_t& y4,
            svuint8_t& y5, svuint8_t& y6, svuint8_t& y7, svuint8_t& y8, svuint8_t& y9,
            svuint8_t& y10, svuint8_t& y11, svuint8_t& y12, svuint8_t& y13, svuint8_t& y14,
            svuint8_t& y15, svuint8_t& y16, svuint8_t& y17, svuint8_t& y18, svuint8_t& y19,
            svuint8_t& z0, svuint8_t& z1, svuint8_t& z2, svuint8_t& z3, svuint8_t& z4,
            svuint8_t& dst0, svuint8_t& dst1, const svbool_t& mask)
        {
            Sort5(x0, x1, x2, x3, x4, mask);
            Sort5(z0, z1, z2, z3, z4, mask);
            PartialSort20(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, mask);

            dst0 = svmin_u8_x(mask, x0, y12);
            dst0 = svmax_u8_x(mask, dst0, y11);
            dst0 = svmin_u8_x(mask, dst0, svmax_u8_x(mask, x1, y10));
            dst0 = svmin_u8_x(mask, dst0, svmax_u8_x(mask, x2, y9));
            dst0 = svmin_u8_x(mask, dst0, svmax_u8_x(mask, x3, y8));
            dst0 = svmin_u8_x(mask, dst0, svmax_u8_x(mask, x4, y7));

            dst1 = svmin_u8_x(mask, z0, y12);
            dst1 = svmax_u8_x(mask, dst1, y11);
            dst1 = svmin_u8_x(mask, dst1, svmax_u8_x(mask, z1, y10));
            dst1 = svmin_u8_x(mask, dst1, svmax_u8_x(mask, z2, y9));
            dst1 = svmin_u8_x(mask, dst1, svmax_u8_x(mask, z3, y8));
            dst1 = svmin_u8_x(mask, dst1, svmax_u8_x(mask, z4, y7));
        }

        template <size_t step> SIMD_INLINE svuint8_t MedianFilterSquare5x5(const uint8_t* y[5], size_t offset, const svbool_t& mask)
        {
            svuint8_t a0 = svld1_u8(mask, y[0] + offset - 2 * step);
            svuint8_t a1 = svld1_u8(mask, y[0] + offset - step);
            svuint8_t a2 = svld1_u8(mask, y[0] + offset);
            svuint8_t a3 = svld1_u8(mask, y[0] + offset + step);
            svuint8_t a4 = svld1_u8(mask, y[0] + offset + 2 * step);
            svuint8_t a5 = svld1_u8(mask, y[1] + offset - 2 * step);
            svuint8_t a6 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a7 = svld1_u8(mask, y[1] + offset);
            svuint8_t a8 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a9 = svld1_u8(mask, y[1] + offset + 2 * step);
            svuint8_t a10 = svld1_u8(mask, y[2] + offset - 2 * step);
            svuint8_t a11 = svld1_u8(mask, y[2] + offset - step);
            svuint8_t a12 = svld1_u8(mask, y[2] + offset);
            svuint8_t a13 = svld1_u8(mask, y[2] + offset + step);
            svuint8_t a14 = svld1_u8(mask, y[2] + offset + 2 * step);
            svuint8_t a15 = svld1_u8(mask, y[3] + offset - 2 * step);
            svuint8_t a16 = svld1_u8(mask, y[3] + offset - step);
            svuint8_t a17 = svld1_u8(mask, y[3] + offset);
            svuint8_t a18 = svld1_u8(mask, y[3] + offset + step);
            svuint8_t a19 = svld1_u8(mask, y[3] + offset + 2 * step);
            svuint8_t a20 = svld1_u8(mask, y[4] + offset - 2 * step);
            svuint8_t a21 = svld1_u8(mask, y[4] + offset - step);
            svuint8_t a22 = svld1_u8(mask, y[4] + offset);
            svuint8_t a23 = svld1_u8(mask, y[4] + offset + step);
            svuint8_t a24 = svld1_u8(mask, y[4] + offset + 2 * step);
            return Median25(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, mask);
        }

        template <size_t step> SIMD_INLINE void MedianFilterSquare5x5x2(const uint8_t* y[6], size_t offset,
            svuint8_t& dst0, svuint8_t& dst1, const svbool_t& mask)
        {
            svuint8_t a0 = svld1_u8(mask, y[0] + offset - 2 * step);
            svuint8_t a1 = svld1_u8(mask, y[0] + offset - step);
            svuint8_t a2 = svld1_u8(mask, y[0] + offset);
            svuint8_t a3 = svld1_u8(mask, y[0] + offset + step);
            svuint8_t a4 = svld1_u8(mask, y[0] + offset + 2 * step);
            svuint8_t a5 = svld1_u8(mask, y[1] + offset - 2 * step);
            svuint8_t a6 = svld1_u8(mask, y[1] + offset - step);
            svuint8_t a7 = svld1_u8(mask, y[1] + offset);
            svuint8_t a8 = svld1_u8(mask, y[1] + offset + step);
            svuint8_t a9 = svld1_u8(mask, y[1] + offset + 2 * step);
            svuint8_t a10 = svld1_u8(mask, y[2] + offset - 2 * step);
            svuint8_t a11 = svld1_u8(mask, y[2] + offset - step);
            svuint8_t a12 = svld1_u8(mask, y[2] + offset);
            svuint8_t a13 = svld1_u8(mask, y[2] + offset + step);
            svuint8_t a14 = svld1_u8(mask, y[2] + offset + 2 * step);
            svuint8_t a15 = svld1_u8(mask, y[3] + offset - 2 * step);
            svuint8_t a16 = svld1_u8(mask, y[3] + offset - step);
            svuint8_t a17 = svld1_u8(mask, y[3] + offset);
            svuint8_t a18 = svld1_u8(mask, y[3] + offset + step);
            svuint8_t a19 = svld1_u8(mask, y[3] + offset + 2 * step);
            svuint8_t a20 = svld1_u8(mask, y[4] + offset - 2 * step);
            svuint8_t a21 = svld1_u8(mask, y[4] + offset - step);
            svuint8_t a22 = svld1_u8(mask, y[4] + offset);
            svuint8_t a23 = svld1_u8(mask, y[4] + offset + step);
            svuint8_t a24 = svld1_u8(mask, y[4] + offset + 2 * step);
            svuint8_t a25 = svld1_u8(mask, y[5] + offset - 2 * step);
            svuint8_t a26 = svld1_u8(mask, y[5] + offset - step);
            svuint8_t a27 = svld1_u8(mask, y[5] + offset);
            svuint8_t a28 = svld1_u8(mask, y[5] + offset + step);
            svuint8_t a29 = svld1_u8(mask, y[5] + offset + 2 * step);

            Sort25x2(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, dst0, dst1, mask);
        }

        template <size_t step> void MedianFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(width > 4 && step * (width - 4) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t body = 2 * step;
            const size_t end = size - body;
            const uint8_t* y[6];
            size_t x[5];

            size_t row = 0;
            for (; row + 1 < height; row += 2, dst += 2 * dstStride)
            {
                y[2] = src + srcStride * row;
                y[1] = row ? y[2] - srcStride : y[2];
                y[0] = row > 1 ? y[2] - 2 * srcStride : y[1];
                y[3] = row + 1 < height ? y[2] + srcStride : y[2];
                y[4] = row + 2 < height ? y[2] + 2 * srcStride : y[3];
                y[5] = row + 3 < height ? y[2] + 3 * srcStride : y[4];

                for (size_t col = 0; col < body; ++col)
                {
                    x[0] = col < step ? col : col - step;
                    x[1] = x[0];
                    x[2] = col;
                    x[3] = col + step;
                    x[4] = col + 2 * step;
                    dst[col] = Median25(y, x);
                    dst[dstStride + col] = Median25(y + 1, x);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svuint8_t dst0, dst1;
                    svbool_t mask = svwhilelt_b8(col, end);
                    MedianFilterSquare5x5x2<step>(y, col, dst0, dst1, mask);
                    svst1_u8(mask, dst + col, dst0);
                    svst1_u8(mask, dst + dstStride + col, dst1);
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - 2 * step;
                    x[1] = col - step;
                    x[2] = col;
                    x[3] = col + step < size ? col + step : col;
                    x[4] = x[3];
                    dst[col] = Median25(y, x);
                    dst[dstStride + col] = Median25(y + 1, x);
                }
            }

            for (; row < height; ++row, dst += dstStride)
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
                    dst[col] = Median25(y, x);
                }

                for (size_t col = body; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    svst1_u8(mask, dst + col, MedianFilterSquare5x5<step>(y, col, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - 2 * step;
                    x[1] = col - step;
                    x[2] = col;
                    x[3] = col + step < size ? col + step : col;
                    x[4] = x[3];
                    dst[col] = Median25(y, x);
                }
            }
        }

        void MedianFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MedianFilterSquare5x5<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterSquare5x5<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterSquare5x5<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterSquare5x5<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
