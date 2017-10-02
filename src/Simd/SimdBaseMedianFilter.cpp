/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
    namespace Base
    {
        SIMD_INLINE void LoadRhomb3x3(const uint8_t * y[3], size_t x[3], int a[5])
        {
            a[0] = y[0][x[1]];
            a[1] = y[1][x[0]]; a[2] = y[1][x[1]]; a[3] = y[1][x[2]];
            a[4] = y[2][x[1]];
        }

        SIMD_INLINE void PartialSort5(int a[5])
        {
            SortU8(a[2], a[3]);
            SortU8(a[1], a[2]);
            SortU8(a[2], a[3]);
            a[4] = MaxU8(a[1], a[4]);
            a[0] = MinU8(a[0], a[3]);
            SortU8(a[2], a[0]);
            a[2] = MaxU8(a[4], a[2]);
            a[2] = MinU8(a[2], a[0]);
        }

        void MedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            int a[5];
            const uint8_t * y[3];
            size_t x[3];

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 1);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                if (row < 1)
                    y[0] = y[1];
                if (row >= height - 1)
                    y[2] = y[1];

                for (size_t col = 0; col < 2 * channelCount; col++)
                {
                    x[0] = col < channelCount ? col : size - 3 * channelCount + col;
                    x[2] = col < channelCount ? col + channelCount : size - 2 * channelCount + col;
                    x[1] = col < channelCount ? x[0] : x[2];

                    LoadRhomb3x3(y, x, a);
                    PartialSort5(a);
                    dst[x[1]] = (uint8_t)a[2];
                }

                for (size_t col = channelCount; col < size - channelCount; ++col)
                {
                    x[0] = col - channelCount;
                    x[1] = col;
                    x[2] = col + channelCount;

                    LoadRhomb3x3(y, x, a);
                    PartialSort5(a);
                    dst[col] = (uint8_t)a[2];
                }
            }
        }

        SIMD_INLINE void LoadSquare3x3(const uint8_t * y[3], size_t x[3], int a[9])
        {
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
            a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
            a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];
        }

        SIMD_INLINE void PartialSort9(int a[9])
        {
            SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]);
            SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[6], a[7]);
            SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]);
            a[3] = MaxU8(a[0], a[3]);
            a[5] = MinU8(a[5], a[8]);
            SortU8(a[4], a[7]);
            a[6] = MaxU8(a[3], a[6]);
            a[4] = MaxU8(a[1], a[4]);
            a[2] = MinU8(a[2], a[5]);
            a[4] = MinU8(a[4], a[7]);
            SortU8(a[4], a[2]);
            a[4] = MaxU8(a[6], a[4]);
            a[4] = MinU8(a[4], a[2]);
        }

        void MedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            int a[9];
            const uint8_t * y[3];
            size_t x[3];

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 1);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                if (row < 1)
                    y[0] = y[1];
                if (row >= height - 1)
                    y[2] = y[1];

                for (size_t col = 0; col < 2 * channelCount; col++)
                {
                    x[0] = col < channelCount ? col : size - 3 * channelCount + col;
                    x[2] = col < channelCount ? col + channelCount : size - 2 * channelCount + col;
                    x[1] = col < channelCount ? x[0] : x[2];

                    LoadSquare3x3(y, x, a);
                    PartialSort9(a);
                    dst[x[1]] = (uint8_t)a[4];
                }

                for (size_t col = channelCount; col < size - channelCount; ++col)
                {
                    x[0] = col - channelCount;
                    x[1] = col;
                    x[2] = col + channelCount;

                    LoadSquare3x3(y, x, a);
                    PartialSort9(a);
                    dst[col] = (uint8_t)a[4];
                }
            }
        }

        SIMD_INLINE void LoadRhomb5x5(const uint8_t * y[5], size_t x[5], int a[13])
        {
            a[0] = y[0][x[2]];
            a[1] = y[1][x[1]]; a[2] = y[1][x[2]]; a[3] = y[1][x[3]];
            a[4] = y[2][x[0]]; a[5] = y[2][x[1]]; a[6] = y[2][x[2]]; a[7] = y[2][x[3]]; a[8] = y[2][x[4]];
            a[9] = y[3][x[1]]; a[10] = y[3][x[2]]; a[11] = y[3][x[3]];
            a[12] = y[4][x[2]];
        }

        SIMD_INLINE void PartialSort13(int a[13])
        {
            SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[2], a[4]);
            SortU8(a[2], a[3]); SortU8(a[6], a[7]); SortU8(a[5], a[7]);
            SortU8(a[5], a[6]); SortU8(a[9], a[10]); SortU8(a[8], a[10]);
            SortU8(a[8], a[9]); SortU8(a[11], a[12]); SortU8(a[5], a[8]);
            SortU8(a[2], a[8]); SortU8(a[2], a[5]); SortU8(a[6], a[9]);
            SortU8(a[3], a[9]); SortU8(a[3], a[6]); SortU8(a[7], a[10]);
            SortU8(a[4], a[10]); SortU8(a[4], a[7]); SortU8(a[3], a[12]);
            SortU8(a[0], a[9]);
            a[1] = MinU8(a[1], a[10]);
            a[1] = MinU8(a[1], a[7]);
            a[1] = MinU8(a[1], a[9]);
            a[11] = MaxU8(a[5], a[11]);
            a[11] = MaxU8(a[3], a[11]);
            a[11] = MaxU8(a[2], a[11]);
            SortU8(a[0], a[6]); SortU8(a[1], a[8]); SortU8(a[6], a[8]);
            a[4] = MinU8(a[4], a[8]);
            SortU8(a[0], a[1]); SortU8(a[4], a[6]); SortU8(a[0], a[4]);
            a[11] = MaxU8(a[0], a[11]);
            SortU8(a[6], a[11]);
            a[1] = MinU8(a[1], a[11]);
            SortU8(a[1], a[4]); SortU8(a[6], a[12]);
            a[6] = MaxU8(a[1], a[6]);
            a[4] = MinU8(a[4], a[12]);
            a[6] = MaxU8(a[4], a[6]);
        }

        void MedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            int a[13];
            const uint8_t * y[5];
            size_t x[5];

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 2);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                y[3] = y[2] + srcStride;
                y[4] = y[3] + srcStride;
                if (row < 2)
                {
                    if (row < 1)
                        y[1] = y[2];
                    y[0] = y[1];
                }
                if (row >= height - 2)
                {
                    if (row >= height - 1)
                        y[3] = y[2];
                    y[4] = y[3];
                }

                for (size_t col = 0; col < 4 * channelCount; col++)
                {
                    if (col < 2 * channelCount)
                    {
                        x[0] = col < channelCount ? col : col - channelCount;
                        x[1] = x[0];
                        x[2] = col;
                        x[3] = x[2] + channelCount;
                        x[4] = x[3] + channelCount;
                    }
                    else
                    {
                        x[0] = size - 6 * channelCount + col;
                        x[1] = x[0] + channelCount;
                        x[2] = x[1] + channelCount;
                        x[3] = col < 3 * channelCount ? x[2] + channelCount : x[2];
                        x[4] = x[3];
                    }

                    LoadRhomb5x5(y, x, a);
                    PartialSort13(a);
                    dst[x[2]] = (uint8_t)a[6];
                }

                for (size_t col = 2 * channelCount; col < size - 2 * channelCount; ++col)
                {
                    x[0] = col - 2 * channelCount;
                    x[1] = col - channelCount;
                    x[2] = col;
                    x[3] = col + channelCount;
                    x[4] = col + 2 * channelCount;

                    LoadRhomb5x5(y, x, a);
                    PartialSort13(a);
                    dst[col] = (uint8_t)a[6];
                }
            }
        }

        SIMD_INLINE void LoadSquare5x5(const uint8_t * y[5], size_t x[5], int a[25])
        {
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]];	a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]];	a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]];	a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]];	a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]];	a[24] = y[4][x[4]];
        }

        SIMD_INLINE void PartialSort25(int a[25])
        {
            SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[2], a[4]);
            SortU8(a[2], a[3]); SortU8(a[6], a[7]); SortU8(a[5], a[7]);
            SortU8(a[5], a[6]); SortU8(a[9], a[10]); SortU8(a[8], a[10]);
            SortU8(a[8], a[9]); SortU8(a[12], a[13]); SortU8(a[11], a[13]);
            SortU8(a[11], a[12]); SortU8(a[15], a[16]); SortU8(a[14], a[16]);
            SortU8(a[14], a[15]); SortU8(a[18], a[19]); SortU8(a[17], a[19]);
            SortU8(a[17], a[18]); SortU8(a[21], a[22]); SortU8(a[20], a[22]);
            SortU8(a[20], a[21]); SortU8(a[23], a[24]); SortU8(a[2], a[5]);
            SortU8(a[3], a[6]); SortU8(a[0], a[6]); SortU8(a[0], a[3]);
            SortU8(a[4], a[7]); SortU8(a[1], a[7]); SortU8(a[1], a[4]);
            SortU8(a[11], a[14]); SortU8(a[8], a[14]); SortU8(a[8], a[11]);
            SortU8(a[12], a[15]); SortU8(a[9], a[15]); SortU8(a[9], a[12]);
            SortU8(a[13], a[16]); SortU8(a[10], a[16]); SortU8(a[10], a[13]);
            SortU8(a[20], a[23]); SortU8(a[17], a[23]); SortU8(a[17], a[20]);
            SortU8(a[21], a[24]); SortU8(a[18], a[24]); SortU8(a[18], a[21]);
            SortU8(a[19], a[22]); SortU8(a[9], a[18]); SortU8(a[0], a[18]);
            a[17] = MaxU8(a[8], a[17]);
            a[9] = MaxU8(a[0], a[9]);
            SortU8(a[10], a[19]); SortU8(a[1], a[19]); SortU8(a[1], a[10]);
            SortU8(a[11], a[20]); SortU8(a[2], a[20]); SortU8(a[12], a[21]);
            a[11] = MaxU8(a[2], a[11]);
            SortU8(a[3], a[21]); SortU8(a[3], a[12]); SortU8(a[13], a[22]);
            a[4] = MinU8(a[4], a[22]);
            SortU8(a[4], a[13]); SortU8(a[14], a[23]);
            SortU8(a[5], a[23]); SortU8(a[5], a[14]); SortU8(a[15], a[24]);
            a[6] = MinU8(a[6], a[24]);
            SortU8(a[6], a[15]);
            a[7] = MinU8(a[7], a[16]);
            a[7] = MinU8(a[7], a[19]);
            a[13] = MinU8(a[13], a[21]);
            a[15] = MinU8(a[15], a[23]);
            a[7] = MinU8(a[7], a[13]);
            a[7] = MinU8(a[7], a[15]);
            a[9] = MaxU8(a[1], a[9]);
            a[11] = MaxU8(a[3], a[11]);
            a[17] = MaxU8(a[5], a[17]);
            a[17] = MaxU8(a[11], a[17]);
            a[17] = MaxU8(a[9], a[17]);
            SortU8(a[4], a[10]);
            SortU8(a[6], a[12]); SortU8(a[7], a[14]); SortU8(a[4], a[6]);
            a[7] = MaxU8(a[4], a[7]);
            SortU8(a[12], a[14]);
            a[10] = MinU8(a[10], a[14]);
            SortU8(a[6], a[7]); SortU8(a[10], a[12]); SortU8(a[6], a[10]);
            a[17] = MaxU8(a[6], a[17]);
            SortU8(a[12], a[17]);
            a[7] = MinU8(a[7], a[17]);
            SortU8(a[7], a[10]); SortU8(a[12], a[18]);
            a[12] = MaxU8(a[7], a[12]);
            a[10] = MinU8(a[10], a[18]);
            SortU8(a[12], a[20]);
            a[10] = MinU8(a[10], a[20]);
            a[12] = MaxU8(a[10], a[12]);
        }

        void MedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            int a[25];
            const uint8_t * y[5];
            size_t x[5];

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 2);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                y[3] = y[2] + srcStride;
                y[4] = y[3] + srcStride;
                if (row < 2)
                {
                    if (row < 1)
                        y[1] = y[2];
                    y[0] = y[1];
                }
                if (row >= height - 2)
                {
                    if (row >= height - 1)
                        y[3] = y[2];
                    y[4] = y[3];
                }

                for (size_t col = 0; col < 4 * channelCount; col++)
                {
                    if (col < 2 * channelCount)
                    {
                        x[0] = col < channelCount ? col : col - channelCount;
                        x[1] = x[0];
                        x[2] = col;
                        x[3] = x[2] + channelCount;
                        x[4] = x[3] + channelCount;
                    }
                    else
                    {
                        x[0] = size - 6 * channelCount + col;
                        x[1] = x[0] + channelCount;
                        x[2] = x[1] + channelCount;
                        x[3] = col < 3 * channelCount ? x[2] + channelCount : x[2];
                        x[4] = x[3];
                    }

                    LoadSquare5x5(y, x, a);
                    PartialSort25(a);
                    dst[x[2]] = (uint8_t)a[12];
                }

                for (size_t col = 2 * channelCount; col < size - 2 * channelCount; ++col)
                {
                    x[0] = col - 2 * channelCount;
                    x[1] = col - channelCount;
                    x[2] = col;
                    x[3] = col + channelCount;
                    x[4] = col + 2 * channelCount;

                    LoadSquare5x5(y, x, a);
                    PartialSort25(a);
                    dst[col] = (uint8_t)a[12];
                }
            }
        }
    }
}
