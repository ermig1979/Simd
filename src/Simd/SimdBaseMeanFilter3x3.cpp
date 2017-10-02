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
#include "Simd/SimdConst.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int DivideBy9(int value)
        {
            return ((value + 5)*DIVISION_BY_9_FACTOR) >> DIVISION_BY_9_SHIFT;
        }

        SIMD_INLINE int MeanFilter3x3(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return DivideBy9(s0[x0] + s0[x1] + s0[x2] + s1[x0] + s1[x1] + s1[x2] + s2[x0] + s2[x1] + s2[x2]);
        }

        void MeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            const uint8_t *src0, *src1, *src2;

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                size_t col = 0;
                for (; col < channelCount; col++)
                    dst[col] = MeanFilter3x3(src0, src1, src2, col, col, col + channelCount);

                for (; col < size - channelCount; ++col)
                    dst[col] = MeanFilter3x3(src0, src1, src2, col - channelCount, col, col + channelCount);

                for (; col < size; col++)
                    dst[col] = MeanFilter3x3(src0, src1, src2, col - channelCount, col, col);

                dst += dstStride;
            }
        }
    }
}
