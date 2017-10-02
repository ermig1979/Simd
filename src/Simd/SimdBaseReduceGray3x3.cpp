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
        template <bool compensation> void ReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            for (size_t col = 0; col < srcHeight; col += 2, dst += dstStride)
            {
                const uint8_t *src0 = src + srcStride*(col - 1);
                const uint8_t *src1 = src0 + srcStride;
                const uint8_t *src2 = src1 + srcStride;
                if (col == 0)
                    src0 = src1;
                if (col == srcHeight - 1)
                    src2 = src1;

                uint8_t *pDst = dst;
                size_t row;

                *pDst++ = GaussianBlur3x3<compensation>(src0, src1, src2, 0, 0, 1);

                for (row = 2; row < srcWidth - 1; row += 2)
                    *pDst++ = GaussianBlur3x3<compensation>(src0, src1, src2, row - 1, row, row + 1);

                if (row == srcWidth - 1)
                    *pDst++ = GaussianBlur3x3<compensation>(src0, src1, src2, srcWidth - 2, srcWidth - 1, srcWidth - 1);
            }
        }

        void ReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray3x3<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray3x3<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
}
