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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(int) * 2 * width);
                    src0 = (int*)_p;
                    src1 = src0 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                int * src0;
                int * src1;
            private:
                void *_p;
            };
        }

        SIMD_INLINE int DivideBy64(int value)
        {
            return (value + 32) >> 6;
        }

        SIMD_INLINE int GaussianBlur(const uint8_t *src, size_t x0, size_t x1, size_t x2, size_t x3)
        {
            return src[x0] + 3 * (src[x1] + src[x2]) + src[x3];
        }

        SIMD_INLINE void ProcessFirstRow(const uint8_t *src, size_t x0, size_t x1, size_t x2, size_t x3, Buffer & buffer, size_t offset)
        {
            int tmp = GaussianBlur(src, x0, x1, x2, x3);
            buffer.src0[offset] = tmp;
            buffer.src1[offset] = tmp;
        }

        SIMD_INLINE void ProcessMainRow(const uint8_t *s2, const uint8_t *s3, size_t x0, size_t x1, size_t x2, size_t x3, Buffer & buffer, uint8_t* dst, size_t offset)
        {
            int tmp2 = GaussianBlur(s2, x0, x1, x2, x3);
            int tmp3 = GaussianBlur(s3, x0, x1, x2, x3);
            dst[offset] = DivideBy64(buffer.src0[offset] + 3 * (buffer.src1[offset] + tmp2) + tmp3);
            buffer.src0[offset] = tmp2;
            buffer.src1[offset] = tmp3;
        }

        void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > 2);

            Buffer buffer(dstWidth);

            ProcessFirstRow(src, 0, 0, 1, 2, buffer, 0);
            size_t srcCol = 2, dstCol = 1;
            for (; srcCol < srcWidth - 2; srcCol += 2, dstCol++)
                ProcessFirstRow(src, srcCol - 1, srcCol, srcCol + 1, srcCol + 2, buffer, dstCol);
            ProcessFirstRow(src, srcCol - 1, srcCol, srcWidth - 1, srcWidth - 1, buffer, dstCol);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                const uint8_t *src3 = src2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    src2 = src + srcStride*(srcHeight - 1);
                    src3 = src2;
                }

                ProcessMainRow(src2, src3, 0, 0, 1, 2, buffer, dst, 0);
                size_t srcCol = 2, dstCol = 1;
                for (; srcCol < srcWidth - 2; srcCol += 2, dstCol++)
                    ProcessMainRow(src2, src3, srcCol - 1, srcCol, srcCol + 1, srcCol + 2, buffer, dst, dstCol);
                ProcessMainRow(src2, src3, srcCol - 1, srcCol, srcWidth - 1, srcWidth - 1, buffer, dst, dstCol);
            }
        }
    }
}
