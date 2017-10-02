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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        void StretchGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth * 2 == dstWidth && srcHeight * 2 == dstHeight);

            for (size_t row = 0; row < srcHeight; ++row)
            {
                uint8_t * dstEven = dst;
                uint8_t * dstOdd = dst + dstStride;
                for (size_t srcCol = 0; srcCol < srcWidth; srcCol += 1, dstEven += 2, dstOdd += 2)
                {
                    uint8_t value = src[srcCol];
                    dstEven[0] = value;
                    dstEven[1] = value;
                    dstOdd[0] = value;
                    dstOdd[1] = value;
                }
                src += srcStride;
                dst += 2 * dstStride;
            }
        }
    }
}
