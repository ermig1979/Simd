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
        SIMD_INLINE int LbpEstimate(const uint8_t * src, ptrdiff_t stride)
        {
            int threshold = src[0];
            int lbp = 0;
            lbp |= (src[-stride - 1] >= threshold ? 0x01 : 0);
            lbp |= (src[-stride] >= threshold ? 0x02 : 0);
            lbp |= (src[-stride + 1] >= threshold ? 0x04 : 0);
            lbp |= (src[1] >= threshold ? 0x08 : 0);
            lbp |= (src[stride + 1] >= threshold ? 0x10 : 0);
            lbp |= (src[stride] >= threshold ? 0x20 : 0);
            lbp |= (src[stride - 1] >= threshold ? 0x40 : 0);
            lbp |= (src[-1] >= threshold ? 0x80 : 0);
            return lbp;
        }

        void LbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                dst[0] = 0;
                for (size_t col = 1; col < width - 1; ++col)
                {
                    dst[col] = LbpEstimate(src + col, srcStride);
                }
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }
    }
}
