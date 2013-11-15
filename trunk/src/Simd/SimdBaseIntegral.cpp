/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
	namespace Base
	{
        void IntegralSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint32_t * sum, size_t sumStride)
        {
            memset(sum, 0, sumStride*sizeof(uint32_t));
            sum += sumStride + 1;

            for(size_t row = 0; row < height; row++)
            {
                uint32_t rowSum = 0;
                sum[-1] = 0;
                for(size_t col = 0; col < width; col++)
                {
                    rowSum += src[col];
                    sum[col] = rowSum + sum[col - sumStride];
                }
                src += srcStride;
                sum += sumStride;
            }
        }

        void IntegralSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * sum, size_t sumStride)
        {
            assert(sumStride%sizeof(uint32_t) == 0);

            IntegralSum(src, srcStride, width, height, (uint32_t*)sum, sumStride/sizeof(uint32_t));
        }
	}
}
