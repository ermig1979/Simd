/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
		SIMD_INLINE int AbsSecondDerivative(const uint8_t * src, ptrdiff_t step)
		{
			return AbsDifferenceU8(Average(src[step], src[-step]), src[0]);
		}

		void AbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
			size_t step, size_t indent, uint32_t * histogram)
		{
			assert(width > 2*indent && height > 2*indent && indent >= step);

			memset(histogram, 0, sizeof(uint32_t)*HISTOGRAM_SIZE);

			src += indent*(stride + 1);
			height -= 2*indent;
			width -= 2*indent;

			size_t rowStep = step*stride;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					const int sdX = AbsSecondDerivative(src + col, step);
					const int sdY = AbsSecondDerivative(src + col, rowStep);
					const int sd = MaxU8(sdY, sdX);
					++histogram[sd];
				}
				src += stride;
			}
		}

        void Histogram(const uint8_t * src, size_t width, size_t height, size_t stride, uint32_t * histogram)
        {
            memset(histogram, 0, sizeof(uint32_t)*HISTOGRAM_SIZE);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                    ++histogram[src[col]];
                src += stride;
            }
        }
	}
}
