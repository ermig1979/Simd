/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
		void AnnConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
		{
            const float k = 1.0f / 255.0f;
			for (size_t row = 0; row < height; ++row)
			{
				if (inversion)
				{
					for (size_t col = 0; col < width; ++col)
						dst[col] = (255 - src[col])* k;
				}
				else
				{
					for (size_t col = 0; col < width; ++col)
						dst[col] = src[col] * k;
				}
				src += stride;
				dst += width;
			}
		}

        void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float sums[4] = {0, 0, 0, 0};
            size_t i = 0;
            for(; i < alignedSize; i += 4)
            {
                sums[0] += a[i + 0]*b[i + 0];
                sums[1] += a[i + 1]*b[i + 1];
                sums[2] += a[i + 2]*b[i + 2];
                sums[3] += a[i + 3]*b[i + 3];
            }
            for(; i < size; ++i)
                sums[0] += a[i]*b[i];
            *sum = sums[0] + sums[1] + sums[2] + sums[3];
        }

		void AnnSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = Sigmoid(src[i] * s);
		}

		void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = RoughSigmoid(src[i] * s);
		}
    }
}
