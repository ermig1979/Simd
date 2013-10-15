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
		SIMD_INLINE void LoadSquare3x3(const uchar * y[3], size_t x[3], int a[9])
		{
			a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
			a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
			a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];
		}

		SIMD_INLINE void Sort9(int a[9])
		{
			SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]); 
			SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[6], a[7]);
			SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]); 
			SortU8(a[0], a[3]); SortU8(a[5], a[8]); SortU8(a[4], a[7]);
			SortU8(a[3], a[6]); SortU8(a[1], a[4]); SortU8(a[2], a[5]); 
			SortU8(a[4], a[7]); SortU8(a[4], a[2]); SortU8(a[6], a[4]);
			SortU8(a[4], a[2]);
		}

		void MedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			int a[9];
			const uchar * y[3];
			size_t x[3];

			size_t size = channelCount*width;
			for(size_t row = 0; row < height; ++row, dst += dstStride)
			{
				y[0] = src + srcStride*(row - 1);
				y[1] = y[0] + srcStride;
				y[2] = y[1] + srcStride;
				if(row < 1)
					y[0] = y[1];
				if(row >= height - 1)
					y[2] = y[1];

				for(size_t col = 0; col < 2*channelCount; col++)
				{
					x[0] = col < channelCount ? col : size - 3*channelCount + col;
					x[2] = col < channelCount ? col + channelCount : size - 2*channelCount + col;
					x[1] = col < channelCount ? x[0] : x[2];

					LoadSquare3x3(y, x, a);
					Sort9(a);
					dst[x[1]] = (uchar)a[4];
				}

				for(size_t col = channelCount; col < size - channelCount; ++col)
				{
					x[0] = col - channelCount;
					x[1] = col;
					x[2] = col + channelCount;

					LoadSquare3x3(y, x, a);
					Sort9(a);
					dst[col] = (uchar)a[4];
				}
			}
		}
	}
}