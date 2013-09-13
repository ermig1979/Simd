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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdMedianFilterSquare5x5.h"

namespace Simd
{
	namespace Base
	{
		SIMD_INLINE void LoadSquare5x5(const uchar * y[5], size_t x[5], int a[25])
		{
			a[0]  = y[0][x[0]]; a[1]  = y[0][x[1]]; a[2]  = y[0][x[2]]; a[3]  = y[0][x[3]];	a[4]  = y[0][x[4]]; 
			a[5]  = y[1][x[0]]; a[6]  = y[1][x[1]]; a[7]  = y[1][x[2]]; a[8]  = y[1][x[3]];	a[9]  = y[1][x[4]]; 
			a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]];	a[14] = y[2][x[4]]; 
			a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]];	a[19] = y[3][x[4]]; 
			a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]];	a[24] = y[4][x[4]];
		}

		SIMD_INLINE void Sort25(int a[25])
		{
			SortU8(a[0] , a[1] ); SortU8(a[3] , a[4] ); SortU8(a[2] , a[4] );
			SortU8(a[2] , a[3] ); SortU8(a[6] , a[7] ); SortU8(a[5] , a[7] );
			SortU8(a[5] , a[6] ); SortU8(a[9] , a[10]); SortU8(a[8] , a[10]);
			SortU8(a[8] , a[9] ); SortU8(a[12], a[13]); SortU8(a[11], a[13]);
			SortU8(a[11], a[12]); SortU8(a[15], a[16]); SortU8(a[14], a[16]);
			SortU8(a[14], a[15]); SortU8(a[18], a[19]); SortU8(a[17], a[19]);
			SortU8(a[17], a[18]); SortU8(a[21], a[22]); SortU8(a[20], a[22]);
			SortU8(a[20], a[21]); SortU8(a[23], a[24]); SortU8(a[2] , a[5] );
			SortU8(a[3] , a[6] ); SortU8(a[0] , a[6] ); SortU8(a[0] , a[3] );
			SortU8(a[4] , a[7] ); SortU8(a[1] , a[7] ); SortU8(a[1] , a[4] );
			SortU8(a[11], a[14]); SortU8(a[8] , a[14]); SortU8(a[8] , a[11]);
			SortU8(a[12], a[15]); SortU8(a[9] , a[15]); SortU8(a[9] , a[12]);
			SortU8(a[13], a[16]); SortU8(a[10], a[16]); SortU8(a[10], a[13]);
			SortU8(a[20], a[23]); SortU8(a[17], a[23]); SortU8(a[17], a[20]);
			SortU8(a[21], a[24]); SortU8(a[18], a[24]); SortU8(a[18], a[21]);
			SortU8(a[19], a[22]); SortU8(a[8] , a[17]); SortU8(a[9] , a[18]);
			SortU8(a[0] , a[18]); SortU8(a[0] , a[9] ); SortU8(a[10], a[19]);
			SortU8(a[1] , a[19]); SortU8(a[1] , a[10]); SortU8(a[11], a[20]);
			SortU8(a[2] , a[20]); SortU8(a[2] , a[11]); SortU8(a[12], a[21]);
			SortU8(a[3] , a[21]); SortU8(a[3] , a[12]); SortU8(a[13], a[22]);
			SortU8(a[4] , a[22]); SortU8(a[4] , a[13]); SortU8(a[14], a[23]);
			SortU8(a[5] , a[23]); SortU8(a[5] , a[14]); SortU8(a[15], a[24]);
			SortU8(a[6] , a[24]); SortU8(a[6] , a[15]); SortU8(a[7] , a[16]);
			SortU8(a[7] , a[19]); SortU8(a[13], a[21]); SortU8(a[15], a[23]);
			SortU8(a[7] , a[13]); SortU8(a[7] , a[15]); SortU8(a[1] , a[9] );
			SortU8(a[3] , a[11]); SortU8(a[5] , a[17]); SortU8(a[11], a[17]);
			SortU8(a[9] , a[17]); SortU8(a[4] , a[10]); SortU8(a[6] , a[12]);
			SortU8(a[7] , a[14]); SortU8(a[4] , a[6] ); SortU8(a[4] , a[7] );
			SortU8(a[12], a[14]); SortU8(a[10], a[14]); SortU8(a[6] , a[7] );
			SortU8(a[10], a[12]); SortU8(a[6] , a[10]); SortU8(a[6] , a[17]);
			SortU8(a[12], a[17]); SortU8(a[7] , a[17]); SortU8(a[7] , a[10]);
			SortU8(a[12], a[18]); SortU8(a[7] , a[12]); SortU8(a[10], a[18]);
			SortU8(a[12], a[20]); SortU8(a[10], a[20]); SortU8(a[10], a[12]);
		}

		void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			int a[25];
			const uchar * y[5];
			size_t x[5];

			size_t size = channelCount*width;
			for(size_t row = 0; row < height; ++row, dst += dstStride)
			{
				y[0] = src + srcStride*(row - 2);
				y[1] = y[0] + srcStride;
				y[2] = y[1] + srcStride;
				y[3] = y[2] + srcStride;
				y[4] = y[3] + srcStride;
				if(row < 2)
				{
					if(row < 1)
						y[1] = y[2];
					y[0] = y[1];
				}
				if(row >= height - 2)
				{
					if(row >= height - 1)
						y[3] = y[2];
					y[4] = y[3];
				}

				for(size_t col = 0; col < 4*channelCount; col++)
				{
					if(col < 2*channelCount)
					{
						x[0] = col < channelCount ? col : col - channelCount; 
						x[1] = x[0]; 
						x[2] = col; 
						x[3] = x[2] + channelCount; 
						x[4] = x[3] + channelCount;
					}
					else
					{
						x[0] = size - 6*channelCount + col;
						x[1] = x[0] + channelCount;
						x[2] = x[1] + channelCount;
						x[3] = col < 3*channelCount ? x[2] + channelCount : x[2];
						x[4] = x[3];
					}

					LoadSquare5x5(y, x, a);
					Sort25(a);
					dst[x[2]] = (uchar)a[12];
				}

				for(size_t col = 2*channelCount; col < size - 2*channelCount; ++col)
				{
					x[0] = col - 2*channelCount;
					x[1] = col - channelCount;
					x[2] = col;
					x[3] = col + channelCount;
					x[4] = col + 2*channelCount;

					LoadSquare5x5(y, x, a);
					Sort25(a);
					dst[col] = (uchar)a[12];
				}
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare5x5(const uchar* y[5], size_t offset, __m128i a[25])
		{
			LoadNose5<align, step>(y[0] + offset, a + 0 );
			LoadNose5<align, step>(y[1] + offset, a + 5 );
			LoadNose5<align, step>(y[2] + offset, a + 10);
			LoadNose5<align, step>(y[3] + offset, a + 15);
			LoadNose5<align, step>(y[4] + offset, a + 20);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBodySquare5x5(const uchar* y[5], size_t offset, __m128i a[25])
		{
			LoadBody5<align, step>(y[0] + offset, a + 0 );
			LoadBody5<align, step>(y[1] + offset, a + 5 );
			LoadBody5<align, step>(y[2] + offset, a + 10);
			LoadBody5<align, step>(y[3] + offset, a + 15);
			LoadBody5<align, step>(y[4] + offset, a + 20);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTailSquare5x5(const uchar* y[5], size_t offset, __m128i a[25])
		{
			LoadTail5<align, step>(y[0] + offset, a + 0 );
			LoadTail5<align, step>(y[1] + offset, a + 5 );
			LoadTail5<align, step>(y[2] + offset, a + 10);
			LoadTail5<align, step>(y[3] + offset, a + 15);
			LoadTail5<align, step>(y[4] + offset, a + 20);
		}

		SIMD_INLINE void Sort25(__m128i a[25])
		{
			SortU8(a[0] , a[1] ); SortU8(a[3] , a[4] ); SortU8(a[2] , a[4] );
			SortU8(a[2] , a[3] ); SortU8(a[6] , a[7] ); SortU8(a[5] , a[7] );
			SortU8(a[5] , a[6] ); SortU8(a[9] , a[10]); SortU8(a[8] , a[10]);
			SortU8(a[8] , a[9] ); SortU8(a[12], a[13]); SortU8(a[11], a[13]);
			SortU8(a[11], a[12]); SortU8(a[15], a[16]); SortU8(a[14], a[16]);
			SortU8(a[14], a[15]); SortU8(a[18], a[19]); SortU8(a[17], a[19]);
			SortU8(a[17], a[18]); SortU8(a[21], a[22]); SortU8(a[20], a[22]);
			SortU8(a[20], a[21]); SortU8(a[23], a[24]); SortU8(a[2] , a[5] );
			SortU8(a[3] , a[6] ); SortU8(a[0] , a[6] ); SortU8(a[0] , a[3] );
			SortU8(a[4] , a[7] ); SortU8(a[1] , a[7] ); SortU8(a[1] , a[4] );
			SortU8(a[11], a[14]); SortU8(a[8] , a[14]); SortU8(a[8] , a[11]);
			SortU8(a[12], a[15]); SortU8(a[9] , a[15]); SortU8(a[9] , a[12]);
			SortU8(a[13], a[16]); SortU8(a[10], a[16]); SortU8(a[10], a[13]);
			SortU8(a[20], a[23]); SortU8(a[17], a[23]); SortU8(a[17], a[20]);
			SortU8(a[21], a[24]); SortU8(a[18], a[24]); SortU8(a[18], a[21]);
			SortU8(a[19], a[22]); SortU8(a[8] , a[17]); SortU8(a[9] , a[18]);
			SortU8(a[0] , a[18]); SortU8(a[0] , a[9] ); SortU8(a[10], a[19]);
			SortU8(a[1] , a[19]); SortU8(a[1] , a[10]); SortU8(a[11], a[20]);
			SortU8(a[2] , a[20]); SortU8(a[2] , a[11]); SortU8(a[12], a[21]);
			SortU8(a[3] , a[21]); SortU8(a[3] , a[12]); SortU8(a[13], a[22]);
			SortU8(a[4] , a[22]); SortU8(a[4] , a[13]); SortU8(a[14], a[23]);
			SortU8(a[5] , a[23]); SortU8(a[5] , a[14]); SortU8(a[15], a[24]);
			SortU8(a[6] , a[24]); SortU8(a[6] , a[15]); SortU8(a[7] , a[16]);
			SortU8(a[7] , a[19]); SortU8(a[13], a[21]); SortU8(a[15], a[23]);
			SortU8(a[7] , a[13]); SortU8(a[7] , a[15]); SortU8(a[1] , a[9] );
			SortU8(a[3] , a[11]); SortU8(a[5] , a[17]); SortU8(a[11], a[17]);
			SortU8(a[9] , a[17]); SortU8(a[4] , a[10]); SortU8(a[6] , a[12]);
			SortU8(a[7] , a[14]); SortU8(a[4] , a[6] ); SortU8(a[4] , a[7] );
			SortU8(a[12], a[14]); SortU8(a[10], a[14]); SortU8(a[6] , a[7] );
			SortU8(a[10], a[12]); SortU8(a[6] , a[10]); SortU8(a[6] , a[17]);
			SortU8(a[12], a[17]); SortU8(a[7] , a[17]); SortU8(a[7] , a[10]);
			SortU8(a[12], a[18]); SortU8(a[7] , a[12]); SortU8(a[10], a[18]);
			SortU8(a[12], a[20]); SortU8(a[10], a[20]); SortU8(a[10], a[12]);
		}

		template <bool align, size_t step> void MedianFilterSquare5x5(
			const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
		{
			assert(step*width >= A);

			const uchar * y[5];
			__m128i a[25];

			size_t size = step*width;
			size_t bodySize = Simd::AlignHi(size, A) - A;

			for(size_t row = 0; row < height; ++row, dst += dstStride)
			{
				y[0] = src + srcStride*(row - 2);
				y[1] = y[0] + srcStride;
				y[2] = y[1] + srcStride;
				y[3] = y[2] + srcStride;
				y[4] = y[3] + srcStride;
				if(row < 2)
				{
					if(row < 1)
						y[1] = y[2];
					y[0] = y[1];
				}
				if(row >= height - 2)
				{
					if(row >= height - 1)
						y[3] = y[2];
					y[4] = y[3];
				}

				LoadNoseSquare5x5<align, step>(y, 0, a);
				Sort25(a);
				Store<align>((__m128i*)(dst), a[12]);

				for(size_t col = A; col < bodySize; col += A)
				{
					LoadBodySquare5x5<align, step>(y, col, a);
					Sort25(a);
					Store<align>((__m128i*)(dst + col), a[12]);
				}

				size_t col = size - A;
				LoadTailSquare5x5<false, step>(y, col, a);
				Sort25(a);
				Store<false>((__m128i*)(dst + col), a[12]);
			}
		}

		template <bool align> void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			assert(channelCount > 0 && channelCount <= 4);

			switch(channelCount)
			{
			case 1: MedianFilterSquare5x5<align, 1>(src, srcStride, width, height, dst, dstStride); break;
			case 2: MedianFilterSquare5x5<align, 2>(src, srcStride, width, height, dst, dstStride); break;
			case 3: MedianFilterSquare5x5<align, 3>(src, srcStride, width, height, dst, dstStride); break;
			case 4: MedianFilterSquare5x5<align, 4>(src, srcStride, width, height, dst, dstStride); break;
			}
		}

		void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			if(Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
				MedianFilterSquare5x5<true>(src, srcStride, width, height, channelCount, dst, dstStride);
			else
				MedianFilterSquare5x5<false>(src, srcStride, width, height, channelCount, dst, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, 
		size_t channelCount, uchar * dst, size_t dstStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width*channelCount >= Avx2::A)
            Avx2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width*channelCount >= Sse2::A)
			Sse2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
		else
#endif//SIMD_SSE2_ENABLE
			Base::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
	}
}