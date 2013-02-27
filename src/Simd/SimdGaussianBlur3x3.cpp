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
#include <memory.h>

#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdGaussianBlur3x3.h"

namespace Simd
{
	namespace Base
	{
		SIMD_INLINE int DivideBy16(int value)
		{
			return (value + 8) >> 4;
		}

		SIMD_INLINE int GaussianBlur(const uchar *s0, const uchar *s1, const uchar *s2, size_t x0, size_t x1, size_t x2)
		{
			return DivideBy16(s0[x0] + 2*s0[x1] + s0[x2] + (s1[x0] + 2*s1[x1] + s1[x2])*2 + s2[x0] + 2*s2[x1] + s2[x2]);
		}

		void GaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			const uchar *src0, *src1, *src2;

			size_t size = channelCount*width;
			for(size_t row = 0; row < height; ++row)
			{
				src0 = src + srcStride*(row - 1);
				src1 = src0 + srcStride;
				src2 = src1 + srcStride;
				if(row == 0)
					src0 = src1;
				if(row == height - 1)
					src2 = src1;

				size_t col = 0;
				for(;col < channelCount; col++)
					dst[col] = GaussianBlur(src0, src1, src2, col, col, col + channelCount);

				for(; col < size - channelCount; ++col)
					dst[col] = GaussianBlur(src0, src1, src2, col - channelCount, col, col + channelCount);

				for(; col < size; col++)
					dst[col] = GaussianBlur(src0, src1, src2, col - channelCount, col, col);

				dst += dstStride;
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		namespace
		{
			struct Buffer
			{
				Buffer(size_t width)
				{
					_p = Allocate(sizeof(ushort)*3*width);
					src0 = (ushort*)_p;
					src1 = src0 + width;
					src2 = src1 + width;
				}

				~Buffer()
				{
					Free(_p);
				}

				ushort * src0;
				ushort * src1;
				ushort * src2;
			private:
				void * _p;
			};	
		}

		SIMD_INLINE __m128i DivideBy16(__m128i value)
		{
			return _mm_srli_epi16(_mm_add_epi16(value, K16_0008), 4);
		}

		SIMD_INLINE __m128i BinomialSum16(const __m128i & a, const __m128i & b, const __m128i & c)
		{
			return _mm_add_epi16(_mm_add_epi16(a, c), _mm_add_epi16(b, b));
		}

		template<bool align> SIMD_INLINE void BlurCol(__m128i a[3], ushort * b)
		{
			Store<align>((__m128i*)(b + 0), BinomialSum16(_mm_unpacklo_epi8(a[0], K_ZERO), 
				_mm_unpacklo_epi8(a[1], K_ZERO), _mm_unpacklo_epi8(a[2], K_ZERO)));
			Store<align>((__m128i*)(b + HA), BinomialSum16(_mm_unpackhi_epi8(a[0], K_ZERO), 
				_mm_unpackhi_epi8(a[1], K_ZERO), _mm_unpackhi_epi8(a[2], K_ZERO)));
		}

		template<bool align> SIMD_INLINE __m128i BlurRow16(const Buffer & buffer, size_t offset)
		{
			return DivideBy16(BinomialSum16(
				Load<align>((__m128i*)(buffer.src0 + offset)), 
				Load<align>((__m128i*)(buffer.src1 + offset)),
				Load<align>((__m128i*)(buffer.src2 + offset))));
		}

		template<bool align> SIMD_INLINE __m128i BlurRow(const Buffer & buffer, size_t offset)
		{
			return _mm_packus_epi16(BlurRow16<align>(buffer, offset), BlurRow16<align>(buffer, offset + HA));
		}

		template <bool align, size_t step> void GaussianBlur3x3(
			const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
		{
			assert(step*width >= A);
			if(align)
				assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

			__m128i a[3];

			size_t size = step*width;
			size_t bodySize = Simd::AlignHi(size, A) - A;

			Buffer buffer(Simd::AlignHi(size, A));

			LoadNose3<align, step>(src + 0, a);
			BlurCol<true>(a, buffer.src0 + 0);
			for(size_t col = A; col < bodySize; col += A)
			{
				LoadBody3<align, step>(src + col, a);
				BlurCol<true>(a, buffer.src0 + col);
			}
			LoadTail3<align, step>(src + size - A, a);
			BlurCol<align>(a, buffer.src0 + size - A);

			memcpy(buffer.src1, buffer.src0, sizeof(ushort)*size);

			for(size_t row = 0; row < height; ++row, dst += dstStride)
			{
				const uchar *src2 = src + srcStride*(row + 1);
				if(row >= height - 2)
					src2 = src + srcStride*(height - 1);

				LoadNose3<align, step>(src2 + 0, a);
				BlurCol<true>(a, buffer.src2 + 0);
				for(size_t col = A; col < bodySize; col += A)
				{
					LoadBody3<align, step>(src2 + col, a);
					BlurCol<true>(a, buffer.src2 + col);
				}
				LoadTail3<align, step>(src2 + size - A, a);
				BlurCol<align>(a, buffer.src2 + size - A);

				for(size_t col = 0; col < bodySize; col += A)
					Store<align>((__m128i*)(dst + col), BlurRow<true>(buffer, col));
				Store<align>((__m128i*)(dst + size - A), BlurRow<align>(buffer, size - A));

				Swap(buffer.src0, buffer.src2);
				Swap(buffer.src0, buffer.src1);
			}
		}

		template <bool align> void GaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			assert(channelCount > 0 && channelCount <= 4);

			switch(channelCount)
			{
			case 1: GaussianBlur3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
			case 2: GaussianBlur3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
			case 3: GaussianBlur3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
			case 4: GaussianBlur3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
			}
		}

		void GaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
			size_t channelCount, uchar * dst, size_t dstStride)
		{
			if(Aligned(src) && Aligned(srcStride) && Aligned(channelCount*width) && Aligned(dst) && Aligned(dstStride))
				GaussianBlur3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
			else
				GaussianBlur3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void GaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
		size_t channelCount, uchar * dst, size_t dstStride)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width*channelCount >= Sse2::A)
			Sse2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
		else
#endif//SIMD_SSE2_ENABLE
			Base::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
	}

	void GaussianBlur3x3(const View & src, View & dst)
	{
		assert(src.width == dst.width && src.height == dst.height && src.format == dst.format);
		assert(src.format == View::Gray8 || src.format == View::Uv16 || src.format == View::Bgr24 || src.format == View::Bgra32);

		GaussianBlur3x3(src.data, src.stride, src.width, src.height, View::SizeOf(src.format), dst.data, dst.stride);
	}
}