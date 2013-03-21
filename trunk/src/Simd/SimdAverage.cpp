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
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAverage.h"

namespace Simd
{
	namespace Base
	{
		void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			size_t size = width*channelCount;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t offset = 0; offset < size; ++offset)
					dst[offset] = Average(a[offset], b[offset]);
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			assert(width*channelCount >= A);
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

			size_t size = channelCount*width;
			size_t alignedSize = Simd::AlignLo(size, A);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t offset = 0; offset < alignedSize; offset += A)
				{
					const __m128i a_ = Load<align>((__m128i*)(a + offset));
					const __m128i b_ = Load<align>((__m128i*)(b + offset));
					Store<align>((__m128i*)(dst + offset), _mm_avg_epu8(a_, b_));
				}
				if(alignedSize != size)
				{
					const __m128i a_ = Load<false>((__m128i*)(a + size - A));
					const __m128i b_ = Load<false>((__m128i*)(b + size - A));
					Store<false>((__m128i*)(dst + size - A), _mm_avg_epu8(a_, b_));
				}
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
				Average<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			else
				Average<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template <bool align> void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			assert(width*channelCount >= A);
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

			size_t size = channelCount*width;
			size_t alignedSize = Simd::AlignLo(size, A);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t offset = 0; offset < alignedSize; offset += A)
				{
					const __m256i a_ = Load<align>((__m256i*)(a + offset));
					const __m256i b_ = Load<align>((__m256i*)(b + offset));
					Store<align>((__m256i*)(dst + offset), _mm256_avg_epu8(a_, b_));
				}
				if(alignedSize != size)
				{
					const __m256i a_ = Load<false>((__m256i*)(a + size - A));
					const __m256i b_ = Load<false>((__m256i*)(b + size - A));
					Store<false>((__m256i*)(dst + size - A), _mm256_avg_epu8(a_, b_));
				}
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
				Average<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			else
				Average<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
		}
	}
#endif// SIMD_AVX2_ENABLE

	void Average(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
		size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
	{
#ifdef SIMD_AVX2_ENABLE
		if(Avx2::Enable && width*channelCount >= Avx2::A)
			Avx2::Average(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
		else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width*channelCount >= Sse2::A)
			Sse2::Average(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::Average(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
	}

	void Average(const View & a, const View & b, View & dst)
	{
		assert(a.width == b.width && a.height == b.height && a.format == b.format);
		assert(a.width == dst.width && a.height == dst.height && a.format == dst.format);
		assert(a.format == View::Gray8 || a.format == View::Uv16 || a.format == View::Bgr24 || a.format == View::Bgra32);

		Average(a.data, a.stride, b.data, b.stride, a.width, a.height, View::SizeOf(a.format), dst.data, dst.stride);
	}
}