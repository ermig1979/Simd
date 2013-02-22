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
#include "Simd/SimdInit.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAbsDifferenceSum.h"

namespace Simd
{
	namespace Base
	{
		void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			*sum = 0;
			for(size_t row = 0; row < height; ++row)
			{
				int rowSum = 0;
				for(size_t col = 0; col < width; ++col)
				{
					rowSum += AbsDifferenceU8(a[col], b[col]);
				}
				*sum += rowSum;
				a += aStride;
				b += bStride;
			}
		}

		void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			*sum = 0;
			for(size_t row = 0; row < height; ++row)
			{
				int rowSum = 0;
				for(size_t col = 0; col < width; ++col)
				{
					int m = mask[col];
					rowSum += AbsDifferenceU8(m & a[col], m & b[col]);
				}
				*sum += rowSum;
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> void AbsDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

			size_t bodyWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
			__m128i fullSum = _mm_setzero_si128();
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m128i a_ = Load<align>((__m128i*)(a + col));
					const __m128i b_ = Load<align>((__m128i*)(b + col));
					fullSum = _mm_add_epi64(_mm_sad_epu8(a_, b_), fullSum);
				}
				if(width - bodyWidth)
				{
					const __m128i a_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(a + width - A)));
					const __m128i b_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(b + width - A))); 
					fullSum = _mm_add_epi64(_mm_sad_epu8(a_, b_), fullSum);
				}
				a += aStride;
				b += bStride;
			}
			*sum = ExtractInt64Sum(fullSum);
		}

		template <bool align> void AbsDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			if(align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			size_t bodyWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
			__m128i fullSum = _mm_setzero_si128();
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m128i mask_ = Load<align>((__m128i*)(mask + col));
					const __m128i a_ = _mm_and_si128(mask_, Load<align>((__m128i*)(a + col)));
					const __m128i b_ = _mm_and_si128(mask_, Load<align>((__m128i*)(b + col))); 
					fullSum = _mm_add_epi64(_mm_sad_epu8(a_, b_), fullSum);
				}
				if(width - bodyWidth)
				{
					const __m128i mask_ = _mm_and_si128(tailMask, Load<align>((__m128i*)(mask + width - A)));
					const __m128i a_ = _mm_and_si128(mask_, Load<false>((__m128i*)(a + width - A)));
					const __m128i b_ = _mm_and_si128(mask_, Load<false>((__m128i*)(b + width - A))); 
					fullSum = _mm_add_epi64(_mm_sad_epu8(a_, b_), fullSum);
				}
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
			*sum = ExtractInt64Sum(fullSum);
		}

		void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
				AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
			else
				AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
		}

		void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
				AbsDifferenceSum<true>(a, aStride, b, bStride, mask, maskStride, width, height, sum);
			else
				AbsDifferenceSum<false>(a, aStride, b, bStride, mask, maskStride, width, height, sum);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
		size_t width, size_t height, uint64_t * sum)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
		else
#endif//SIMD_SSE2_ENABLE
			Base::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
	}

	void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
		const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::AbsDifferenceSum(a, aStride, b, bStride, mask, maskStride, width, height, sum);
		else
#endif//SIMD_SSE2_ENABLE
			Base::AbsDifferenceSum(a, aStride, b, bStride, mask, maskStride, width, height, sum);
	}

	void AbsDifferenceSum(const View & a, const View & b, uint64_t & sum)
	{
		assert(a.width == b.width && a.height == b.height);
		assert(a.format == View::Gray8 && b.format == View::Gray8);

		AbsDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
	}

	void AbsDifferenceSum(const View & a, const View & b, const View & mask, uint64_t & sum)
	{
		assert(a.width == b.width && a.height == b.height && a.width == mask.width && a.height == mask.height);
		assert(a.format == View::Gray8 && b.format == View::Gray8 && mask.format == View::Gray8);

		AbsDifferenceSum(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, a.width, a.height, &sum);
	}
}
