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
#include "Simd/SimdSquaredDifferenceSum.h"

namespace Simd
{
    namespace Base
    {
		void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);

			*sum = 0;
			for(size_t row = 0; row < height; ++row)
			{
				int rowSum = 0;
				for(size_t col = 0; col < width; ++col)
				{
					rowSum += SquaredDifference(a[col], b[col]);
				}
				*sum += rowSum;
				a += aStride;
				b += bStride;
			}
		}

		void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);

			*sum = 0;
			for(size_t row = 0; row < height; ++row)
			{
				int rowSum = 0;
				for(size_t col = 0; col < width; ++col)
				{
					int m = mask[col];
					rowSum += SquaredDifference(m & a[col], m & b[col]);
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
        SIMD_INLINE __m128i SquaredDifference(__m128i a, __m128i b)
        {
            const __m128i aLo = _mm_unpacklo_epi8(a, _mm_setzero_si128());
            const __m128i bLo = _mm_unpacklo_epi8(b, _mm_setzero_si128());
            const __m128i dLo = _mm_sub_epi16(aLo, bLo);

            const __m128i aHi = _mm_unpackhi_epi8(a, _mm_setzero_si128());
            const __m128i bHi = _mm_unpackhi_epi8(b, _mm_setzero_si128());
            const __m128i dHi = _mm_sub_epi16(aHi, bHi);

            return _mm_add_epi32(_mm_madd_epi16(dLo, dLo), _mm_madd_epi16(dHi, dHi));
        }

		template <bool align> void SquaredDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);
			if(align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
			}

			size_t bodyWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
			__m128i fullSum = _mm_setzero_si128();
			for(size_t row = 0; row < height; ++row)
			{
				__m128i rowSum = _mm_setzero_si128();
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m128i a_ = Load<align>((__m128i*)(a + col));
					const __m128i b_ = Load<align>((__m128i*)(b + col)); 
					rowSum = _mm_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				if(width - bodyWidth)
				{
					const __m128i a_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(a + width - A)));
					const __m128i b_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(b + width - A))); 
					rowSum = _mm_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				fullSum = _mm_add_epi64(fullSum, HorizontalSum32(rowSum));
				a += aStride;
				b += bStride;
			}
			*sum = ExtractInt64Sum(fullSum);
		}

		template <bool align> void SquaredDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);
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
				__m128i rowSum = _mm_setzero_si128();
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m128i mask_ = Load<align>((__m128i*)(mask + col));
					const __m128i a_ = _mm_and_si128(mask_, Load<align>((__m128i*)(a + col)));
					const __m128i b_ = _mm_and_si128(mask_, Load<align>((__m128i*)(b + col))); 
					rowSum = _mm_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				if(width - bodyWidth)
				{
					const __m128i mask_ = _mm_and_si128(tailMask, Load<align>((__m128i*)(mask + width - A)));
					const __m128i a_ = _mm_and_si128(mask_, Load<false>((__m128i*)(a + width - A)));
					const __m128i b_ = _mm_and_si128(mask_, Load<false>((__m128i*)(b + width - A))); 
					rowSum = _mm_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				fullSum = _mm_add_epi64(fullSum, HorizontalSum32(rowSum));
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
			*sum = ExtractInt64Sum(fullSum);
		}

		void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
				SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
			else
				SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
		}

		void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
				SquaredDifferenceSum<true>(a, aStride, b, bStride, mask, maskStride, width, height, sum);
			else
				SquaredDifferenceSum<false>(a, aStride, b, bStride, mask, maskStride, width, height, sum);
		}
    }
#endif// SIMD_SSE2_ENABLE

	void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
		size_t width, size_t height, uint64_t * sum)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
		else
#endif//SIMD_SSE2_ENABLE
			Base::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
	}

	void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
		const uchar *mask, size_t maskStride, size_t width, size_t height, uint64_t * sum)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::SquaredDifferenceSum(a, aStride, b, bStride, mask, maskStride, width, height, sum);
		else
#endif//SIMD_SSE2_ENABLE
			Base::SquaredDifferenceSum(a, aStride, b, bStride, mask, maskStride, width, height, sum);
	}

	void SquaredDifferenceSum(const View & a, const View & b, uint64_t & sum)
	{
		assert(a.width == b.width && a.height == b.height);
		assert(a.format == View::Gray8 && b.format == View::Gray8);

		SquaredDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
	}

	void SquaredDifferenceSum(const View & a, const View & b, const View & mask, uint64_t & sum)
	{
		assert(a.width == b.width && a.height == b.height && a.width == mask.width && a.height == mask.height);
		assert(a.format == View::Gray8 && b.format == View::Gray8 && mask.format == View::Gray8);

		SquaredDifferenceSum(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, a.width, a.height, &sum);
	}
}
