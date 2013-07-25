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
#include "Simd/SimdSet.h"
#include "Simd/SimdAbsDifferenceSum.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template <bool align> void AbsDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
            assert(width >= A);
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uchar>(0, A - width + bodyWidth, 0xFF);
			__m256i fullSum = _mm256_setzero_si256();
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i a_ = Load<align>((__m256i*)(a + col));
					const __m256i b_ = Load<align>((__m256i*)(b + col));
					fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
				}
				if(width - bodyWidth)
				{
					const __m256i a_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(a + width - A)));
					const __m256i b_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(b + width - A))); 
					fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
				}
				a += aStride;
				b += bStride;
			}
            *sum = ExtractSum<uint64_t>(fullSum);
		}

		template <bool align> void AbsDifferenceSum(
			const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
			const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
		{
            assert(width >= A);
			if(align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uchar>(0, A - width + bodyWidth, 0xFF);
			__m256i fullSum = _mm256_setzero_si256();
			__m256i index_= _mm256_set1_epi8(index);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i mask_ = LoadMaskI8<align>((__m256i*)(mask + col), index_);
					const __m256i a_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(a + col)));
					const __m256i b_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(b + col))); 
					fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
				}
				if(width - bodyWidth)
				{
					const __m256i mask_ = _mm256_and_si256(tailMask, LoadMaskI8<align>((__m256i*)(mask + width - A), index_));
					const __m256i a_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(a + width - A)));
					const __m256i b_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(b + width - A))); 
					fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
				}
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
			*sum = ExtractSum<uint64_t>(fullSum);
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
			const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
				AbsDifferenceSum<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
			else
				AbsDifferenceSum<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
		}
	}
#endif// SIMD_AVX2_ENABLE
}
