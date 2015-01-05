/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdInit.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i SquaredDifference(__m256i a, __m256i b)
        {
            const __m256i aLo = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
            const __m256i bLo = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
            const __m256i dLo = _mm256_sub_epi16(aLo, bLo);

            const __m256i aHi = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
            const __m256i bHi = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());
            const __m256i dHi = _mm256_sub_epi16(aHi, bHi);

            return _mm256_add_epi32(_mm256_madd_epi16(dLo, dLo), _mm256_madd_epi16(dHi, dHi));
        }

		template <bool align> void SquaredDifferenceSum(
			const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);
			if(align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
			}

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
			__m256i fullSum = _mm256_setzero_si256();
			for(size_t row = 0; row < height; ++row)
			{
				__m256i rowSum = _mm256_setzero_si256();
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i a_ = Load<align>((__m256i*)(a + col));
					const __m256i b_ = Load<align>((__m256i*)(b + col)); 
					rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				if(width - bodyWidth)
				{
					const __m256i a_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(a + width - A)));
					const __m256i b_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(b + width - A))); 
					rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
				a += aStride;
				b += bStride;
			}
			*sum = ExtractSum<uint64_t>(fullSum);
		}

		template <bool align> void SquaredDifferenceSumMasked(
			const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
			const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
		{
			assert(width < 0x10000);
			if(align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
			__m256i fullSum = _mm256_setzero_si256();
			__m256i index_= _mm256_set1_epi8(index);
			for(size_t row = 0; row < height; ++row)
			{
				__m256i rowSum = _mm256_setzero_si256();
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i mask_ = LoadMaskI8<align>((__m256i*)(mask + col), index_);
					const __m256i a_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(a + col)));
					const __m256i b_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(b + col))); 
					rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				if(width - bodyWidth)
				{
					const __m256i mask_ = _mm256_and_si256(tailMask, LoadMaskI8<align>((__m256i*)(mask + width - A), index_));
					const __m256i a_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(a + width - A)));
					const __m256i b_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(b + width - A))); 
					rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
				}
				fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
			*sum = ExtractSum<uint64_t>(fullSum);
		}

		void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
			size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
				SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
			else
				SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
		}

		void SquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
			const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
				SquaredDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
			else
				SquaredDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
		}
    }
#endif// SIMD_AVX2_ENABLE
}
