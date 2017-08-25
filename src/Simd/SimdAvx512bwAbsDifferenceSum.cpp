/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdLoad.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
		template <bool align> void AbsDifferenceSum2(const uint8_t * a, const uint8_t * b, __m512i * sums)
		{
			const __m512i a0 = Load<align>(a + 0);
			const __m512i b0 = Load<align>(b + 0);
			sums[0] = _mm512_add_epi64(_mm512_sad_epu8(a0, b0), sums[0]);
			const __m512i a1 = Load<align>(a + A);
			const __m512i b1 = Load<align>(b + A);
			sums[1] = _mm512_add_epi64(_mm512_sad_epu8(a1, b1), sums[1]);
		}

		template <bool align, bool mask> void AbsDifferenceSum1(const uint8_t * a, const uint8_t * b, __m512i * sums, __mmask64 m = -1)
		{
			const __m512i a0 = Load<align, mask>(a, m);
			const __m512i b0 = Load<align, mask>(b, m);
			sums[0] = _mm512_add_epi64(_mm512_sad_epu8(a0, b0), sums[0]);
		}

		template <bool align> void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, size_t width, size_t height, uint64_t * sum)
		{
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

			size_t fullAlignedWidth = AlignLo(width, DA);
			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
			__m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
			for(size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < fullAlignedWidth; col += DA)
					AbsDifferenceSum2<align>(a + col, b + col, sums);
				for (; col < alignedWidth; col += A)
					AbsDifferenceSum1<align, false>(a + col, b + col, sums);
				if (col < width)
					AbsDifferenceSum1<align, true>(a + col, b + col, sums, tailMask);
				a += aStride;
				b += bStride;
			}
			sums[0] = _mm512_add_epi64(sums[0], sums[1]);
            *sum = ExtractSum<uint64_t>(sums[0]);
		}

		void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
			size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
				AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
			else
				AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
		}

		template <bool align> void AbsDifferenceSumMasked2(const uint8_t * a, const uint8_t * b, const uint8_t * m, const __m512i & index, __m512i * sums)
		{
			__mmask64 m0 = _mm512_cmpeq_epu8_mask(Load<align>(m + 0), index);
			const __m512i a0 = Load<align, true>(a + 0, m0);
			const __m512i b0 = Load<align, true>(b + 0, m0);
			sums[0] = _mm512_add_epi64(_mm512_sad_epu8(a0, b0), sums[0]);
			__mmask64 m1 = _mm512_cmpeq_epu8_mask(Load<align>(m + A), index);
			const __m512i a1 = Load<align, true>(a + A, m1);
			const __m512i b1 = Load<align, true>(b + A, m1);
			sums[1] = _mm512_add_epi64(_mm512_sad_epu8(a1, b1), sums[1]);
		}

		template <bool align, bool mask> void AbsDifferenceSumMasked1(const uint8_t * a, const uint8_t * b, const uint8_t * m, __m512i & index, __m512i * sums, __mmask64 mm = -1)
		{
			__mmask64 m0 = _mm512_cmpeq_epu8_mask(Load<align>(m + 0), index) & mm;
			const __m512i a0 = Load<align, true>(a + 0, m0);
			const __m512i b0 = Load<align, true>(b + 0, m0);
			sums[0] = _mm512_add_epi64(_mm512_sad_epu8(a0, b0), sums[0]);
		}

		template <bool align> void AbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
		{
			if (align)
			{
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			__m512i _index = _mm512_set1_epi8(index);
			size_t fullAlignedWidth = AlignLo(width, DA);
			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
			__m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < fullAlignedWidth; col += DA)
					AbsDifferenceSumMasked2<align>(a + col, b + col, mask + col, _index, sums);
				for (; col < alignedWidth; col += A)
					AbsDifferenceSumMasked1<align, false>(a + col, b + col, mask + col, _index, sums);
				if (col < width)
					AbsDifferenceSumMasked1<align, true>(a + col, b + col, mask + col, _index, sums, tailMask);
				a += aStride;
				b += bStride;
				mask += maskStride;
			}
			sums[0] = _mm512_add_epi64(sums[0], sums[1]);
			*sum = ExtractSum<uint64_t>(sums[0]);
		}

		void AbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
				AbsDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
			else
				AbsDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
		}
	}
#endif// SIMD_AVX512BW_ENABLE
}
