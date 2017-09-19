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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
		template <bool align> SIMD_INLINE void GetStatistic(const uint8_t * src, __m512i & min, __m512i & max, __m512i & sum)
		{
			const __m512i _src = Load<align>(src);
			min = _mm512_min_epu8(min, _src);
			max = _mm512_max_epu8(max, _src);
			sum = _mm512_add_epi64(_mm512_sad_epu8(_src, K_ZERO), sum);
		}

		template <bool align> SIMD_INLINE void GetStatistic(const uint8_t * src, __m512i & min, __m512i & max, __m512i & sum, __mmask64 tail)
		{
			const __m512i _src = Load<align, true>(src, tail);
			min = _mm512_mask_min_epu8(min, tail, min, _src);
			max = _mm512_mask_max_epu8(max, tail, max, _src);
			sum = _mm512_add_epi64(_mm512_sad_epu8(_src, K_ZERO), sum);
		}

        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			
			__m512i sum = _mm512_setzero_si512();
            __m512i min512 = _mm512_set1_epi8(-1);
            __m512i max512 = _mm512_set1_epi8(0);
            for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
				for (; col < alignedWidth; col += A)
					GetStatistic<align>(src + col, min512, max512, sum);
				if (col < width)
					GetStatistic<align>(src + col, min512, max512, sum, tailMask);
                src += stride;
            }

			__m128i min128 = _mm_min_epu8(_mm_min_epu8(_mm512_extracti32x4_epi32(min512, 0), _mm512_extracti32x4_epi32(min512, 1)),
				_mm_min_epu8(_mm512_extracti32x4_epi32(min512, 2), _mm512_extracti32x4_epi32(min512, 3)));
			__m128i max128 = _mm_max_epu8(_mm_max_epu8(_mm512_extracti32x4_epi32(max512, 0), _mm512_extracti32x4_epi32(max512, 1)),
				_mm_max_epu8(_mm512_extracti32x4_epi32(max512, 2), _mm512_extracti32x4_epi32(max512, 3)));

            uint8_t min_buffer[Sse2::A], max_buffer[Sse2::A];
			Sse2::Store<false>((__m128i*)min_buffer, min128);
			Sse2::Store<false>((__m128i*)max_buffer, max128);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < Sse2::A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum<uint64_t>(sum) + width*height/2)/(width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if(Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
