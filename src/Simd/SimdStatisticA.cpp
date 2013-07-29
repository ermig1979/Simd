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
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStatistic.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template <bool align> void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			assert(width*height && width >= A);
			if(align)
				assert(Aligned(src) && Aligned(stride));

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uchar>(0, A - width + bodyWidth, 0xFF);
			__m256i sum = _mm256_setzero_si256();
			__m256i min_ = K_INV_ZERO;
			__m256i max_ = K_ZERO;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i value = Load<align>((__m256i*)(src + col));
					min_ = _mm256_min_epu8(min_, value);
					max_ = _mm256_max_epu8(max_, value);
					sum = _mm256_add_epi64(_mm256_sad_epu8(value, K_ZERO), sum);
				}
				if(width - bodyWidth)
				{
					const __m256i value = Load<false>((__m256i*)(src + width - A));
					min_ = _mm256_min_epu8(min_, value);
					max_ = _mm256_max_epu8(max_, value);
					sum = _mm256_add_epi64(_mm256_sad_epu8(_mm256_and_si256(tailMask, value), K_ZERO), sum);
				}
				src += stride;
			}

			uchar min_buffer[A], max_buffer[A];
			_mm256_storeu_si256((__m256i*)min_buffer, min_);
			_mm256_storeu_si256((__m256i*)max_buffer, max_);
			*min = UCHAR_MAX;
			*max = 0;
			for (size_t i = 0; i < A; ++i)
			{
				*min = Base::MinU8(min_buffer[i], *min);
				*max = Base::MaxU8(max_buffer[i], *max);
			}
			*average = (uchar)((ExtractSum<uint64_t>(sum) + UCHAR_MAX/2)/(width*height));
		}

		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			if(Aligned(src) && Aligned(stride))
				GetStatistic<true>(src, stride, width, height, min, max, average);
			else
				GetStatistic<false>(src, stride, width, height, min, max, average);
		}
	}
#endif// SIMD_AVX2_ENABLE
}