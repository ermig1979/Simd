/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2019-2019 Facundo Galan,
*               2026-2026 Yu Changming.
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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		void AbsDifference(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, uint8_t *c, size_t cStride,
			size_t width, size_t height)
		{
			assert(width >= A);
			size_t bodyWidth = AlignLo(width, A);
			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i a_ = _mm256_loadu_si256((__m256i*)(a + col));
					const __m256i b_ = _mm256_loadu_si256((__m256i*)(b + col));
					_mm256_storeu_si256((__m256i*)(c + col), _mm256_sub_epi8(_mm256_max_epu8(a_, b_), _mm256_min_epu8(a_, b_)));
				}
				if (width - bodyWidth)
				{
					const __m256i a_ = _mm256_loadu_si256((__m256i*)(a + width - A));
					const __m256i b_ = _mm256_loadu_si256((__m256i*)(b + width - A));
					_mm256_storeu_si256((__m256i*)(c + width - A), _mm256_sub_epi8(_mm256_max_epu8(a_, b_), _mm256_min_epu8(a_, b_)));
				}
				a += aStride;
				b += bStride;
				c += cStride;
			}
		}
	}
#endif// SIMD_AVX2_ENABLE
}
