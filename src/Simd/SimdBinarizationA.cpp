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
#include "Simd/SimdBinarization.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template<CompareType compareType> SIMD_INLINE __m256i Compare(__m256i a, __m256i b);

		template<> SIMD_INLINE __m256i Compare<CompareGreaterThen>(__m256i a, __m256i b)
		{
			return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a), K_INV_ZERO);
		}

		template<> SIMD_INLINE __m256i Compare<CompareLesserThen>(__m256i a, __m256i b)
		{
			return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a), K_INV_ZERO);
		}

		template<> SIMD_INLINE __m256i Compare<CompareEqualTo>(__m256i a, __m256i b)
		{
			return _mm256_cmpeq_epi8(a, b);
		}

		SIMD_INLINE __m256i Combine(__m256i mask, __m256i positive, __m256i negative)
		{
			return _mm256_or_si256(_mm256_and_si256(mask, positive), _mm256_andnot_si256(mask, negative));
		}

		template <bool align, CompareType compareType> 
		void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
			uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride)
		{
			assert(width >= A);
			if(align)
				assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

			size_t alignedWidth = Simd::AlignLo(width, A);

			__m256i value_ = _mm256_set1_epi8(value);
			__m256i positive_ = _mm256_set1_epi8(positive);
			__m256i negative_ = _mm256_set1_epi8(negative);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
				{
					const __m256i mask = Compare<compareType>(Load<align>((__m256i*)(src + col)), value_);
					Store<align>((__m256i*)(dst + col), Combine(mask, positive_, negative_));
				}
				if(alignedWidth != width)
				{
					const __m256i mask = Compare<compareType>(Load<false>((__m256i*)(src + width - A)), value_);
					Store<false>((__m256i*)(dst + width - A), Combine(mask, positive_, negative_));
				}
				src += srcStride;
				dst += dstStride;
			}
		}

        template <CompareType compareType> 
		void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
			uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride)
		{
			if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
				Binarization<true, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
			else
				Binarization<false, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
		}

		void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
			uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, CompareType compareType)
		{
            switch(compareType)
            {
            case CompareGreaterThen:
                return Binarization<CompareGreaterThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareLesserThen:
                return Binarization<CompareLesserThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareEqualTo:
                return Binarization<CompareEqualTo>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
		}
	}
#endif// SIMD_AVX2_ENABLE
}