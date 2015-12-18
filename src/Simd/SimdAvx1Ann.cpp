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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_a, _b));
        }

        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 8);
            size_t fullAlignedSize = AlignLo(size, 32);
            size_t i = 0;
            if(partialAlignedSize)
            {
                __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += 32)
                    {
						AnnProductSum<align>(a, b, i, sums[0]);
						AnnProductSum<align>(a, b, i + 8, sums[1]);
						AnnProductSum<align>(a, b, i + 16, sums[2]);
						AnnProductSum<align>(a, b, i + 24, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for(; i < partialAlignedSize; i += 8)
					AnnProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += a[i]*b[i];
        }

        void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
				AnnProductSum<true>(a, b, size, sum);
            else
				AnnProductSum<false>(a, b, size, sum);
        }

		template <bool align> SIMD_INLINE void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			size_t alignedSize = Simd::AlignLo(size, 8);
			__m256 _slope = _mm256_set1_ps(*slope);
			__m256 _0 = _mm256_set1_ps(-0.0f);
			__m256 _1 = _mm256_set1_ps(1.0f);
			__m256 _0555 = _mm256_set1_ps(0.555f);
			__m256 _0143 = _mm256_set1_ps(0.143f);
			size_t i = 0;
			for (; i < alignedSize; i += 8)
			{
				__m256 _src = Load<align>(src + i);
				__m256 x = _mm256_andnot_ps(_0, _mm256_mul_ps(_src, _slope));
				__m256 x2 = _mm256_mul_ps(x, x);
				__m256 x4 = _mm256_mul_ps(x2, x2);
				__m256 series = _mm256_add_ps(_mm256_add_ps(_1, x), _mm256_add_ps(_mm256_mul_ps(x2, _0555), _mm256_mul_ps(x4, _0143)));
				__m256 mask = _mm256_cmp_ps(_src, _0, _CMP_GT_OS);
				__m256 exp = _mm256_or_ps(_mm256_and_ps(_mm256_rcp_ps(series), mask), _mm256_andnot_ps(mask, series));
				__m256 sigmoid = _mm256_rcp_ps(_mm256_add_ps(_1, exp));
				Store<align>(dst + i, sigmoid);
			}
			for (; i < size; ++i)
				dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
		}

		void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				AnnRoughSigmoid<true>(src, size, slope, dst);
			else
				AnnRoughSigmoid<false>(src, size, slope, dst);
		}
    }
#endif// SIMD_AVX_ENABLE
}
