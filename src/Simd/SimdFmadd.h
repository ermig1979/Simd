/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#ifndef __SimdFmadd_h__
#define __SimdFmadd_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template<bool nofma> __m128 Fmadd(__m128 a, __m128 b, __m128 c);

        template <> SIMD_INLINE __m128 Fmadd<false>(__m128 a, __m128 b, __m128 c)
        {
            return _mm_fmadd_ps(a, b, c);
        }

        template <> SIMD_INLINE __m128 Fmadd<true>(__m128 a, __m128 b, __m128 c)
        {
            return _mm_add_ps(_mm_or_ps(_mm_mul_ps(a, b), _mm_setzero_ps()), c);
        }

        //-----------------------------------------------------------------------------------------

        template<bool nofma> __m128 Fmadd(__m128 a, __m128 b, __m128 c, const __m128 & d);

        template <> SIMD_INLINE __m128 Fmadd<false>(__m128 a, __m128 b, __m128 c, const __m128 & d)
        {
            return _mm_fmadd_ps(a, b, _mm_mul_ps(c, d));
        }

        template <> SIMD_INLINE __m128 Fmadd<true>(__m128 a, __m128 b, __m128 c, const __m128 & d)
        {
            return _mm_add_ps(_mm_or_ps(_mm_mul_ps(a, b), _mm_setzero_ps()), _mm_or_ps(_mm_mul_ps(c, d), _mm_setzero_ps()));
        }

        //-----------------------------------------------------------------------------------------

        template<bool nofma> __m256 Fmadd(__m256 a, __m256 b, __m256 c);

        template <> SIMD_INLINE __m256 Fmadd<false>(__m256 a, __m256 b, __m256 c)
        {
            return _mm256_fmadd_ps(a, b, c);
        }

        template <> SIMD_INLINE __m256 Fmadd<true>(__m256 a, __m256 b, __m256 c)
        {
            return _mm256_add_ps(_mm256_or_ps(_mm256_mul_ps(a, b), _mm256_setzero_ps()), c);
        }

        //-----------------------------------------------------------------------------------------

        template<bool nofma> __m256 Fmadd(__m256 a, __m256 b, __m256 c, const __m256 &  d);

        template <> SIMD_INLINE __m256 Fmadd<false>(__m256 a, __m256 b, __m256 c, const __m256 & d)
        {
            return _mm256_fmadd_ps(a, b, _mm256_mul_ps(c, d));
        }

        template <> SIMD_INLINE __m256 Fmadd<true>(__m256 a, __m256 b, __m256 c, const __m256 & d)
        {
            return _mm256_add_ps(_mm256_or_ps(_mm256_mul_ps(a, b), _mm256_setzero_ps()), _mm256_or_ps(_mm256_mul_ps(c, d), _mm256_setzero_ps()));
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool nofma> __m512 Fmadd(__m512 a, __m512 b, __m512 c);

        template <> SIMD_INLINE __m512 Fmadd<false>(__m512 a, __m512 b, __m512 c)
        {
            return _mm512_fmadd_ps(a, b, c);
        }

        template <> SIMD_INLINE __m512 Fmadd<true>(__m512 a, __m512 b, __m512 c)
        {
#ifdef _MSC_VER
            return _mm512_add_ps(_mm512_fmadd_ps(a, b, _mm512_setzero_ps()), c);
#else
            return _mm512_maskz_add_ps(-1, _mm512_mul_ps(a, b), c);
#endif
        }
    }
#endif
}

#endif
