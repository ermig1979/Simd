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
#ifndef __SimdErf_h__
#define __SimdErf_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdExp.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE float Erf(float value)
        {
            return ::erff(value);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        namespace Detail
        {
            SIMD_INLINE __m128 Poly4(__m128 x, float a, float b, float c, float d, float e)
            {
                __m128 p = _mm_set1_ps(e);
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(d));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(c));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(b));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m128 ExpNegSqr(__m128 x)
            {
                x = _mm_mul_ps(_mm_set1_ps(-1.44269504f), _mm_mul_ps(x, x));
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));
                __m128 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm_mul_ps(expipart, expfpart);
            }
        }

        SIMD_INLINE __m128 Erf(__m128 x)
        {
            const __m128 _max = _mm_set1_ps(9);
            const __m128 _m0 = _mm_set1_ps(-0.0f);
            const __m128 _1 = _mm_set1_ps(1.0f);
            __m128 a, p, q, r;
            a = _mm_min_ps(_mm_andnot_ps(_m0, x), _max);
            q = _mm_div_ps(_1, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.3275911f), a), _1));
            p = Detail::Poly4(q, 0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f, 1.061405429f);
            r = _mm_sub_ps(_1, _mm_mul_ps(_mm_mul_ps(p, q), Detail::ExpNegSqr(a)));
            return _mm_or_ps(_mm_and_ps(_m0, x), r);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        namespace Detail
        {
            SIMD_INLINE __m256 Poly4(__m256 x, float a, float b, float c, float d, float e)
            {
                __m256 p = _mm256_set1_ps(e);
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(d));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(c));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(b));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m256 ExpNegSqr(__m256 x)
            {
                x = _mm256_mul_ps(_mm256_set1_ps(-1.44269504f), _mm256_mul_ps(x, x));
                __m256i ipart = _mm256_cvtps_epi32(_mm256_sub_ps(x, _mm256_set1_ps(0.5f)));
                __m256 fpart = _mm256_sub_ps(x, _mm256_cvtepi32_ps(ipart));
                __m256 expipart = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ipart, _mm256_set1_epi32(127)), 23));
                __m256 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm256_mul_ps(expipart, expfpart);
            }
        }

        SIMD_INLINE __m256 Erf(__m256 x)
        {
            const __m256 _max = _mm256_set1_ps(9);
            const __m256 _m0 = _mm256_set1_ps(-0.0f);
            const __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 a, p, q, r;
            a = _mm256_min_ps(_mm256_andnot_ps(_m0, x), _max);
            q = _mm256_div_ps(_1, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), a, _1));
            p = Detail::Poly4(q, 0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f, 1.061405429f);
            r = _mm256_fnmadd_ps(_mm256_mul_ps(p, q), Detail::ExpNegSqr(a), _1);
            return _mm256_or_ps(_mm256_and_ps(_m0, x), r);
        }
    }
#endif 

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        namespace Detail
        {
            SIMD_INLINE __m512 Poly4(__m512 x, float a, float b, float c, float d, float e)
            {
                __m512 p = _mm512_set1_ps(e);
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(d));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(c));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(b));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m512 ExpNegSqr(__m512 x)
            {
                x = _mm512_mul_ps(_mm512_set1_ps(-1.44269504f), _mm512_mul_ps(x, x));
                __m512i ipart = _mm512_cvtps_epi32(_mm512_sub_ps(x, _mm512_set1_ps(0.5f)));
                __m512 fpart = _mm512_sub_ps(x, _mm512_cvtepi32_ps(ipart));
                __m512 expipart = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ipart, _mm512_set1_epi32(127)), 23));
                __m512 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm512_mul_ps(expipart, expfpart);
            }
        }

        SIMD_INLINE __m512 Erf(__m512 x)
        {
            const __m512 _max = _mm512_set1_ps(9);
            const __m512 _m0 = _mm512_set1_ps(-0.0f);
            const __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 a, p, q, r;
            a = _mm512_min_ps(_mm512_andnot_ps(_m0, x), _max);
            q = _mm512_div_ps(_1, _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), a, _1));
            p = Detail::Poly4(q, 0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f, 1.061405429f);
            r = _mm512_fnmadd_ps(_mm512_mul_ps(p, q), Detail::ExpNegSqr(a), _1);
            return _mm512_or_ps(_mm512_and_ps(_m0, x), r);
        }
    }
#endif 
}
#endif//__SimdErf_h__
