/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __SimdExp_h__
#define __SimdExp_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE float Exp(float value)
        {
            return ::expf(value);
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class Exp
        {
            __m128i _exponent, _mantissa, _127;
            __m128 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m128 Poly5(__m128 x)
            {
                __m128 p = _exp5;
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp4);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp3);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp2);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp1);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp0);
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x)
            {
                x = _mm_max_ps(_mm_min_ps(x, _max), _min);
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _0_5));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _127), 23));
                __m128 expfpart = Poly5(fpart);
                return _mm_mul_ps(expipart, expfpart);
            }

        public:

            SIMD_INLINE Exp(float k = 1.0f)
            {
                _exponent = _mm_set1_epi32(0x7F800000);
                _mantissa = _mm_set1_epi32(0x007FFFFF);
                _127 = _mm_set1_epi32(127);
                _1_0 = _mm_set1_ps(1.0f);
                _0_5 = _mm_set1_ps(0.5f);
                _min = _mm_set1_ps(-126.99999f);
                _max = _mm_set1_ps(129.00000f);
                _exp0 = _mm_set1_ps(9.9999994e-1f);
                _exp1 = _mm_set1_ps(6.9315308e-1f);
                _exp2 = _mm_set1_ps(2.4015361e-1f);
                _exp3 = _mm_set1_ps(5.5826318e-2f);
                _exp4 = _mm_set1_ps(8.9893397e-3f);
                _exp5 = _mm_set1_ps(1.8775767e-3f);
                _k = _mm_set1_ps(k / 0.69314718056f);
            }

            SIMD_INLINE __m128 Exponent(__m128 value)
            {
                return Exp2(_mm_mul_ps(_k, value));
            }

            SIMD_INLINE __m128 Sigmoid(__m128 value)
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_1_0, _mm_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m128 Tanh(__m128 value)
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_mm_sub_ps(_1_0, exp), _mm_add_ps(_1_0, exp));
            }
        };
    }
#endif //SIMD_SSE2_ENABLE   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class Exp
        {
            __m256i _exponent, _mantissa, _127;
            __m256 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m256 Poly5(__m256 x)
            {
                __m256 p = _exp5;
                p = _mm256_fmadd_ps(x, p, _exp4);
                p = _mm256_fmadd_ps(x, p, _exp3);
                p = _mm256_fmadd_ps(x, p, _exp2);
                p = _mm256_fmadd_ps(x, p, _exp1);
                p = _mm256_fmadd_ps(x, p, _exp0);
                return p;
            }

            SIMD_INLINE __m256 Exp2(__m256 x)
            {
                x = _mm256_max_ps(_mm256_min_ps(x, _max), _min);
                __m256i ipart = _mm256_cvtps_epi32(_mm256_sub_ps(x, _0_5));
                __m256 fpart = _mm256_sub_ps(x, _mm256_cvtepi32_ps(ipart));
                __m256 expipart = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ipart, _127), 23));
                __m256 expfpart = Poly5(fpart);
                return _mm256_mul_ps(expipart, expfpart);
            }

        public:

            SIMD_INLINE Exp(float k = 1.0f)
            {
                _exponent = _mm256_set1_epi32(0x7F800000);
                _mantissa = _mm256_set1_epi32(0x007FFFFF);
                _127 = _mm256_set1_epi32(127);
                _1_0 = _mm256_set1_ps(1.0f);
                _0_5 = _mm256_set1_ps(0.5f);
                _min = _mm256_set1_ps(-126.99999f);
                _max = _mm256_set1_ps(129.00000f);
                _exp0 = _mm256_set1_ps(9.9999994e-1f);
                _exp1 = _mm256_set1_ps(6.9315308e-1f);
                _exp2 = _mm256_set1_ps(2.4015361e-1f);
                _exp3 = _mm256_set1_ps(5.5826318e-2f);
                _exp4 = _mm256_set1_ps(8.9893397e-3f);
                _exp5 = _mm256_set1_ps(1.8775767e-3f);
                _k = _mm256_set1_ps(k / 0.69314718056f);
            }

            SIMD_INLINE __m256 Exponent(__m256 value)
            {
                return Exp2(_mm256_mul_ps(_k, value));
            }

            SIMD_INLINE __m256 Sigmoid(__m256 value)
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                return _mm256_div_ps(_1_0, _mm256_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m256 Tanh(__m256 value)
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                return _mm256_div_ps(_mm256_sub_ps(_1_0, exp), _mm256_add_ps(_1_0, exp));
            }
        };
    }
#endif //SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class Exp
        {
            __m512i _exponent, _mantissa, _127;
            __m512 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m512 Poly5(__m512 x)
            {
                __m512 p = _exp5;
                p = _mm512_fmadd_ps(x, p, _exp4);
                p = _mm512_fmadd_ps(x, p, _exp3);
                p = _mm512_fmadd_ps(x, p, _exp2);
                p = _mm512_fmadd_ps(x, p, _exp1);
                p = _mm512_fmadd_ps(x, p, _exp0);
                return p;
            }

            SIMD_INLINE __m512 Exp2(__m512 x)
            {
                x = _mm512_max_ps(_mm512_min_ps(x, _max), _min);
                __m512i ipart = _mm512_cvtps_epi32(_mm512_sub_ps(x, _0_5));
                __m512 fpart = _mm512_sub_ps(x, _mm512_cvtepi32_ps(ipart));
                __m512 expipart = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ipart, _127), 23));
                __m512 expfpart = Poly5(fpart);
                return _mm512_mul_ps(expipart, expfpart);
            }

        public:

            SIMD_INLINE Exp(float k = 1.0f)
            {
                _exponent = _mm512_set1_epi32(0x7F800000);
                _mantissa = _mm512_set1_epi32(0x007FFFFF);
                _127 = _mm512_set1_epi32(127);
                _1_0 = _mm512_set1_ps(1.0f);
                _0_5 = _mm512_set1_ps(0.5f);
                _min = _mm512_set1_ps(-126.99999f);
                _max = _mm512_set1_ps(129.00000f);
                _exp0 = _mm512_set1_ps(9.9999994e-1f);
                _exp1 = _mm512_set1_ps(6.9315308e-1f);
                _exp2 = _mm512_set1_ps(2.4015361e-1f);
                _exp3 = _mm512_set1_ps(5.5826318e-2f);
                _exp4 = _mm512_set1_ps(8.9893397e-3f);
                _exp5 = _mm512_set1_ps(1.8775767e-3f);
                _k = _mm512_set1_ps(k / 0.69314718056f);
            }

            SIMD_INLINE __m512 Exponent(__m512 value)
            {
                return Exp2(_mm512_mul_ps(_k, value));
            }

            SIMD_INLINE __m512 Sigmoid(__m512 value)
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                return _mm512_div_ps(_1_0, _mm512_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m512 Tanh(__m512 value)
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                return _mm512_div_ps(_mm512_sub_ps(_1_0, exp), _mm512_add_ps(_1_0, exp));
            }
        };
    }
#endif //SIMD_AVX512F_ENABLE
}

#endif//__SimdExp_h__
