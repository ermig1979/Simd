/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

        SIMD_INLINE float Log(float value)
        {
            return ::logf(value);
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class Exp
        {
            __m128i _exponent, _mantissa, _127;
            __m128 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m128 Poly5(__m128 x) const
            {
                __m128 p = _exp5;
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp4);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp3);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp2);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp1);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp0);
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x) const
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

            SIMD_INLINE __m128 Exponent(__m128 value) const
            {
                return Exp2(_mm_mul_ps(_k, value));
            }

            SIMD_INLINE __m128 Sigmoid(__m128 value) const
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_1_0, _mm_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m128 Tanh(__m128 value) const
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_mm_sub_ps(_1_0, exp), _mm_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m128 Elu(__m128 value, __m128 alpha) const
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                __m128 neg = _mm_mul_ps(alpha, _mm_sub_ps(exp, _1_0));
                __m128 mask = _mm_cmpgt_ps(_mm_setzero_ps(), value);
                return Combine(mask, neg, value);
            }

            SIMD_INLINE __m128 Swish(__m128 value) const
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(value, _mm_add_ps(_1_0, exp));
            }
        };

        namespace Detail
        {
            SIMD_INLINE __m128 Poly5(__m128 x, float a, float b, float c, float d, float e, float f)
            {
                __m128 p = _mm_set1_ps(f);
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(e));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(d));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(c));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(b));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x)
            {
                x = _mm_max_ps(_mm_min_ps(x, _mm_set1_ps(129.00000f)), _mm_set1_ps(-126.99999f));
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));
                __m128 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m128 Log2(__m128 x)
            {
                __m128 _1 = _mm_set1_ps(1.0f);
                __m128i i = _mm_castps_si128(x);
                __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(i, _mm_set1_epi32(0x7F800000)), 23), _mm_set1_epi32(127)));
                __m128 m = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, _mm_set1_epi32(0x007FFFFF))), _1);
                __m128 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm_add_ps(_mm_mul_ps(p, _mm_sub_ps(m, _1)), e);
            }
        }

        SIMD_INLINE __m128 Exponent(__m128 value)
        {
            return Detail::Exp2(_mm_mul_ps(_mm_set1_ps(1.44269504f), value));
        }

        SIMD_INLINE __m128 Elu(__m128 value, __m128 alpha) 
        {
            __m128 exp = Exponent(value);
            __m128 neg = _mm_mul_ps(alpha, _mm_sub_ps(exp, _mm_set1_ps(1.0f)));
            __m128 mask = _mm_cmpgt_ps(_mm_setzero_ps(), value);
            return Combine(mask, neg, value);
        }

        SIMD_INLINE __m128 Logarithm(__m128 value)
        {
            return _mm_mul_ps(_mm_set1_ps(0.693147181f), Detail::Log2(value));
        }

        SIMD_INLINE __m128 Mish(__m128 value, __m128 threshold)
        {
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 mish = _mm_add_ps(Exponent(value), _1);
            mish = _mm_add_ps(_mm_mul_ps(mish, mish), _1);
            mish = _mm_mul_ps(value, _mm_sub_ps(_1, _mm_div_ps(_mm_set1_ps(2.0f), mish)));
            return Combine(_mm_cmpgt_ps(threshold, value), mish, value);
        }

        SIMD_INLINE __m128 Softplus(__m128 value, __m128 beta, __m128 threshold)
        {
            __m128 exp = Exponent(_mm_mul_ps(value, beta));
            __m128 log = Logarithm(_mm_add_ps(_mm_set1_ps(1.0f), exp));
            __m128 mask = _mm_cmpgt_ps(threshold, value);
            return Combine(mask, _mm_div_ps(log, beta), value);
        }

        SIMD_INLINE __m128 Swish(__m128 value, __m128 slope)
        {
            __m128 exp = Exponent(_mm_sub_ps(_mm_setzero_ps(), _mm_mul_ps(value, slope)));
            return _mm_div_ps(value, _mm_add_ps(_mm_set1_ps(1.0f), exp));
        }

        SIMD_INLINE __m128 Tanh(__m128 value)
        {
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 exp = Detail::Exp2(_mm_mul_ps(_mm_set1_ps(2.88539008f), value));
            return _mm_div_ps(_mm_sub_ps(exp, _1), _mm_add_ps(_1, exp));
        }
    }
#endif //SIMD_SSE2_ENABLE   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class Exp
        {
            __m256i _exponent, _mantissa, _127;
            __m256 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m256 Poly5(__m256 x) const
            {
                __m256 p = _exp5;
                p = _mm256_fmadd_ps(x, p, _exp4);
                p = _mm256_fmadd_ps(x, p, _exp3);
                p = _mm256_fmadd_ps(x, p, _exp2);
                p = _mm256_fmadd_ps(x, p, _exp1);
                p = _mm256_fmadd_ps(x, p, _exp0);
                return p;
            }

            SIMD_INLINE __m256 Exp2(__m256 x) const
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

            SIMD_INLINE __m256 Exponent(__m256 value) const
            {
                return Exp2(_mm256_mul_ps(_k, value));
            }

            SIMD_INLINE __m256 Sigmoid(__m256 value) const
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                return _mm256_div_ps(_1_0, _mm256_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m256 Tanh(__m256 value) const
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                return _mm256_div_ps(_mm256_sub_ps(_1_0, exp), _mm256_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m256 Elu(__m256 value, __m256 alpha) const
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                __m256 neg = _mm256_mul_ps(alpha, _mm256_sub_ps(exp, _1_0));
                __m256 mask = _mm256_cmp_ps(_mm256_setzero_ps(), value, _CMP_GT_OS);
                return _mm256_blendv_ps(value, neg, mask);
            }

            SIMD_INLINE __m256 Swish(__m256 value) const
            {
                __m256 exp = Exp2(_mm256_mul_ps(_k, value));
                return _mm256_div_ps(value, _mm256_add_ps(_1_0, exp));
            }
        };

        namespace Detail
        {
            SIMD_INLINE __m256 Poly5(__m256 x, float a, float b, float c, float d, float e, float f)
            {
                __m256 p = _mm256_set1_ps(f);
                p = _mm256_add_ps(_mm256_mul_ps(x, p), _mm256_set1_ps(e));
                p = _mm256_add_ps(_mm256_mul_ps(x, p), _mm256_set1_ps(d));
                p = _mm256_add_ps(_mm256_mul_ps(x, p), _mm256_set1_ps(c));
                p = _mm256_add_ps(_mm256_mul_ps(x, p), _mm256_set1_ps(b));
                p = _mm256_add_ps(_mm256_mul_ps(x, p), _mm256_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m256 Exp2(__m256 x)
            {
                x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(129.00000f)), _mm256_set1_ps(-126.99999f));
                __m256i ipart = _mm256_cvtps_epi32(_mm256_sub_ps(x, _mm256_set1_ps(0.5f)));
                __m256 fpart = _mm256_sub_ps(x, _mm256_cvtepi32_ps(ipart));
                __m256 expipart = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ipart, _mm256_set1_epi32(127)), 23));
                __m256 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm256_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m256 Log2(__m256 x)
            {
                __m256 _1 = _mm256_set1_ps(1.0f);
                __m256i i = _mm256_castps_si256(x);
                __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0x7F800000)), 23), _mm256_set1_epi32(127)));
                __m256 m = _mm256_or_ps(_mm256_castsi256_ps(_mm256_and_si256(i, _mm256_set1_epi32(0x007FFFFF))), _1);
                __m256 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm256_add_ps(_mm256_mul_ps(p, _mm256_sub_ps(m, _1)), e);
            }
        }

        SIMD_INLINE __m256 Exponent(__m256 value)
        {
            return Detail::Exp2(_mm256_mul_ps(_mm256_set1_ps(1.44269504f), value));
        }

        SIMD_INLINE __m256 Elu(__m256 value, __m256 alpha)
        {
            __m256 exp = Exponent(value);
            __m256 neg = _mm256_mul_ps(alpha, _mm256_sub_ps(exp, _mm256_set1_ps(1.0f)));
            __m256 mask = _mm256_cmp_ps(_mm256_setzero_ps(), value, _CMP_GT_OS);
            return _mm256_blendv_ps(value, neg, mask);
        }

        SIMD_INLINE __m256 Logarithm(__m256 value)
        {
            return _mm256_mul_ps(_mm256_set1_ps(0.693147181f), Detail::Log2(value));
        }

        SIMD_INLINE __m256 Mish(__m256 value, __m256 threshold)
        {
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 mish = _mm256_add_ps(Exponent(value), _1);
            mish = Fmadd<true>(mish, mish, _1);
            mish = _mm256_mul_ps(value, _mm256_sub_ps(_1, _mm256_div_ps(_mm256_set1_ps(2.0f), mish)));
            return _mm256_blendv_ps(value, mish, _mm256_cmp_ps(threshold, value, _CMP_GT_OS));
        }

        SIMD_INLINE __m256 Softplus(__m256 value, __m256 beta, __m256 threshold)
        {
            __m256 exp = Exponent(_mm256_mul_ps(value, beta));
            __m256 log = Logarithm(_mm256_add_ps(_mm256_set1_ps(1.0f), exp));
            __m256 mask = _mm256_cmp_ps(threshold, value, _CMP_GT_OS);
            return _mm256_blendv_ps(value, _mm256_div_ps(log, beta), mask);
        }

        SIMD_INLINE __m256 Swish(__m256 value, __m256 slope)
        {
            __m256 exp = Exponent(_mm256_fnmadd_ps(value, slope, _mm256_setzero_ps()));
            return _mm256_div_ps(value, _mm256_add_ps(_mm256_set1_ps(1.0f), exp));
        }

        SIMD_INLINE __m256 Tanh(__m256 value)
        {
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 exp = Detail::Exp2(_mm256_mul_ps(_mm256_set1_ps(2.88539008f), value));
            return _mm256_div_ps(_mm256_sub_ps(exp, _1), _mm256_add_ps(_1, exp));
        }
    }
#endif //SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class Exp
        {
            __m512i _exponent, _mantissa, _127;
            __m512 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE __m512 Poly5(__m512 x) const
            {
                __m512 p = _exp5;
                p = _mm512_fmadd_ps(x, p, _exp4);
                p = _mm512_fmadd_ps(x, p, _exp3);
                p = _mm512_fmadd_ps(x, p, _exp2);
                p = _mm512_fmadd_ps(x, p, _exp1);
                p = _mm512_fmadd_ps(x, p, _exp0);
                return p;
            }

            SIMD_INLINE __m512 Exp2(__m512 x) const
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

            SIMD_INLINE __m512 Exponent(__m512 value) const
            {
                return Exp2(_mm512_mul_ps(_k, value));
            }

            SIMD_INLINE __m512 Sigmoid(__m512 value) const
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                return _mm512_div_ps(_1_0, _mm512_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m512 Tanh(__m512 value) const
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                return _mm512_div_ps(_mm512_sub_ps(_1_0, exp), _mm512_add_ps(_1_0, exp));
            }

            SIMD_INLINE __m512 Elu(__m512 value, __m512 alpha) const
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                __m512 neg = _mm512_mul_ps(alpha, _mm512_sub_ps(exp, _1_0));
                __mmask16 mask = _mm512_cmp_ps_mask(_mm512_setzero_ps(), value, _CMP_GT_OS);
                return _mm512_mask_blend_ps(mask, value, neg);
            }

            SIMD_INLINE __m512 Swish(__m512 value) const
            {
                __m512 exp = Exp2(_mm512_mul_ps(_k, value));
                return _mm512_div_ps(value, _mm512_add_ps(_1_0, exp));
            }
        };

        namespace Detail
        {
            SIMD_INLINE __m512 Poly5(__m512 x, float a, float b, float c, float d, float e, float f)
            {
                __m512 p = _mm512_set1_ps(f);
                p = _mm512_add_ps(_mm512_mul_ps(x, p), _mm512_set1_ps(e));
                p = _mm512_add_ps(_mm512_mul_ps(x, p), _mm512_set1_ps(d));
                p = _mm512_add_ps(_mm512_mul_ps(x, p), _mm512_set1_ps(c));
                p = _mm512_add_ps(_mm512_mul_ps(x, p), _mm512_set1_ps(b));
                p = _mm512_add_ps(_mm512_mul_ps(x, p), _mm512_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m512 Exp2(__m512 x)
            {
                x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(129.00000f)), _mm512_set1_ps(-126.99999f));
                __m512i ipart = _mm512_cvtps_epi32(_mm512_sub_ps(x, _mm512_set1_ps(0.5f)));
                __m512 fpart = _mm512_sub_ps(x, _mm512_cvtepi32_ps(ipart));
                __m512 expipart = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ipart, _mm512_set1_epi32(127)), 23));
                __m512 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm512_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m512 Log2(__m512 x)
            {
                __m512 _1 = _mm512_set1_ps(1.0f);
                __m512i i = _mm512_castps_si512(x);
                __m512 e = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_srli_epi32(_mm512_and_si512(i, _mm512_set1_epi32(0x7F800000)), 23), _mm512_set1_epi32(127)));
                __m512 m = Or(_mm512_castsi512_ps(_mm512_and_si512(i, _mm512_set1_epi32(0x007FFFFF))), _1);
                __m512 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm512_add_ps(_mm512_mul_ps(p, _mm512_sub_ps(m, _1)), e);
            }
        }

        SIMD_INLINE __m512 Exponent(__m512 value)
        {
            return Detail::Exp2(_mm512_mul_ps(_mm512_set1_ps(1.44269504f), value));
        }

        SIMD_INLINE __m512 Elu(__m512 value, __m512 alpha)
        {
            __m512 exp = Exponent(value);
            __m512 neg = _mm512_mul_ps(alpha, _mm512_sub_ps(exp, _mm512_set1_ps(1.0f)));
            __mmask16 mask = _mm512_cmp_ps_mask(_mm512_setzero_ps(), value, _CMP_GT_OS);
            return _mm512_mask_blend_ps(mask, value, neg);
        }

        SIMD_INLINE __m512 Logarithm(__m512 value)
        {
            return _mm512_mul_ps(_mm512_set1_ps(0.693147181f), Detail::Log2(value));
        }

        SIMD_INLINE __m512 Mish(__m512 value, __m512 threshold)
        {
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 mish = _mm512_add_ps(Exponent(value), _1);
            mish = Fmadd<true>(mish, mish, _1);
            mish = _mm512_mul_ps(value, _mm512_sub_ps(_1, _mm512_div_ps(_mm512_set1_ps(2.0f), mish)));
            return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(threshold, value, _CMP_GT_OS), value, mish);
        }

        SIMD_INLINE __m512 Softplus(__m512 value, __m512 beta, __m512 threshold)
        {
            __m512 exp = Exponent(_mm512_mul_ps(value, beta));
            __m512 log = Logarithm(_mm512_add_ps(_mm512_set1_ps(1.0f), exp));
            __mmask16 mask = _mm512_cmp_ps_mask(threshold, value, _CMP_GT_OS);
            return _mm512_mask_blend_ps(mask, value, _mm512_div_ps(log, beta));
        }

        SIMD_INLINE __m512 Swish(__m512 value, __m512 slope)
        {
            __m512 exp = Exponent(_mm512_fnmadd_ps(value, slope, _mm512_setzero_ps()));
            return _mm512_div_ps(value, _mm512_add_ps(_mm512_set1_ps(1.0f), exp));
        }

        SIMD_INLINE __m512 Tanh(__m512 value)
        {
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 exp = Detail::Exp2(_mm512_mul_ps(_mm512_set1_ps(2.88539008f), value));
            return _mm512_div_ps(_mm512_sub_ps(exp, _1), _mm512_add_ps(_1, exp));
        }
    }
#endif //SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class Exp
        {
            int32x4_t _exponent, _mantissa, _127;
            float32x4_t _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            SIMD_INLINE float32x4_t Poly5(float32x4_t x) const
            {
                float32x4_t p = _exp5;
                p = vmlaq_f32(_exp4, x, p);
                p = vmlaq_f32(_exp3, x, p);
                p = vmlaq_f32(_exp2, x, p);
                p = vmlaq_f32(_exp1, x, p);
                p = vmlaq_f32(_exp0, x, p);
                return p;
            }

            SIMD_INLINE float32x4_t Exp2(float32x4_t x) const
            {
                x = vmaxq_f32(vminq_f32(x, _max), _min);
                int32x4_t ipart = vcvtq_s32_f32(vsubq_f32(x, _0_5));
                float32x4_t fpart = vsubq_f32(x, vcvtq_f32_s32(ipart));
                float32x4_t expipart = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ipart, _127), 23));
                float32x4_t expfpart = Poly5(fpart);
                return vmulq_f32(expipart, expfpart);
            }

        public:

            SIMD_INLINE Exp(float k = 1.0f)
            {
                _exponent = vdupq_n_s32(0x7F800000);
                _mantissa = vdupq_n_s32(0x007FFFFF);
                _127 = vdupq_n_s32(127);
                _1_0 = vdupq_n_f32(1.0f);
                _0_5 = vdupq_n_f32(0.5f);
                _min = vdupq_n_f32(-126.99999f);
                _max = vdupq_n_f32(129.00000f);
                _exp0 = vdupq_n_f32(9.9999994e-1f);
                _exp1 = vdupq_n_f32(6.9315308e-1f);
                _exp2 = vdupq_n_f32(2.4015361e-1f);
                _exp3 = vdupq_n_f32(5.5826318e-2f);
                _exp4 = vdupq_n_f32(8.9893397e-3f);
                _exp5 = vdupq_n_f32(1.8775767e-3f);
                _k = vdupq_n_f32(k / 0.69314718056f);
            }

            SIMD_INLINE float32x4_t Exponent(float32x4_t value) const
            {
                return Exp2(vmulq_f32(_k, value));
            }

            template<int iter> SIMD_INLINE float32x4_t Sigmoid(float32x4_t value) const
            {
                float32x4_t exp = Exp2(vmulq_f32(_k, value));
                return Reciprocal<iter>(vaddq_f32(_1_0, exp));
            }

            template<int iter> SIMD_INLINE float32x4_t Tanh(float32x4_t value) const
            {
                float32x4_t exp = Exp2(vmulq_f32(_k, value));
                return Div<iter>(vsubq_f32(_1_0, exp), vaddq_f32(_1_0, exp));
            }

            SIMD_INLINE float32x4_t Elu(float32x4_t value, float32x4_t alpha) const
            {
                float32x4_t exp = Exp2(vmulq_f32(_k, value));
                float32x4_t neg = vmulq_f32(alpha, vsubq_f32(exp, _1_0));
                uint32x4_t mask = vcgtq_f32(vdupq_n_f32(0.0f), value);
                return vbslq_f32(mask, neg, value);
            }

            template<int iter> SIMD_INLINE float32x4_t Swish(float32x4_t value) const
            {
                float32x4_t exp = Exp2(vmulq_f32(_k, value));
                return Div<iter>(value, vaddq_f32(_1_0, exp));
            }
        };

        namespace Detail
        {
            SIMD_INLINE float32x4_t Poly5(float32x4_t x, float a, float b, float c, float d, float e, float f)
            {
                float32x4_t p = vdupq_n_f32(f);
                p = vmlaq_f32(vdupq_n_f32(e), x, p);
                p = vmlaq_f32(vdupq_n_f32(d), x, p);
                p = vmlaq_f32(vdupq_n_f32(c), x, p);
                p = vmlaq_f32(vdupq_n_f32(b), x, p);
                p = vmlaq_f32(vdupq_n_f32(a), x, p);
                return p;
            }

            SIMD_INLINE float32x4_t Exp2(float32x4_t x)
            {
                x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(129.00000f)), vdupq_n_f32(-126.99999f));
                int32x4_t ipart = vcvtq_s32_f32(vsubq_f32(x, vdupq_n_f32(0.5f)));
                float32x4_t fpart = vsubq_f32(x, vcvtq_f32_s32(ipart));
                float32x4_t expipart = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ipart, vdupq_n_s32(127)), 23));
                float32x4_t expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return vmulq_f32(expipart, expfpart);
            }

            SIMD_INLINE float32x4_t Log2(float32x4_t x)
            {
                float32x4_t _1 = vdupq_n_f32(1.0f);
                int32x4_t i = vreinterpretq_s32_f32(x);
                float32x4_t e = vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(vandq_s32(i, vdupq_n_s32(0x7F800000)), 23), vdupq_n_s32(127)));
                float32x4_t m = Or(vreinterpretq_f32_s32(vandq_s32(i, vdupq_n_s32(0x007FFFFF))), _1);
                float32x4_t p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return vaddq_f32(vmulq_f32(p, vsubq_f32(m, _1)), e);
            }
        }

        SIMD_INLINE float32x4_t Exponent(float32x4_t value)
        {
            return Detail::Exp2(vmulq_f32(vdupq_n_f32(1.44269504f), value));
        }

        SIMD_INLINE float32x4_t Elu(float32x4_t value, float32x4_t alpha)
        {
            float32x4_t exp = Exponent(value);
            float32x4_t neg = vmulq_f32(alpha, vsubq_f32(exp, vdupq_n_f32(1.0f)));
            uint32x4_t mask = vcgtq_f32(vdupq_n_f32(0.0f), value);
            return vbslq_f32(mask, neg, value);
        }

        SIMD_INLINE float32x4_t Logarithm(float32x4_t value)
        {
            return vmulq_f32(vdupq_n_f32(0.693147181f), Detail::Log2(value));
        }

        template<int iter> SIMD_INLINE float32x4_t Mish(float32x4_t value, float32x4_t threshold)
        {
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t mish = vaddq_f32(Exponent(value), _1);
            mish = Fmadd<true>(mish, mish, _1);
            mish = vmulq_f32(value, vsubq_f32(_1, Div<iter>(vdupq_n_f32(2.0f), mish)));
            return vbslq_f32(vcgtq_f32(threshold, value), mish, value);
        }

        template<int iter> SIMD_INLINE float32x4_t Softplus(float32x4_t value, float32x4_t beta, float32x4_t threshold)
        {
            float32x4_t exp = Exponent(vmulq_f32(value, beta));
            float32x4_t log = Logarithm(vaddq_f32(vdupq_n_f32(1.0f), exp));
            uint32x4_t mask = vcgtq_f32(threshold, value);
            return vbslq_f32(mask, Div<iter>(log, beta), value);
        }

        template<int iter> SIMD_INLINE float32x4_t Swish(float32x4_t value, float32x4_t slope)
        {
            float32x4_t exp = Exponent(vsubq_f32(vdupq_n_f32(0.0f), vmulq_f32(value, slope)));
            return Div<iter>(value, vaddq_f32(vdupq_n_f32(1.0f), exp));
        }

        template<int iter> SIMD_INLINE float32x4_t Tanh(float32x4_t value)
        {
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t exp = Detail::Exp2(vmulq_f32(vdupq_n_f32(2.88539008f), value));
            return Div<iter>(vsubq_f32(exp, _1), vaddq_f32(_1, exp));
        }
    }
#endif //SIMD_NEON_ENABLE
}

#endif//__SimdExp_h__
