/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdPow_h__
#define __SimdPow_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE float Pow(float basis, float exponent)
        {
            return ::expf(::logf(basis)*exponent);
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class Pow
        {
            __m128i _exponent, _mantissa;
            __m128 _one;

            SIMD_INLINE __m128 Poly5(__m128 x, float a, float b, float c, float d, float e, float f) const
            {
                __m128 p = _mm_set1_ps(f);
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(e));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(d));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(c));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(b));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x) const
            {
                x = _mm_max_ps(_mm_min_ps(x, _mm_set1_ps(129.00000f)), _mm_set1_ps(-126.99999f));
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));
                __m128 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m128 Log2(__m128 x) const
            {
                __m128i i = _mm_castps_si128(x);
                __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(i, _exponent), 23), _mm_set1_epi32(127)));
                __m128 m = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, _mantissa)), _one);
                __m128 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm_add_ps(_mm_mul_ps(p, _mm_sub_ps(m, _one)), e);
            }

        public:

            SIMD_INLINE Pow()
            {
                _exponent = _mm_set1_epi32(0x7F800000);
                _mantissa = _mm_set1_epi32(0x007FFFFF);
                _one = _mm_set1_ps(1.0f);
            }

            SIMD_INLINE __m128 operator() (__m128 basis, __m128 exponent) const 
            {
                return Exp2(_mm_mul_ps(Log2(basis), exponent));
            }            
        };
    }
#endif //SIMD_SSE2_ENABLE   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class Pow
        {
            __m256i _exponent, _mantissa;
            __m256 _one;

            SIMD_INLINE __m256 Poly5(__m256 x, float a, float b, float c, float d, float e, float f) const
            {
                __m256 p = _mm256_set1_ps(f);
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(e));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(d));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(c));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(b));
                p = _mm256_fmadd_ps(x, p, _mm256_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m256 Exp2(__m256 x) const
            {
                x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(129.00000f)), _mm256_set1_ps(-126.99999f));
                __m256i ipart = _mm256_cvtps_epi32(_mm256_sub_ps(x, _mm256_set1_ps(0.5f)));
                __m256 fpart = _mm256_sub_ps(x, _mm256_cvtepi32_ps(ipart));
                __m256 expipart = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ipart, _mm256_set1_epi32(127)), 23));
                __m256 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm256_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m256 Log2(__m256 x) const
            {
                __m256i i = _mm256_castps_si256(x);
                __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(i, _exponent), 23), _mm256_set1_epi32(127)));
                __m256 m = _mm256_or_ps(_mm256_castsi256_ps(_mm256_and_si256(i, _mantissa)), _one);
                __m256 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm256_fmadd_ps(p, _mm256_sub_ps(m, _one), e);
            }

        public:

            SIMD_INLINE Pow()
            {
                _exponent = _mm256_set1_epi32(0x7F800000);
                _mantissa = _mm256_set1_epi32(0x007FFFFF);
                _one = _mm256_set1_ps(1.0f);
            }

            SIMD_INLINE __m256 operator()(__m256 basis, __m256 exponent) const
            {
                return Exp2(_mm256_mul_ps(Log2(basis), exponent));
            }
        };
    }
#endif //SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class Pow
        {
            __m512i _exponent, _mantissa;
            __m512 _one;

            SIMD_INLINE __m512 Poly5(__m512 x, float a, float b, float c, float d, float e, float f) const
            {
                __m512 p = _mm512_set1_ps(f);
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(e));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(d));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(c));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(b));
                p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m512 Exp2(__m512 x) const
            {
                x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(129.00000f)), _mm512_set1_ps(-126.99999f));
                __m512i ipart = _mm512_cvtps_epi32(_mm512_sub_ps(x, _mm512_set1_ps(0.5f)));
                __m512 fpart = _mm512_sub_ps(x, _mm512_cvtepi32_ps(ipart));
                __m512 expipart = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ipart, _mm512_set1_epi32(127)), 23));
                __m512 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm512_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m512 Log2(__m512 x) const
            {
                __m512i i = _mm512_castps_si512(x);
                __m512 e = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_srli_epi32(_mm512_and_si512(i, _exponent), 23), _mm512_set1_epi32(127)));
                __m512 m = Or(_mm512_castsi512_ps(_mm512_and_si512(i, _mantissa)), _one);
                __m512 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm512_fmadd_ps(p, _mm512_sub_ps(m, _one), e);
            }

        public:

            SIMD_INLINE Pow()
            {
                _exponent = _mm512_set1_epi32(0x7F800000);
                _mantissa = _mm512_set1_epi32(0x007FFFFF);
                _one = _mm512_set1_ps(1.0f);
            }

            SIMD_INLINE __m512 operator()(__m512 basis, __m512 exponent) const
            {
                return Exp2(_mm512_mul_ps(Log2(basis), exponent));
            }
        };
    }
#endif //SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class Pow
        {
            int32x4_t _exponent, _mantissa;
            float32x4_t _one;

            SIMD_INLINE float32x4_t Poly5(float32x4_t x, float a, float b, float c, float d, float e, float f) const
            {
                float32x4_t p = vdupq_n_f32(f);
                p = vmlaq_f32(vdupq_n_f32(e), x, p);
                p = vmlaq_f32(vdupq_n_f32(d), x, p);
                p = vmlaq_f32(vdupq_n_f32(c), x, p);
                p = vmlaq_f32(vdupq_n_f32(b), x, p);
                p = vmlaq_f32(vdupq_n_f32(a), x, p);
                return p;
            }

            SIMD_INLINE float32x4_t Exp2(float32x4_t x) const
            {
                x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(129.00000f)), vdupq_n_f32(-126.99999f));
                int32x4_t ipart = vcvtq_s32_f32(vsubq_f32(x, vdupq_n_f32(0.5f)));
                float32x4_t fpart = vsubq_f32(x, vcvtq_f32_s32(ipart));
                float32x4_t expipart = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ipart, vdupq_n_s32(127)), 23));
                float32x4_t expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return vmulq_f32(expipart, expfpart);
            }

            SIMD_INLINE float32x4_t Log2(float32x4_t x) const
            {
                int32x4_t i = vreinterpretq_s32_f32(x);
                float32x4_t e = vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(vandq_s32(i, _exponent), 23), vdupq_n_s32(127)));
                float32x4_t m = Or(vreinterpretq_f32_s32(vandq_s32(i, _mantissa)), _one);
                float32x4_t p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return vaddq_f32(vmulq_f32(p, vsubq_f32(m, _one)), e);
            }

        public:

            SIMD_INLINE Pow()
            {
                _exponent = vdupq_n_s32(0x7F800000);
                _mantissa = vdupq_n_s32(0x007FFFFF);
                _one = vdupq_n_f32(1.0f);
            }

            SIMD_INLINE float32x4_t operator() (float32x4_t basis, float32x4_t exponent) const
            {
                return Exp2(vmulq_f32(Log2(basis), exponent));
            }
        };
    }
#endif //SIMD_NEON_ENABLE 
}

#endif//__SimdPow_h__
