/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdTrigonometric_h__
#define __SimdTrigonometric_h__

#include "Simd/SimdPoly.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE float Sin32f(float value)
        {
            return ::sinf(value);
        }

        SIMD_INLINE float Cos32f(float value)
        {
            return ::cosf(value);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        namespace Detail
        {
            SIMD_INLINE __m128 NormSin(__m128 x)
            {
                static const __m128 _1 = _mm_set1_ps(1.0f);
                __m128 p = Poly5(_mm_mul_ps(x, x), -3.1415926444234477f, 2.0261194642649887f, -0.5240361513980939f, 0.0751872634325299f, -0.006860187425683514f, 0.000385937753182769f);
                return _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(x, _1), _mm_add_ps(x, _1)), _mm_mul_ps(p, x));
            }
        }

        SIMD_INLINE __m128 Sin(__m128 x)
        {
            static const __m128 _1_pi = _mm_set1_ps(float(M_1_PI));
            x = _mm_mul_ps(_1_pi, x);
            __m128 f = _mm_floor_ps(x);
            __m128 s = Detail::NormSin(_mm_sub_ps(x, f));
            __m128i n = _mm_slli_epi32(_mm_cvtps_epi32(f), 31);
            return _mm_or_ps(s, _mm_castsi128_ps(n));
        }

        SIMD_INLINE __m128 Cos(__m128 x)
        {
            static const __m128 _pi_2 = _mm_set1_ps(float(M_PI_2));
            return Sin(_mm_sub_ps(_pi_2, x));
        }
    }
#endif   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        namespace Detail
        {
            SIMD_INLINE __m256 NormSin(__m256 x)
            {
                static const __m256 _1 = _mm256_set1_ps(1.0f);
                __m256 p = Poly5(_mm256_mul_ps(x, x), -3.1415926444234477f, 2.0261194642649887f, -0.5240361513980939f, 0.0751872634325299f, -0.006860187425683514f, 0.000385937753182769f);
                return _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(x, _1), _mm256_add_ps(x, _1)), _mm256_mul_ps(p, x));
            }
        }

        SIMD_INLINE __m256 Sin(__m256 x)
        {
            static const __m256 _1_pi = _mm256_set1_ps(float(M_1_PI));
            x = _mm256_mul_ps(_1_pi, x);
            __m256 f = _mm256_floor_ps(x);
            __m256 s = Detail::NormSin(_mm256_sub_ps(x, f));
            __m256i n = _mm256_slli_epi32(_mm256_cvtps_epi32(f), 31);
            return _mm256_or_ps(s, _mm256_castsi256_ps(n));
        }

        SIMD_INLINE __m256 Cos(__m256 x)
        {
            static const __m256 _pi_2 = _mm256_set1_ps(float(M_PI_2));
            return Sin(_mm256_sub_ps(_pi_2, x));
        }
    }
#endif  

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        namespace Detail
        {
            SIMD_INLINE __m512 NormSin(__m512 x)
            {
                static const __m512 _1 = _mm512_set1_ps(1.0f);
                __m512 p = Poly5(_mm512_mul_ps(x, x), -3.1415926444234477f, 2.0261194642649887f, -0.5240361513980939f, 0.0751872634325299f, -0.006860187451283514f, 0.000385937753182769f);
                return _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(x, _1), _mm512_add_ps(x, _1)), _mm512_mul_ps(p, x));
            }
        }

        SIMD_INLINE __m512 Sin(__m512 x)
        {
            static const __m512 _1_pi = _mm512_set1_ps(float(M_1_PI));
            x = _mm512_mul_ps(_1_pi, x);
            __m512 f = _mm512_floor_ps(x);
            __m512 s = Detail::NormSin(_mm512_sub_ps(x, f));
            __m512i n = _mm512_slli_epi32(_mm512_cvtps_epi32(f), 31);
            return _mm512_or_ps(s, _mm512_castsi512_ps(n));
        }

        SIMD_INLINE __m512 Cos(__m512 x)
        {
            static const __m512 _pi_2 = _mm512_set1_ps(float(M_PI_2));
            return Sin(_mm512_sub_ps(_pi_2, x));
        }
    }
#endif 

#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {

    }
#endif 
}

#endif
