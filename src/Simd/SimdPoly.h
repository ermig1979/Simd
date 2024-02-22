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
#ifndef __SimdPoly_h__
#define __SimdPoly_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
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
    }
#endif   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
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
    }
#endif  

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
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
