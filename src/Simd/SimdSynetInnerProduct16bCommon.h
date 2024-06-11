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
#ifndef __SimdSynetInnerProduct16bCommon_h__
#define __SimdSynetInnerProduct16bCommon_h__

#include "Simd/SimdStore.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void Save1(float* ptr, __m128 value)
        {
            _mm_storeu_ps(ptr, value);
        }

        SIMD_INLINE void Save1(float* ptr, __m128 value, size_t tail)
        {
            SIMD_ALIGNED(16) float tmp[F];
            _mm_storeu_ps(tmp, value);
            for (size_t i = 0; i < tail; ++i)
                ptr[i] = tmp[i];
        }

        SIMD_INLINE void Save2(float* ptr, __m128 val0, __m128 val1)
        {
            _mm_storeu_ps(ptr + 0, val0);
            _mm_storeu_ps(ptr + F, val1);
        }

        SIMD_INLINE void Save2(float* ptr, __m128 val0, __m128 val1, size_t tail)
        {
            _mm_storeu_ps(ptr + 0, val0);
            Save1(ptr + F, val1, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void Save1(float* ptr, __m256 value)
        {
            _mm256_storeu_ps(ptr, value);
        }

        SIMD_INLINE void Save1(float* ptr, __m256 value, size_t tail)
        {
            SIMD_ALIGNED(32) float tmp[F];
            _mm256_storeu_ps(tmp, value);
            for (size_t i = 0; i < tail; ++i)
                ptr[i] = tmp[i];
        }

        SIMD_INLINE void Save2(float* ptr, __m256 val0, __m256 val1)
        {
            _mm256_storeu_ps(ptr + 0, val0);
            _mm256_storeu_ps(ptr + F, val1);
        }

        SIMD_INLINE void Save2(float* ptr, __m256 val0, __m256 val1, size_t tail)
        {
            _mm256_storeu_ps(ptr + 0, val0);
            Save1(ptr + F, val1, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE void Save1(float* ptr, __m512 value, __mmask16 tail = __mmask16(-1))
        {
            _mm512_mask_storeu_ps(ptr, tail, value);
        }

        SIMD_INLINE void Save2(float* ptr, __m512 val0, __m512 val1, __mmask16 tail = __mmask16(-1))
        {
            _mm512_storeu_ps(ptr + 0, val0);
            Save1(ptr + F, val1, tail);
        }
    }
#endif

}

#endif
