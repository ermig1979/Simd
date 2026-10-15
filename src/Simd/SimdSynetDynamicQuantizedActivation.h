/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifndef __SimdSynetDynamicQuantizedActivation_h__
#define __SimdSynetDynamicQuantizedActivation_h__

#include "Simd/SimdSynetActivation.h"

namespace Simd
{
    namespace Base
    {
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float* dst, __m128i sum,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params)
        {
            __m128 val = _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum, corr[index])), norm[index]);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(val, bias[index]), params, index));
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float* dst, __m128i sum,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params, size_t tail)
        {
            SIMD_ALIGNED(16) float tmp[F];
            Save<type, index>(tmp, sum, corr, norm, bias, params);
            for (size_t i = 0; i < tail; ++i)
                dst[i] = tmp[i];
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m128i sum,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params)
        {
            Save<type, 0>(dst, sum, corr, norm, bias, params);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m128i sum,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params, size_t tail)
        {
            Save<type, 0>(dst, sum, corr, norm, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m128i sum0, __m128i sum1,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params)
        {
            Save<type, 0>(dst + 0, sum0, corr, norm, bias, params);
            Save<type, 1>(dst + F, sum1, corr, norm, bias, params);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m128i sum0, __m128i sum1,
            const __m128i* corr, const __m128* norm, const __m128* bias, const __m128* params, size_t tail)
        {
            Save<type, 0>(dst + 0, sum0, corr, norm, bias, params);
            Save<type, 1>(dst + F, sum1, corr, norm, bias, params, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float* dst, __m256i sum,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params)
        {
            __m256 val = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, corr[index])), norm[index]);
            _mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(val, bias[index]), params, index));
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float* dst, __m256i sum,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params, size_t tail)
        {
            SIMD_ALIGNED(32) float tmp[F];
            Save<type, index>(tmp, sum, corr, norm, bias, params);
            for (size_t i = 0; i < tail; ++i)
                dst[i] = tmp[i];
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256i sum,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params)
        {
            Save<type, 0>(dst, sum, corr, norm, bias, params);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256i sum,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params, size_t tail)
        {
            Save<type, 0>(dst, sum, corr, norm, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256i sum0, __m256i sum1,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params)
        {
            Save<type, 0>(dst + 0, sum0, corr, norm, bias, params);
            Save<type, 1>(dst + F, sum1, corr, norm, bias, params);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256i sum0, __m256i sum1,
            const __m256i* corr, const __m256* norm, const __m256* bias, const __m256* params, size_t tail)
        {
            Save<type, 0>(dst + 0, sum0, corr, norm, bias, params);
            Save<type, 1>(dst + F, sum1, corr, norm, bias, params, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float* dst, __m512i sum,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            __m512 val = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(sum, corr[index])), norm[index]);
            _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(val, bias[index]), params, index));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m512i sum,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            Save<type, 0>(dst, sum, corr, norm, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m512i sum0, __m512i sum1,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            Save<type, 0>(dst + 0, sum0, corr, norm, bias, params);
            Save<type, 1>(dst + F, sum1, corr, norm, bias, params, tail);
        }
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
    }
#endif

#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
    }
#endif
}

#endif
