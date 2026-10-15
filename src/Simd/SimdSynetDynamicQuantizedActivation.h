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
        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(float* dst, const int32_t *sum,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            __m512 val = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(sum), corr[index])), norm[index]);
            _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(val, bias[index]), params, index));
        }

        template<SimdConvolutionActivationType type, int M> SIMD_INLINE void ApplyMx1(float* dst, const int32_t* sum,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            if (M > 1)
            {
                Apply<type, 0>(dst + 0, sum + 0, corr, norm, bias, params);
                Apply<type, 1>(dst + F, sum + F, corr, norm, bias, params, tail);
            }
            else
                Apply<type, 0>(dst + 0, sum + 0, corr, norm, bias, params, tail);
        }

        template<SimdConvolutionActivationType type, int M, int N> static SIMD_INLINE void ApplyMxN(float* dst, size_t dD, const int32_t* sum,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, __mmask16 tail = -1)
        {
            if (N > 0) ApplyMx1<type, M>(dst + 0 * dD, sum + 0 * DF, corr, norm, bias, params, tail);
            if (N > 1) ApplyMx1<type, M>(dst + 1 * dD, sum + 1 * DF, corr, norm, bias, params, tail);
            if (N > 2) ApplyMx1<type, M>(dst + 2 * dD, sum + 2 * DF, corr, norm, bias, params, tail);
            if (N > 3) ApplyMx1<type, M>(dst + 3 * dD, sum + 3 * DF, corr, norm, bias, params, tail);
            if (N > 4) ApplyMx1<type, M>(dst + 4 * dD, sum + 4 * DF, corr, norm, bias, params, tail);
            if (N > 5) ApplyMx1<type, M>(dst + 5 * dD, sum + 5 * DF, corr, norm, bias, params, tail);
            if (N > 6) ApplyMx1<type, M>(dst + 6 * dD, sum + 6 * DF, corr, norm, bias, params, tail);
            if (N > 7) ApplyMx1<type, M>(dst + 7 * dD, sum + 7 * DF, corr, norm, bias, params, tail);
        }
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
