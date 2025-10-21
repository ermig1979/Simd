/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdSynetQuantizedDepthwise_h__
#define __SimdSynetQuantizedDepthwise_h__

#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"

namespace Simd
{
    namespace Base
    {

    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const int32_t* bias, const float* norm, const __m128i& zero, size_t offset)
        {
            __m128i _bias = _mm_loadu_si128((__m128i*)(bias + offset));
            __m128 _norm = _mm_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const int32_t* bias, const float* norm, const __m128i& zero, size_t offset, size_t tail)
        {
            __m128i _bias = _mm_loadu_si128((__m128i*)(bias + offset));
            __m128 _norm = _mm_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const __m128i& bias, const __m128& norm, const __m128i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const __m128i& bias, const __m128& norm, const __m128i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save2(uint8_t* dst0, uint8_t* dst1, __m128i sum0, __m128i sum1, const int32_t* bias, const float* norm, const __m128i& zero, size_t offset)
        {
            __m128i _bias = _mm_loadu_si128((__m128i*)(bias + offset));
            __m128 _norm = _mm_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst0 + offset, (int32_t*)NULL, sum0, &_bias, &_norm, zero);
            QuntizedTerm8i<term>::template Save<0>(dst1 + offset, (int32_t*)NULL, sum1, &_bias, &_norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save2(uint8_t* dst0, uint8_t* dst1, __m128i sum0, __m128i sum1, const int32_t* bias, const float* norm, const __m128i& zero, size_t offset, size_t tail)
        {
            __m128i _bias = _mm_loadu_si128((__m128i*)(bias + offset));
            __m128 _norm = _mm_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst0 + offset, (int32_t*)NULL, sum0, &_bias, &_norm, zero, tail);
            QuntizedTerm8i<term>::template Save<0>(dst1 + offset, (int32_t*)NULL, sum1, &_bias, &_norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const __m128i& sBias, 
            const __m128& sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero)
        {
            Save<term, type, 0>(dst, (int32_t*)NULL, sum, &sBias, &sNorm, iLo, iHi, iScale, params, dNorm, dZero);
        }

        template <Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const __m128i& sBias, 
            const __m128& sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, (int32_t*)NULL, sum, &sBias, &sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const int32_t* bias, const float* norm, const __m256i& zero, size_t offset)
        {
            __m256i _bias = _mm256_loadu_si256((__m256i*)(bias + offset));
            __m256 _norm = _mm256_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const int32_t* bias, const float* norm, const __m256i& zero, size_t offset, size_t tail)
        {
            __m256i _bias = _mm256_loadu_si256((__m256i*)(bias + offset));
            __m256 _norm = _mm256_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save2(uint8_t* dst0, uint8_t* dst1, __m256i sum0, __m256i sum1, const int32_t* bias, const float* norm, const __m256i& zero, size_t offset)
        {
            __m256i _bias = _mm256_loadu_si256((__m256i*)(bias + offset));
            __m256 _norm = _mm256_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst0 + offset, (int32_t*)NULL, sum0, &_bias, &_norm, zero);
            QuntizedTerm8i<term>::template Save<0>(dst1 + offset, (int32_t*)NULL, sum1, &_bias, &_norm, zero);
        }

        template <Term8iType term> SIMD_INLINE void Save2(uint8_t* dst0, uint8_t* dst1, __m256i sum0, __m256i sum1, const int32_t* bias, const float* norm, const __m256i& zero, size_t offset, size_t tail)
        {
            __m256i _bias = _mm256_loadu_si256((__m256i*)(bias + offset));
            __m256 _norm = _mm256_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst0 + offset, (int32_t*)NULL, sum0, &_bias, &_norm, zero, tail);
            QuntizedTerm8i<term>::template Save<0>(dst1 + offset, (int32_t*)NULL, sum1, &_bias, &_norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& sBias,
            const __m256& sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            Save<term, type, 0>(dst, (int32_t*)NULL, sum, &sBias, &sNorm, iLo, iHi, iScale, params, dNorm, dZero);
        }

        template <Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& sBias,
            const __m256& sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, (int32_t*)NULL, sum, &sBias, &sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m512i sum, const int32_t* bias, const float* norm, const __m512i& zero, size_t offset, __mmask16 tail = -1)
        {
            __m512i _bias = _mm512_maskz_loadu_epi32(tail, bias + offset);
            __m512 _norm = _mm512_maskz_loadu_ps(tail, norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m512i sum, const __m512i& bias, const __m512& norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        template <Term8iType term> SIMD_INLINE void Save2(uint8_t* dst0, uint8_t* dst1, __m512i sum0, __m512i sum1, const int32_t* bias, const float* norm, const __m512i& zero, size_t offset, __mmask16 tail = -1)
        {
            __m512i _bias = _mm512_loadu_si512((__m512i*)(bias + offset));
            __m512 _norm = _mm512_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst0 + offset, (int32_t*)NULL, sum0, &_bias, &_norm, zero, tail);
            QuntizedTerm8i<term>::template Save<0>(dst1 + offset, (int32_t*)NULL, sum1, &_bias, &_norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, __m512i sum, const __m512i& sBias,
            const __m512& sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Save<term, type, 0>(dst, (int32_t*)NULL, sum, &sBias, &sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif
}

#endif
