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
#include "Simd/SimdSynetQuantizedAdd.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE __m512i QuantizedAdd(const __m512i& a, const __m512& adScale, const __m512i& b, const __m512& bdScale, const __m512& term)
        {
            return _mm512_cvtps_epi32(Fmadd<false>(_mm512_cvtepi32_ps(a), adScale, Fmadd<false>(_mm512_cvtepi32_ps(b), bdScale, term)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u16(const uint8_t* a, const __m512& adScale, const uint8_t* b, const __m512& bdScale, const __m512& term, uint8_t* dst, __mmask16 tail = -1)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a)), adScale, _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b)), bdScale, term);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0), K_ZERO)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u64(const uint8_t* a, const __m512& adScale, const uint8_t* b, const __m512& bdScale, const __m512& term, uint8_t* dst)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bdScale, term);
            __m512i d1 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bdScale, term);
            __m512i d2 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bdScale, term);
            __m512i d3 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bdScale, term);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        static void QuantizedAddUniform8u8u8u(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
            __m512 _adScale = _mm512_set1_ps(adScale), _bdScale = _mm512_set1_ps(bdScale), _term = _mm512_set1_ps(term);
            size_t i = 0, size16 = AlignLo(size, 16), size64 = AlignLo(size, 64);
            __mmask16 tail = TailMask16(size - size16);
            for (; i < size64; i += 64)
                QuantizedAdd8u8u8u64(a + i, _adScale, b + i, _bdScale, _term, dst + i);
            for (; i < size16; i += 16)
                QuantizedAdd8u8u8u16(a + i, _adScale, b + i, _bdScale, _term, dst + i);
            if(i < size)
                QuantizedAdd8u8u8u16(a + i, _adScale, b + i, _bdScale, _term, dst + i, tail);
        }

        static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform8u8u8u(SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case SimdConvolutionActivationIdentity:
            case SimdConvolutionActivationRelu: return QuantizedAddUniform8u8u8u;
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedAddUniform::SynetQuantizedAddUniform(const QuantizedAddParam& p)
            : Avx2::SynetQuantizedAddUniform(p)
        {
            if(p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
                _uniform = GetQuantizedAddUniform8u8u8u(p.actType);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero)
        {
            QuantizedAddParam param(aShape, aCount, aType, aScale, aZero, bShape, bCount, bType, bScale, bZero, actType, actParams, dstType, dstScale, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedAddUniform::Preferable(param))
                return new SynetQuantizedAddUniform(param);
            return NULL;
        }
    }
#endif
}
