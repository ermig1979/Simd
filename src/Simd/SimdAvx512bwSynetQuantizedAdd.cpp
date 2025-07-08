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
#include "Simd/SimdSynetQuantizedAddCommon.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        template <SimdConvolutionActivationType type> __m512i QuantizedAdd(const __m512i& a, const __m512i& aBias, const __m512& aNorm, const __m512i& b, const __m512i& bBias, const __m512& bNorm, const __m512* params, const __m512& dNorm, const __m512i& dZero)
        {
            return QuantizeLinear(Activate<type>(_mm512_add_ps(DequantizeLinear(a, aBias, aNorm), DequantizeLinear(b, bBias, bNorm)), params, 0), dNorm, dZero);
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u16(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, const __m512* params, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i d0 = QuantizedAdd<type>(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b)), bBias, bNorm, params, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0), K_ZERO)));
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u64(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, const __m512* params, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedAdd<type>(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bBias, bNorm, params, dNorm, dZero);
            __m512i d1 = QuantizedAdd<type>(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bBias, bNorm, params, dNorm, dZero);
            __m512i d2 = QuantizedAdd<type>(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bBias, bNorm, params, dNorm, dZero);
            __m512i d3 = QuantizedAdd<type>(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bBias, bNorm, params, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        template <SimdConvolutionActivationType type> static void QuantizedAddUniform8u8u8u(const uint8_t* a, int aBias, float aNorm, const uint8_t* b, int bBias, float bNorm, size_t size, const float* params, float dNorm, int dZero, uint8_t* dst)
        {
            __m512i _aBias = _mm512_set1_epi32(aBias), _bBias = _mm512_set1_epi32(bBias), _dZero = _mm512_set1_epi32(dZero);
            __m512 _aNorm = _mm512_set1_ps(aNorm), _bNorm = _mm512_set1_ps(bNorm), _dNorm = _mm512_set1_ps(dNorm), _params[2];
            size_t i = 0, size16 = AlignLo(size, 16), size64 = AlignLo(size, 64);
            __mmask16 tail = TailMask16(size - size16);
            for (; i < size64; i += 64)
                QuantizedAdd8u8u8u64<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
            for (; i < size16; i += 16)
                QuantizedAdd8u8u8u16<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
            if(i < size)
                QuantizedAdd8u8u8u16<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero, tail);
        }

        static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform8u8u8u(SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case SimdConvolutionActivationIdentity: return QuantizedAddUniform8u8u8u<SimdConvolutionActivationIdentity>;
            case SimdConvolutionActivationRelu: return QuantizedAddUniform8u8u8u<SimdConvolutionActivationRelu>;
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, SimdConvolutionActivationType type, typename D> static void QuantizedAddUniform(const uint8_t* a8, int aBias, float aNorm, const uint8_t* b8, int bBias, float bNorm, size_t size, const float* params, float dNorm, int dZero, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            for (size_t i = 0; i < size; ++i)
                Base::QuantizedAdd<A, B, type, D>(a[i], aBias, aNorm, b[i], bBias, bNorm, params, dst[i], dNorm, dZero);
        }

        template<class A, class B, SimdConvolutionActivationType type> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return QuantizedAddUniform<A, B, type, float>;
            case SimdTensorData8u: return QuantizedAddUniform<A, B, type, uint8_t>;
            default:
                return NULL;
            }
        }

        template<class A, class B> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (type)
            {
            case SimdConvolutionActivationIdentity: return GetQuantizedAddUniform<A, B, SimdConvolutionActivationIdentity>(dType);
            case SimdConvolutionActivationRelu: return GetQuantizedAddUniform<A, B, SimdConvolutionActivationRelu>(dType);
            default:
                return NULL;
            }
        }

        template<class A> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType bType, SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetQuantizedAddUniform<A, float>(type, dType);
            case SimdTensorData8u: return GetQuantizedAddUniform<A, uint8_t>(type, dType);
            default:
                return NULL;
            }
        }

        static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType aType, SimdTensorDataType bType, SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetQuantizedAddUniform<float>(bType, type, dType);
            case SimdTensorData8u: return GetQuantizedAddUniform<uint8_t>(bType, type, dType);
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
            else
                _uniform = GetQuantizedAddUniform(p.aType, p.bType, p.actType, p.dType);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, int32_t aBias, const float* aNorm,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, int32_t bBias, const float* bNorm,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstNorm, int32_t dstZero)
        {
            QuantizedAddParam param(aShape, aCount, aType, aBias, aNorm, bShape, bCount, bType, bBias, bNorm, actType, actParams, dstType, dstNorm, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedAddUniform::Preferable(param))
                return new SynetQuantizedAddUniform(param);
            return NULL;
        }
    }
#endif
}
