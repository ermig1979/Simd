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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        template <SimdConvolutionActivationType type> __m256i QuantizedAdd(const __m256i& a, const __m256i& aBias, const __m256& aNorm, const __m256i& b, const __m256i& bBias, const __m256& bNorm, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            return QuantizeLinear(Activate<type>(_mm256_add_ps(DequantizeLinear(a, aBias, aNorm), DequantizeLinear(b, bBias, bNorm)), params, 0), dNorm, dZero);
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u1(const uint8_t* a, const __m256i& aBias, const __m256& aNorm, const uint8_t* b, const __m256i& bBias, const __m256& bNorm, const __m256* params, uint8_t* dst, const __m256 & dNorm, const __m256i& dZero)
        {
            __m256i d0 = QuantizedAdd<type>(_mm256_set1_epi32(a[0]), aBias, aNorm, _mm256_set1_epi32(b[0]), bBias, bNorm, params, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u4(const uint8_t* a, const __m256i& aBias, const __m256& aNorm, const uint8_t* b, const __m256i& bBias, const __m256& bNorm, const __m256* params, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m256i a0 = _mm256_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)a)[0]));
            __m256i b0 = _mm256_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)b)[0]));
            __m256i d0 = QuantizedAdd<type>(a0, aBias, aNorm, b0, bBias, bNorm, params, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u16(const uint8_t* a, const __m256i& aBias, const __m256& aNorm, const uint8_t* b, const __m256i& bBias, const __m256& bNorm, const __m256* params, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);   
            __m256i d0 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(_a, 0 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(_b, 0 * 8)), bBias, bNorm, params, dNorm, dZero);
            __m256i d1 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(_a, 1 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(_b, 1 * 8)), bBias, bNorm, params, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
        }

        template <SimdConvolutionActivationType type> void QuantizedAdd8u8u8u32(const uint8_t* a, const __m256i& aBias, const __m256& aNorm, const uint8_t* b, const __m256i& bBias, const __m256& bNorm, const __m256* params, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i a0 = _mm_loadu_si128((__m128i*)a + 0), b0 = _mm_loadu_si128((__m128i*)b + 0);
            __m256i d0 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(a0, 0 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(b0, 0 * 8)), bBias, bNorm, params, dNorm, dZero);
            __m256i d1 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(a0, 1 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(b0, 1 * 8)), bBias, bNorm, params, dNorm, dZero);
            __m128i a1 = _mm_loadu_si128((__m128i*)a + 1), b1 = _mm_loadu_si128((__m128i*)b + 1);
            __m256i d2 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(a1, 0 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(b1, 0 * 8)), bBias, bNorm, params, dNorm, dZero);
            __m256i d3 = QuantizedAdd<type>(_mm256_cvtepu8_epi32(_mm_srli_si128(a1, 1 * 8)), aBias, aNorm, _mm256_cvtepu8_epi32(_mm_srli_si128(b1, 1 * 8)), bBias, bNorm, params, dNorm, dZero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        template <SimdConvolutionActivationType type> static void QuantizedAddUniform8u8u8u(const uint8_t* a, int aBias, float aNorm, const uint8_t* b, int bBias, float bNorm, size_t size, const float* params, float dNorm, int dZero, uint8_t* dst)
        {
            __m256i _aBias = _mm256_set1_epi32(aBias), _bBias = _mm256_set1_epi32(bBias), _dZero = _mm256_set1_epi32(dZero);
            __m256 _aNorm = _mm256_set1_ps(aNorm), _bNorm = _mm256_set1_ps(bNorm), _dNorm = _mm256_set1_ps(dNorm), _params[2];
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size32; i += 32)
                QuantizedAdd8u8u8u32<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
            for (; i < size16; i += 16)
                QuantizedAdd8u8u8u16<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
            for (; i < size4; i += 4)
                QuantizedAdd8u8u8u4<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
            for (; i < size; i += 1)
                QuantizedAdd8u8u8u1<type>(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, _params, dst + i, _dNorm, _dZero);
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
            : Sse41::SynetQuantizedAddUniform(p)
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
