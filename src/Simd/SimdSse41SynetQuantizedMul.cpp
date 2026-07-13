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
#include "Simd/SimdSynetQuantizedMul.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        SIMD_INLINE __m128i QuantizedMul8u8u8u(const __m128i& a, const __m128i& aBias, const __m128& aNorm, 
            const __m128i& b, const __m128i& bBias, const __m128& bNorm, const __m128& dNorm, const __m128i& dZero)
        {
            __m128 _a = DequantizeLinear(a, aBias, aNorm);
            __m128 _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(_mm_mul_ps(_a, _b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul8u8u8u1(const uint8_t* a, const __m128i& aBias, const __m128& aNorm, 
            const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i a0 = _mm_set1_epi32(a[0]);
            __m128i b0 = _mm_set1_epi32(b[0]);
            __m128i d0 = QuantizedMul8u8u8u(a0, aBias, aNorm, b0, bBias, bNorm, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul8u8u8u4(const uint8_t* a, const __m128i& aBias, const __m128& aNorm,
            const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i a0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)a)[0]));
            __m128i b0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)b)[0]));
            __m128i d0 = QuantizedMul8u8u8u(a0, aBias, aNorm, b0, bBias, bNorm, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul8u8u8u16(const uint8_t* a, const __m128i& aBias, const __m128& aNorm,
            const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m128i d0 = QuantizedMul8u8u8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 0 * 4)), aBias, aNorm, 
                _mm_cvtepu8_epi32(_mm_srli_si128(_b, 0 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d1 = QuantizedMul8u8u8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 1 * 4)), aBias, aNorm, 
                _mm_cvtepu8_epi32(_mm_srli_si128(_b, 1 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d2 = QuantizedMul8u8u8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 2 * 4)), aBias, aNorm, 
                _mm_cvtepu8_epi32(_mm_srli_si128(_b, 2 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d3 = QuantizedMul8u8u8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 3 * 4)), aBias, aNorm, 
                _mm_cvtepu8_epi32(_mm_srli_si128(_b, 3 * 4)), bBias, bNorm, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i QuantizedMul32f8u8u(const __m128& a, const __m128i& b, const __m128i& bBias, const __m128& bNorm, const __m128& dNorm, const __m128i& dZero)
        {
            __m128 _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(_mm_mul_ps(a, _b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul32f8u8u1(const __m128& a, const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i b0 = _mm_set1_epi32(b[0]);
            __m128i d0 = QuantizedMul32f8u8u(a, b0, bBias, bNorm, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul32f8u8u4(const __m128& a, const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i b0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)b)[0]));
            __m128i d0 = QuantizedMul32f8u8u(a, b0, bBias, bNorm, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul32f8u8u16(const __m128& a, const uint8_t* b, const __m128i& bBias, const __m128& bNorm, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m128i d0 = QuantizedMul32f8u8u(a, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 0 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d1 = QuantizedMul32f8u8u(a, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 1 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d2 = QuantizedMul32f8u8u(a, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 2 * 4)), bBias, bNorm, dNorm, dZero);
            __m128i d3 = QuantizedMul32f8u8u(a, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 3 * 4)), bBias, bNorm, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i QuantizedMul8u32f8u(const __m128i& a, const __m128i& aBias, const __m128& aNorm, const __m128& b, const __m128& dNorm, const __m128i& dZero)
        {
            __m128 _a = DequantizeLinear(a, aBias, aNorm);
            return QuantizeLinear(_mm_mul_ps(_a, b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul8u32f8u1(const uint8_t* a, const __m128i& aBias, const __m128& aNorm, const __m128& b, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i a0 = _mm_set1_epi32(a[0]);
            __m128i d0 = QuantizedMul8u32f8u(a0, aBias, aNorm, b, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul8u32f8u4(const uint8_t* a, const __m128i& aBias, const __m128& aNorm, const __m128& b, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i a0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)a)[0]));
            __m128i d0 = QuantizedMul8u32f8u(a0, aBias, aNorm, b, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedMul8u32f8u16(const uint8_t* a, const __m128i& aBias, const __m128& aNorm, const __m128& b, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i d0 = QuantizedMul8u32f8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 0 * 4)), aBias, aNorm, b, dNorm, dZero);
            __m128i d1 = QuantizedMul8u32f8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 1 * 4)), aBias, aNorm, b, dNorm, dZero);
            __m128i d2 = QuantizedMul8u32f8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 2 * 4)), aBias, aNorm, b, dNorm, dZero);
            __m128i d3 = QuantizedMul8u32f8u(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 3 * 4)), aBias, aNorm, b, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        //-------------------------------------------------------------------------------------------------

        template <size_t N> void QuantizedMulUniversal8u8u8u(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
            const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero)
        {
            __m128i _aBias = _mm_set1_epi32(-aZero), _bBias = _mm_set1_epi32(-bZero), _dZero = _mm_set1_epi32(dZero);
            __m128 _aScale = _mm_set1_ps(aScale), _bScale = _mm_set1_ps(bScale), _scale = _mm_set1_ps(1.0f / dScale);
            const uint8_t* a0 = (uint8_t*)a8;
            const uint8_t* b0 = (uint8_t*)b8;
            uint8_t* dst = (uint8_t*)dst8;
            if (N == 1)
            {
                size_t n0 = dstShape[0], n04 = AlignLo(n0, 4), n016 = AlignLo(n0, 16), i0 = 0;
                if (aSteps[0] == 1 && bSteps[0] == 1)
                {
                    for (; i0 < n016; i0 += 16)
                        QuantizedMul8u8u8u16(a0 + i0, _aBias, _aScale, b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                    for (; i0 < n04; i0 += 4)
                        QuantizedMul8u8u8u4(a0 + i0, _aBias, _aScale, b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                    for (; i0 < n0; i0 += 1)
                        QuantizedMul8u8u8u1(a0 + i0, _aBias, _aScale, b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                }
                else if (aSteps[0] == 0)
                {
                    __m128 _a = DequantizeLinear(_mm_set1_epi32(a0[0]), _aBias, _aScale);
                    for (; i0 < n016; i0 += 16)
                        QuantizedMul32f8u8u16(_a,  b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                    for (; i0 < n04; i0 += 4)
                        QuantizedMul32f8u8u4(_a, b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                    for (; i0 < n0; i0 += 1)
                        QuantizedMul32f8u8u1(_a, b0 + i0, _bBias, _bScale, dst + i0, _scale, _dZero);
                }
                else if (bSteps[0] == 0)
                {
                    __m128 _b = DequantizeLinear(_mm_set1_epi32(b0[0]), _bBias, _bScale);
                    for (; i0 < n016; i0 += 16)
                        QuantizedMul8u32f8u16(a0 + i0, _aBias, _aScale, _b, dst + i0, _scale, _dZero);
                    for (; i0 < n04; i0 += 4)
                        QuantizedMul8u32f8u4(a0 + i0, _aBias, _aScale, _b, dst + i0, _scale, _dZero);
                    for (; i0 < n0; i0 += 1)
                        QuantizedMul8u32f8u1(a0 + i0, _aBias, _aScale, _b, dst + i0, _scale, _dZero);
                }
                else
                {
                    for (size_t i0 = 0; i0 < n0; ++i0)
                    {
                        QuantizedMul8u8u8u1(a0, _aBias, _aScale, b0, _bBias, _bScale, dst, _scale, _dZero);
                        a0 += aSteps[0];
                        b0 += bSteps[0];
                        dst += 1;
                    }
                }
            }
            else if (N == 2)
            {
                size_t n0 = dstShape[0], n1 = dstShape[1], n14 = AlignLo(n1, 4), n116 = AlignLo(n1, 16);
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const uint8_t* a1 = a0;
                    const uint8_t* b1 = b0;
                    size_t i1 = 0;
                    if (aSteps[1] == 1 && bSteps[1] == 1)
                    {
                        for (; i1 < n116; i1 += 16)
                            QuantizedMul8u8u8u16(a1 + i1, _aBias, _aScale, b1 + i1, _bBias, _bScale, dst + i1, _scale, _dZero);
                        for (; i1 < n14; i1 += 4)
                            QuantizedMul8u8u8u4(a1 + i1, _aBias, _aScale, b1 + i1, _bBias, _bScale, dst + i1, _scale, _dZero);
                        for (; i1 < n1; i1 += 1)
                            QuantizedMul8u8u8u1(a1 + i1, _aBias, _aScale, b1 + i1, _bBias, _bScale, dst + i1, _scale, _dZero);
                        dst += n1;
                    }
                    else
                    {
                        for (; i1 < dstShape[1]; ++i1)
                        {
                            QuantizedMul8u8u8u1(a1, _aBias, _aScale, b1, _bBias, _bScale, dst, _scale, _dZero);
                            a1 += aSteps[1];
                            b1 += bSteps[1];
                            dst += 1;
                        }
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else if (N == 3)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const uint8_t* a1 = a0;
                    const uint8_t* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        const uint8_t* a2 = a1;
                        const uint8_t* b2 = b1;
                        for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                        {
                            QuantizedMul8u8u8u1(a2, _aBias, _aScale, b2, _bBias, _bScale, dst, _scale, _dZero);
                            a2 += aSteps[2];
                            b2 += bSteps[2];
                            dst += 1;
                        }
                        a1 += aSteps[1];
                        b1 += bSteps[1];
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else if (N == 4)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const uint8_t* a1 = a0;
                    const uint8_t* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        const uint8_t* a2 = a1;
                        const uint8_t* b2 = b1;
                        for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                        {
                            const uint8_t* a3 = a2;
                            const uint8_t* b3 = b2;
                            for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                            {
                                QuantizedMul8u8u8u1(a3, _aBias, _aScale, b3, _bBias, _bScale, dst, _scale, _dZero);
                                a3 += aSteps[3];
                                b3 += bSteps[3];
                                dst += 1;
                            }
                            a2 += aSteps[2];
                            b2 += bSteps[2];
                        }
                        a1 += aSteps[1];
                        b1 += bSteps[1];
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else
                assert(0);
        }


        //static void QuantizedAddUniform8u8u8u(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        //{
        //    float adScale = aScale / dScale;
        //    float bdScale = bScale / dScale;
        //    float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
        //    __m128 _adScale = _mm_set1_ps(adScale), _bdScale = _mm_set1_ps(bdScale), _term = _mm_set1_ps(term);
        //    size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
        //    for (; i < size16; i += 16)
        //        QuantizedAdd8u8u8u16(a + i, _adScale, b + i, _bdScale, _term, dst + i);
        //    for (; i < size4; i += 4)
        //        QuantizedAdd8u8u8u4(a + i, _adScale, b + i, _bdScale, _term, dst + i);
        //    for (; i < size; i += 1)
        //        QuantizedAdd8u8u8u1(a + i, _adScale, b + i, _bdScale, _term, dst + i);
        //}

        //static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform8u8u8u(SimdConvolutionActivationType type)
        //{
        //    switch (type)
        //    {
        //    case SimdConvolutionActivationIdentity:
        //    case SimdConvolutionActivationRelu: return QuantizedAddUniform8u8u8u;
        //    default:
        //        return NULL;
        //    }
        //}

        //-------------------------------------------------------------------------------------------------

        static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal8u8u8u(size_t dim)
        {
            switch (dim)
            {
            case 1: return QuantizedMulUniversal8u8u8u<1>;
            case 2: return QuantizedMulUniversal8u8u8u<2>;
            case 3: return QuantizedMulUniversal8u8u8u<3>;
            case 4: return QuantizedMulUniversal8u8u8u<4>;
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMulUniversal::SynetQuantizedMulUniversal(const QuantizedMulParam& p)
            : Base::SynetQuantizedMulUniversal(p)
        {
            if(p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
                _universal = GetQuantizedMulUniversal8u8u8u(_dShape.size());
        }

        ////-------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero)
        {
            QuantizedMulParam param(aShape, aCount, aType, aScale, aZero, bShape, bCount, bType, bScale, bZero, dstType, dstScale, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedMulUniversal::Preferable(param))
                return new SynetQuantizedMulUniversal(param);
            return NULL;
        }
    }
#endif
}
