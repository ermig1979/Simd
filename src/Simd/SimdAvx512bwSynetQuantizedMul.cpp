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
#include "Simd/SimdSynetQuantizedMul.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512bw
    {
        SIMD_INLINE __m512i QuantizedMul8u8u8u(const __m512i& a, const __m512i& aBias, const __m512& aNorm,
            const __m512i& b, const __m512i& bBias, const __m512& bNorm, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _a = DequantizeLinear(a, aBias, aNorm);
            __m512 _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(_mm512_mul_ps(_a, _b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul8u8u8u16(const uint8_t* a, const __m512i& aBias, const __m512& aNorm,
            const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i a0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a));
            __m512i b0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b));
            __m512i d0 = QuantizedMul8u8u8u(a0, aBias, aNorm, b0, bBias, bNorm, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedMul8u8u8u64(const uint8_t* a, const __m512i& aBias, const __m512& aNorm,
            const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedMul8u8u8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), aBias, aNorm,
                _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bBias, bNorm, dNorm, dZero);
            __m512i d1 = QuantizedMul8u8u8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), aBias, aNorm,
                _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bBias, bNorm, dNorm, dZero);
            __m512i d2 = QuantizedMul8u8u8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), aBias, aNorm,
                _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bBias, bNorm, dNorm, dZero);
            __m512i d3 = QuantizedMul8u8u8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), aBias, aNorm,
                _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bBias, bNorm, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedMul8u8u8uNN(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm,
            size_t n64, size_t n16, size_t n1, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            size_t i = 0;
            for (; i < n64; i += 64)
                QuantizedMul8u8u8u64(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n16; i += 16)
                QuantizedMul8u8u8u16(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            if (i < n1)
                QuantizedMul8u8u8u16(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero, TailMask16(n1 - n16));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i QuantizedMul32f8u8u(const __m512& a, const __m512i& b, const __m512i& bBias, const __m512& bNorm, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(_mm512_mul_ps(a, _b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul32f8u8u16(const __m512& a, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i b0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b));
            __m512i d0 = QuantizedMul32f8u8u(a, b0, bBias, bNorm, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedMul32f8u8u64(const __m512& a, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedMul32f8u8u(a, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bBias, bNorm, dNorm, dZero);
            __m512i d1 = QuantizedMul32f8u8u(a, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bBias, bNorm, dNorm, dZero);
            __m512i d2 = QuantizedMul32f8u8u(a, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bBias, bNorm, dNorm, dZero);
            __m512i d3 = QuantizedMul32f8u8u(a, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bBias, bNorm, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedMul8u8u8u1N(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm,
            size_t n64, size_t n16, size_t n1, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _a = DequantizeLinear(_mm512_set1_epi32(a[0]), aBias, aNorm);
            size_t i = 0;
            for (; i < n64; i += 64)
                QuantizedMul32f8u8u64(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n16; i += 16)
                QuantizedMul32f8u8u16(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            if (i < n1)
                QuantizedMul32f8u8u16(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero, TailMask16(n1 - n16));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i QuantizedMul8u32f8u(const __m512i& a, const __m512i& aBias, const __m512& aNorm, const __m512& b, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _a = DequantizeLinear(a, aBias, aNorm);
            return QuantizeLinear(_mm512_mul_ps(_a, b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul8u32f8u16(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const __m512& b, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i a0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a));
            __m512i d0 = QuantizedMul8u32f8u(a0, aBias, aNorm, b, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedMul8u32f8u64(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const __m512& b, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedMul8u32f8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), aBias, aNorm, b, dNorm, dZero);
            __m512i d1 = QuantizedMul8u32f8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), aBias, aNorm, b, dNorm, dZero);
            __m512i d2 = QuantizedMul8u32f8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), aBias, aNorm, b, dNorm, dZero);
            __m512i d3 = QuantizedMul8u32f8u(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), aBias, aNorm, b, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedMul8u8u8uN1(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm,
            size_t n64, size_t n16, size_t n1, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _b = DequantizeLinear(_mm512_set1_epi32(b[0]), bBias, bNorm);
            size_t i = 0;
            for (; i < n64; i += 64)
                QuantizedMul8u32f8u64(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero);
            for (; i < n16; i += 16)
                QuantizedMul8u32f8u16(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero);
            if (i < n1)
                QuantizedMul8u32f8u16(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero, TailMask16(n1 - n16));
        }

        //-------------------------------------------------------------------------------------------------

        template <size_t N> void QuantizedMulUniversal8u8u8u(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
            const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero)
        {
            __m512i _aBias = _mm512_set1_epi32(-aZero), _bBias = _mm512_set1_epi32(-bZero), _dZero = _mm512_set1_epi32(dZero);
            __m512 _aScale = _mm512_set1_ps(aScale), _bScale = _mm512_set1_ps(bScale), _scale = _mm512_set1_ps(1.0f / dScale);
            const uint8_t* a0 = (uint8_t*)a8;
            const uint8_t* b0 = (uint8_t*)b8;
            uint8_t* dst = (uint8_t*)dst8;
            size_t n1 = dstShape[N - 1], n16 = AlignLo(n1, 16), n64 = AlignLo(n1, 64);
            bool aN = aSteps[N - 1] == 1, bN = bSteps[N - 1] == 1;
            if (N == 1)
            {
                if (aN && bN)
                    QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                else if (bN)
                    QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                else if (aN)
                    QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
            }
            else if (N == 2)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    if (aN && bN)
                        QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                    else if (bN)
                        QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                    else if (aN)
                        QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                    dst += n1;
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
                        if (aN && bN)
                            QuantizedMul8u8u8uNN(a1, _aBias, _aScale, b1, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                        else if (bN)
                            QuantizedMul8u8u8u1N(a1, _aBias, _aScale, b1, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                        else if (aN)
                            QuantizedMul8u8u8uN1(a1, _aBias, _aScale, b1, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                        dst += n1;
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
                            if (aN && bN)
                                QuantizedMul8u8u8uNN(a2, _aBias, _aScale, b2, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                            else if (bN)
                                QuantizedMul8u8u8u1N(a2, _aBias, _aScale, b2, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                            else if (aN)
                                QuantizedMul8u8u8uN1(a2, _aBias, _aScale, b2, _bBias, _bScale, n64, n16, n1, dst, _scale, _dZero);
                            dst += n1;
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
            : Avx2::SynetQuantizedMulUniversal(p)
        {
            if (p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
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
