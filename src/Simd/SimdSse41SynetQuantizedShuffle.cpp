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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdDeinterleave.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cd = 0, spatial4  = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            __m128i _bias0 = _mm_set1_epi32(bias0), _bias1 = _mm_set1_epi32(bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm0 = _mm_set1_ps(norm0), _norm1 = _mm_set1_ps(norm1), _scale = _mm_set1_ps(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }        
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_1(const uint8_t* src, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_set1_epi32(((int16_t*)src)[0])), bias, norm), scale, zero);
            __m128i u0 = _mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO);
            dst0[0] = _mm_extract_epi8(u0, 0);
            dst1[0] = _mm_extract_epi8(u0, 1);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_4(const uint8_t* src, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)src);
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), bias, norm), scale, zero);
            __m128i d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), bias, norm), scale, zero);
            __m128i u0 = Deinterleave8To64(_mm_packus_epi16(_mm_packs_epi32(d0, d1), K_ZERO));
            ((uint32_t*)dst0)[0] = _mm_extract_epi32(u0, 0);
            ((uint32_t*)dst1)[0] = _mm_extract_epi32(u0, 2);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_8(const uint8_t* src, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), bias, norm), scale, zero);
            __m128i d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), bias, norm), scale, zero);
            __m128i d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), bias, norm), scale, zero);
            __m128i d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), bias, norm), scale, zero);
            __m128i u0 = Deinterleave8To64(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            StoreHalf<0>((__m128i*)dst0, u0);
            StoreHalf<1>((__m128i*)dst1, u0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_16(const uint8_t* src, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m128i _src, d0, d1, d2, d3;
            _src = _mm_loadu_si128((__m128i*)src + 0);
            d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), bias, norm), scale, zero);
            __m128i u0 = Deinterleave8To64(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            _src = _mm_loadu_si128((__m128i*)src + 1);
            d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), bias, norm), scale, zero);
            __m128i u1 = Deinterleave8To64(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            _mm_storeu_si128((__m128i*)dst0, Deinterleave64<0>(u0, u1));
            _mm_storeu_si128((__m128i*)dst1, Deinterleave64<1>(u0, u1));
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs, cd, srcC0_8 = AlignLo(srcC0, 8), srcC1_8 = AlignLo(srcC1, 8);
            size_t srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16), srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            __m128i _bias0 = _mm_set1_epi32(bias0), _bias1 = _mm_set1_epi32(bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm0 = _mm_set1_ps(norm0), _norm1 = _mm_set1_ps(norm1), _scale = _mm_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                cd = 0, cs = 0;
                for (; cs < srcC0_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_8; cs += 8, cd += 4)
                    DequantizeQuantizeLinearNhwc0_4(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0; cs += 2, cd += 1)
                    DequantizeQuantizeLinearNhwc0_1(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                cs = 0;
                for (; cs < srcC1_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_8; cs += 8, cd += 4)
                    DequantizeQuantizeLinearNhwc0_4(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1; cs += 2, cd += 1)
                    DequantizeQuantizeLinearNhwc0_1(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                src0 += srcC0;
                src1 += srcC1;
                dst0 += dstC;
                dst1 += dstC;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForwardNchw1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs = 0, spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            __m128i _bias0 = _mm_set1_epi32(bias0), _bias1 = _mm_set1_epi32(bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm0 = _mm_set1_ps(norm0), _norm1 = _mm_set1_ps(norm1), _scale = _mm_set1_ps(scale);
            for (size_t cd = 0; cd < srcC0; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_1(const uint8_t* src0, const uint8_t* src1, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst)
        {
            __m128i s0 = _mm_set1_epi8(src0[0]);
            __m128i s1 = _mm_set1_epi8(src1[0]);
            __m128i s01 = _mm_unpacklo_epi8(s0, s1);
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(s01), bias, norm), scale, zero);
            __m128i u0 = _mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO);
            ((uint16_t*)dst)[0] = _mm_cvtsi128_si32(u0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_4(const uint8_t* src0, const uint8_t* src1, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst)
        {
            __m128i _src0 = _mm_set1_epi32(((int32_t*)src0)[0]);
            __m128i _src1 = _mm_set1_epi32(((int32_t*)src1)[0]);
            __m128i s0 = _mm_unpacklo_epi8(_src0, _src1);
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 0 * 4)), bias, norm), scale, zero);
            __m128i d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 1 * 4)), bias, norm), scale, zero);
            __m128i u0 = _mm_packus_epi16(_mm_packs_epi32(d0, d1), K_ZERO);
            _mm_storel_epi64((__m128i*)dst, u0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_8(const uint8_t* src0, const uint8_t* src1, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst)
        {
            __m128i _src0 = _mm_loadl_epi64((__m128i*)src0);
            __m128i _src1 = _mm_loadl_epi64((__m128i*)src1);
            __m128i s0 = _mm_unpacklo_epi8(_src0, _src1);
            __m128i d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 0 * 4)), bias, norm), scale, zero);
            __m128i d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 1 * 4)), bias, norm), scale, zero);
            __m128i d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 2 * 4)), bias, norm), scale, zero);
            __m128i d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 3 * 4)), bias, norm), scale, zero);
            __m128i u0 = _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3));
            _mm_storeu_si128((__m128i*)dst, u0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_16(const uint8_t* src0, const uint8_t* src1, const __m128i& bias, const __m128& norm, const __m128& scale, const __m128i& zero, uint8_t* dst)
        {
            __m128i _src0 = _mm_loadu_si128((__m128i*)src0);
            __m128i _src1 = _mm_loadu_si128((__m128i*)src1);
            __m128i d0, d1, d2, d3, s0;
            s0 = _mm_unpacklo_epi8(_src0, _src1);
            d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 0 * 4)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 1 * 4)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 2 * 4)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 3 * 4)), bias, norm), scale, zero);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            s0 = _mm_unpackhi_epi8(_src0, _src1);
            d0 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 0 * 4)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 1 * 4)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 2 * 4)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(s0, 3 * 4)), bias, norm), scale, zero);
            _mm_storeu_si128((__m128i*)dst + 1, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, srcC0_8 = AlignLo(srcC0, 8), srcC1_8 = AlignLo(srcC1, 8);
            size_t srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16), srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            __m128i _bias01 = SetInt32(bias0, bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm01 = SetFloat(norm0, norm1), _scale = _mm_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cs = 0, cd;
                for (cd = 0; cd < srcC0_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_8; cd += 8, cs += 4)
                    DequantizeQuantizeLinearNhwc1_4(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0; cd += 2, cs += 1)
                    DequantizeQuantizeLinearNhwc1_1(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (cd = 0; cd < srcC1_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_8; cd += 8, cs += 4)
                    DequantizeQuantizeLinearNhwc1_4(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1; cd += 2, cs += 1)
                    DequantizeQuantizeLinearNhwc1_1(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                src0 += dstC;
                src1 += dstC;
                dst0 += srcC0;
                dst1 += srcC1;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForward(const uint8_t* src0, int bias0, const float* norm0, size_t srcC0, const uint8_t* src1, int bias1, const float* norm1, size_t srcC1,
            size_t spatial, uint8_t* dst0, uint8_t* dst1, const float* scale, int zero, SimdTensorFormatType format, int shuffleType)
        {
            switch (shuffleType)
            {
            case 0:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            case 1:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            }
        }
    }
#endif
}
