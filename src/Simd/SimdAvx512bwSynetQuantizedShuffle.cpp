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
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, spatial16  = AlignLo(spatial, 16), spatial64 = AlignLo(spatial, 64), s;
            __mmask16 tail = TailMask16(spatial - spatial16);
            __m512i _bias0 = _mm512_set1_epi32(bias0), _bias1 = _mm512_set1_epi32(bias1), _zero = _mm512_set1_epi32(zero);
            __m512 _norm0 = _mm512_set1_ps(norm0), _norm1 = _mm512_set1_ps(norm1), _scale = _mm512_set1_ps(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2)
            {
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s, tail);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s, tail);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2)
            {
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s, tail);
                src1 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s, tail);
                src1 += spatial;
                dst1 += spatial;
            }        
        }

        //--------------------------------------------------------------------------------------------------

        const __m512i K8_DEINTERLEAVE_8 = SIMD_MM512_SETR_EPI8(
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_16(const uint8_t* src, const __m512i& bias, const __m512& norm, const __m512& scale, const __m512i& zero, uint8_t* dst0, uint8_t* dst1, __mmask16 tail = -1)
        {
            __m256i _src = _mm256_maskz_loadu_epi16(tail, (int16_t*)src);
            __m512i d0 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(_src, 0)), bias, norm), scale, zero);
            __m512i d1 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(_src, 1)), bias, norm), scale, zero);
            __m512i u0 = _mm512_permutex_epi64(_mm512_shuffle_epi8(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO), K8_DEINTERLEAVE_8), 0xD8);
            _mm_mask_storeu_epi8(dst0, tail, _mm512_extracti32x4_epi32(u0, 0));
            _mm_mask_storeu_epi8(dst1, tail, _mm512_extracti32x4_epi32(u0, 1));
        }

        const __m512i K64_PERMUTE_16_0 = SIMD_MM512_SETR_EPI64(0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE);
        const __m512i K64_PERMUTE_16_1 = SIMD_MM512_SETR_EPI64(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_64(const uint8_t* src, const __m512i& bias, const __m512& norm, const __m512& scale, const __m512i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m512i d0, d1, d2, d3;
            d0 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 0)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 1)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 2)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 3)), bias, norm), scale, zero);
            __m512i u0 = _mm512_shuffle_epi8(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)), K8_DEINTERLEAVE_8);
            d0 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 4)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 5)), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 6)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 7)), bias, norm), scale, zero);
            __m512i u1 = _mm512_shuffle_epi8(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)), K8_DEINTERLEAVE_8);
            _mm512_storeu_si512((__m512i*)dst0, _mm512_permutex2var_epi64(u0, K64_PERMUTE_16_0, u1));
            _mm512_storeu_si512((__m512i*)dst1, _mm512_permutex2var_epi64(u0, K64_PERMUTE_16_1, u1));
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs, cd, srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            size_t srcC0_128 = AlignLo(srcC0, 128), srcC1_128 = AlignLo(srcC1, 128), lastD0 = (srcC0 - srcC0_32) / 2, lastD1 = (srcC1 - srcC1_32) / 2;
            __mmask16 tail0 = TailMask16(lastD0), tail1 = TailMask16(lastD1);
            __m512i _bias0 = _mm512_set1_epi32(bias0), _bias1 = _mm512_set1_epi32(bias1), _zero = _mm512_set1_epi32(zero);
            __m512 _norm0 = _mm512_set1_ps(norm0), _norm1 = _mm512_set1_ps(norm1), _scale = _mm512_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                cd = 0, cs = 0;
                for (; cs < srcC0_128; cs += 128, cd += 64)
                    DequantizeQuantizeLinearNhwc0_64(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                if(cs < srcC0)
                    DequantizeQuantizeLinearNhwc0_16(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd, tail0), cd += lastD0;
                cs = 0;
                for (; cs < srcC1_128; cs += 128, cd += 64)
                    DequantizeQuantizeLinearNhwc0_64(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                if (cs < srcC1)
                    DequantizeQuantizeLinearNhwc0_16(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd, tail1), cd += lastD1;
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
            size_t dstC = (srcC0 + srcC1) / 2, spatial16 = AlignLo(spatial, 16), spatial64 = AlignLo(spatial, 64), s;
            __mmask16 tail = TailMask16(spatial - spatial16);
            __m512i _bias0 = _mm512_set1_epi32(bias0), _bias1 = _mm512_set1_epi32(bias1), _zero = _mm512_set1_epi32(zero);
            __m512 _norm0 = _mm512_set1_ps(norm0), _norm1 = _mm512_set1_ps(norm1), _scale = _mm512_set1_ps(scale);
            for (size_t cd = 0; cd < srcC0; cd += 2)
            {
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s, tail);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s, tail);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cd += 2)
            {
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s, tail);
                src0 += spatial;
                dst1 += spatial;
                for (s = 0; s < spatial64; s += 64)
                    DequantizeQuantizeLinear64(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                if (s < spatial)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s, tail);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_16(const uint8_t* src0, const uint8_t* src1, const __m512i& bias, const __m512& norm, const __m512& scale, const __m512i& zero, uint8_t* dst, __mmask16 tail = -1)
        {
            __m128i _src0 = _mm_maskz_loadu_epi8(tail, src0);
            __m128i _src1 = _mm_maskz_loadu_epi8(tail, src1);
            __m512i d0 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(_src0, _src1)), bias, norm), scale, zero);
            __m512i d1 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(_src0, _src1)), bias, norm), scale, zero);
            __m512i u0 = PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO);
            _mm256_mask_storeu_epi16((int16_t*)dst, tail, _mm512_castsi512_si256(u0));
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_32(const uint8_t* src0, const uint8_t* src1, const __m512i& bias, const __m512& norm, const __m512& scale, const __m512i& zero, uint8_t* dst)
        {
            __m128i _src0, _src1, s0;
            __m512i d0, d1, d2, d3;
            _src0 = _mm_loadu_si128((__m128i*)src0 + 0);
            _src1 = _mm_loadu_si128((__m128i*)src1 + 0);
            d0 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(_src0, _src1)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(_src0, _src1)), bias, norm), scale, zero);
            _src0 = _mm_loadu_si128((__m128i*)src0 + 1);
            _src1 = _mm_loadu_si128((__m128i*)src1 + 1);
            d2 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(_src0, _src1)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(_src0, _src1)), bias, norm), scale, zero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            size_t srcC0_64 = AlignLo(srcC0, 64), srcC1_64 = AlignLo(srcC1, 64), lastC0 = (srcC0 - srcC0_32) / 2, lastC1 = (srcC1 - srcC1_32) / 2;
            __mmask16 tail0 = TailMask16(lastC0), tail1 = TailMask16(lastC1);
            __m512i _bias01 = SetInt32(bias0, bias1), _zero = _mm512_set1_epi32(zero);
            __m512 _norm01 = SetFloat(norm0, norm1), _scale = _mm512_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cs = 0, cd = 0;
                for (; cd < srcC0_64; cd += 64, cs += 32)
                    DequantizeQuantizeLinearNhwc1_32(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                if(cd < srcC0)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd, tail0), cs += lastC0;
                cd = 0;
                for (; cd < srcC1_64; cd += 64, cs += 32)
                    DequantizeQuantizeLinearNhwc1_32(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                if(cd < srcC1)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd, tail1), cs += lastC1;
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
