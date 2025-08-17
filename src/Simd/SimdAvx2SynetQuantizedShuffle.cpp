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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cd = 0, spatial8  = AlignLo(spatial, 8), spatial32 = AlignLo(spatial, 32), s;
            __m256i _bias0 = _mm256_set1_epi32(bias0), _bias1 = _mm256_set1_epi32(bias1), _zero = _mm256_set1_epi32(zero);
            __m256 _norm0 = _mm256_set1_ps(norm0), _norm1 = _mm256_set1_ps(norm1), _scale = _mm256_set1_ps(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }        
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_1(const uint8_t* src, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m256i d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_set1_epi32(((int16_t*)src)[0])), bias, norm), scale, zero);
            __m256i u0 = _mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO);
            dst0[0] = _mm256_extract_epi8(u0, 0);
            dst1[0] = _mm256_extract_epi8(u0, 1);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_8(const uint8_t* src, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m256i d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(_src, 0)), bias, norm), scale, zero);
            __m256i d1 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(_src, 8)), bias, norm), scale, zero);
            __m256i u0 = Deinterleave8To64(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO));
#if defined(SIMD_X64_ENABLE)
            ((uint64_t*)dst0)[0] = _mm256_extract_epi64(u0, 0);
            ((uint64_t*)dst1)[0] = _mm256_extract_epi64(u0, 1);
#else
            SIMD_ALIGNED(32) uint64_t tmp[4];
            _mm256_store_si256((__m256i*)tmp, u0);
            ((uint64_t*)dst0)[0] = tmp[0];
            ((uint64_t*)dst1)[0] = tmp[1];
#endif
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_32(const uint8_t* src, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst0, uint8_t* dst1)
        {
            __m256i d0, d1, d2, d3, u0, u1;
            __m128i s0;
            s0 = _mm_loadu_si128((__m128i*)src + 0);
            d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            s0 = _mm_loadu_si128((__m128i*)src + 1);
            d2 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            u0 = Deinterleave8To64(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            s0 = _mm_loadu_si128((__m128i*)src + 2);
            d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            s0 = _mm_loadu_si128((__m128i*)src + 3);
            d2 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            u1 = Deinterleave8To64(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            _mm256_storeu_si256((__m256i*)dst0, Deinterleave64<0>(u0, u1));
            _mm256_storeu_si256((__m256i*)dst1, Deinterleave64<1>(u0, u1));
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs, cd, srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16), srcC0_64 = AlignLo(srcC0, 64), srcC1_64 = AlignLo(srcC1, 64);
            __m256i _bias0 = _mm256_set1_epi32(bias0), _bias1 = _mm256_set1_epi32(bias1), _zero = _mm256_set1_epi32(zero);
            __m256 _norm0 = _mm256_set1_ps(norm0), _norm1 = _mm256_set1_ps(norm1), _scale = _mm256_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                cd = 0, cs = 0;
                for (; cs < srcC0_64; cs += 64, cd += 32)
                    DequantizeQuantizeLinearNhwc0_32(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0; cs += 2, cd += 1)
                    DequantizeQuantizeLinearNhwc0_1(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                cs = 0;
                for (; cs < srcC1_64; cs += 64, cd += 32)
                    DequantizeQuantizeLinearNhwc0_32(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
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
            size_t dstC = (srcC0 + srcC1) / 2, cs = 0, spatial8 = AlignLo(spatial, 8), spatial32 = AlignLo(spatial, 32), s;
            __m256i _bias0 = _mm256_set1_epi32(bias0), _bias1 = _mm256_set1_epi32(bias1), _zero = _mm256_set1_epi32(zero);
            __m256 _norm0 = _mm256_set1_ps(norm0), _norm1 = _mm256_set1_ps(norm1), _scale = _mm256_set1_ps(scale);
            for (size_t cd = 0; cd < srcC0; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
                for (s = 0; s < spatial32; s += 32)
                    DequantizeQuantizeLinear32(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial8; s += 8)
                    DequantizeQuantizeLinear8(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_1(const uint8_t* src0, const uint8_t* src1, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst)
        {
            __m128i s0 = _mm_set1_epi8(src0[0]);
            __m128i s1 = _mm_set1_epi8(src1[0]);
            __m128i s01 = _mm_unpacklo_epi8(s0, s1);
            __m256i d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(s01), bias, norm), scale, zero);
            __m256i u0 = _mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO);
            ((uint16_t*)dst)[0] = _mm256_cvtsi256_si32(u0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_8(const uint8_t* src0, const uint8_t* src1, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst)
        {
            __m128i _src0 = _mm_loadl_epi64((__m128i*)src0);
            __m128i _src1 = _mm_loadl_epi64((__m128i*)src1);
            __m128i s0 = _mm_unpacklo_epi8(_src0, _src1);
            __m256i d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            __m256i d1 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            __m256i u0 = PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO);
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(u0));
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_16(const uint8_t* src0, const uint8_t* src1, const __m256i& bias, const __m256& norm, const __m256& scale, const __m256i& zero, uint8_t* dst)
        {
            __m128i _src0, _src1, s0;
            __m256i d0, d1, d2, d3;
            _src0 = _mm_loadu_si128((__m128i*)src0 + 0);
            _src1 = _mm_loadu_si128((__m128i*)src1 + 0);
            s0 = _mm_unpacklo_epi8(_src0, _src1);
            d0 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            s0 = _mm_unpackhi_epi8(_src0, _src1);
            d2 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)), bias, norm), scale, zero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, 
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16), srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            __m256i _bias01 = SetInt32(bias0, bias1), _zero = _mm256_set1_epi32(zero);
            __m256 _norm01 = SetFloat(norm0, norm1), _scale = _mm256_set1_ps(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cs = 0, cd = 0;
                for (; cd < srcC0_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0; cd += 2, cs += 1)
                    DequantizeQuantizeLinearNhwc1_1(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                cd = 0;
                for (; cd < srcC1_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
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
