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
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        SIMD_INLINE __m256i QuantizedPrelu(const __m256i& src, const __m256i& sBias, const __m256& sNorm, const __m256& slope, const __m256& dNorm, const __m256i& dZero)
        {
            __m256 _src = DequantizeLinear(src, sBias, sNorm);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _src);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _src);
            __m256 _dst = Fmadd<false>(slope, neg, pos);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedPrelu1(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& slope, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m256i _src = _mm256_set1_epi32(src[0]);
            __m256i d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedPrelu8(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& slope, const uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m256i _src = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)src));
            __m256i d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            _mm_storel_epi64((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedPrelu32(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& slope, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i src0 = _mm_loadu_si128((__m128i*)src + 0);
            __m256i d0 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 0)), sBias, sNorm, slope, dNorm, dZero);
            __m256i d1 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 8)), sBias, sNorm, slope, dNorm, dZero);
            __m128i src1 = _mm_loadu_si128((__m128i*)src + 1);
            __m256i d2 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 0)), sBias, sNorm, slope, dNorm, dZero);
            __m256i d3 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 8)), sBias, sNorm, slope, dNorm, dZero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedPrelu32(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const float* slope, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i src0 = _mm_loadu_si128((__m128i*)src + 0);
            __m256i d0 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 0)), sBias, sNorm, _mm256_loadu_ps(slope + 0 * 8), dNorm, dZero);
            __m256i d1 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 8)), sBias, sNorm, _mm256_loadu_ps(slope + 1 * 8), dNorm, dZero);
            __m128i src1 = _mm_loadu_si128((__m128i*)src + 1);
            __m256i d2 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 0)), sBias, sNorm, _mm256_loadu_ps(slope + 2 * 8), dNorm, dZero);
            __m256i d3 = QuantizedPrelu(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 8)), sBias, sNorm, _mm256_loadu_ps(slope + 3 * 8), dNorm, dZero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void SynetQuantizedPreluLayerForward(const uint8_t* src, const float* srcPrelu, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstPrelu, int dstZero, SimdTensorFormatType format)
        {
            __m256i sBias = _mm256_set1_epi32(-srcZero), dZero = _mm256_set1_epi32(dstZero);
            __m256 sNorm = _mm256_set1_ps(srcPrelu[0]), dNorm = _mm256_set1_ps(1.0f / dstPrelu[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels8 = AlignLo(channels, 8), channels32 = AlignLo(channels, 32);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels32; c += 32)
                        QuantizedPrelu32(src + c, sBias, sNorm, slope + c, dst + c, dNorm, dZero);
                    for (; c < channels8; c += 8)
                        QuantizedPrelu8(src + c, sBias, sNorm, _mm256_load_ps(slope + c), dst + c, dNorm, dZero);
                    for (; c < channels; ++c)
                        QuantizedPrelu1(src + c, sBias, sNorm, _mm256_castps128_ps256(_mm_load_ss(slope + c)), dst + c, dNorm, dZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial8 = AlignLo(spatial, 8), spatial32 = AlignLo(spatial, 32);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _slope = _mm256_set1_ps(slope[c]);
                    size_t s = 0;
                    for (; s < spatial32; s += 32)
                        QuantizedPrelu32(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    for (; s < spatial8; s += 8)
                        QuantizedPrelu8(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    for (; s < spatial; ++s)
                        QuantizedPrelu1(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
