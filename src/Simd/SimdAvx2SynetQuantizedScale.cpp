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
        SIMD_INLINE __m256i QuantizedScale(const __m256i& src, const __m256i& sBias, const __m256& sNorm, const __m256& scale, const __m256& bias, const __m256& dNorm, const __m256i& dZero)
        {
            __m256 _src = DequantizeLinear(src, sBias, sNorm);
            __m256 _dst = Fmadd<false>(_src, scale, bias);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedScale1(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& scale, const __m256& bias, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m256i _src = _mm256_set1_epi32(src[0]);
            __m256i d0 = QuantizedScale(_src, sBias, sNorm, scale, bias, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedScale8(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& scale, const __m256& bias, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m256i _src = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)src));
            __m256i d0 = QuantizedScale(_src, sBias, sNorm, scale, bias, dNorm, dZero);
            _mm_storel_epi64((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedScale32(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const __m256& scale, const __m256& bias, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i src0 = _mm_loadu_si128((__m128i*)src + 0);
            __m256i d0 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 0)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m256i d1 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 8)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m128i src1 = _mm_loadu_si128((__m128i*)src + 1);
            __m256i d2 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 0)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m256i d3 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 8)), sBias, sNorm, scale, bias, dNorm, dZero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedScale32(const uint8_t* src, const __m256i& sBias, const __m256& sNorm, const float* scale, const float* bias, uint8_t* dst, const __m256& dNorm, const __m256i& dZero)
        {
            __m128i src0 = _mm_loadu_si128((__m128i*)src + 0);
            __m256i d0 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 0)), sBias, sNorm, _mm256_loadu_ps(scale + 0 * 8), _mm256_loadu_ps(bias + 0 * 8), dNorm, dZero);
            __m256i d1 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src0, 8)), sBias, sNorm, _mm256_loadu_ps(scale + 1 * 8), _mm256_loadu_ps(bias + 1 * 8), dNorm, dZero);
            __m128i src1 = _mm_loadu_si128((__m128i*)src + 1);
            __m256i d2 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 0)), sBias, sNorm, _mm256_loadu_ps(scale + 2 * 8), _mm256_loadu_ps(bias + 2 * 8), dNorm, dZero);
            __m256i d3 = QuantizedScale(_mm256_cvtepu8_epi32(_mm_srli_si128(src1, 8)), sBias, sNorm, _mm256_loadu_ps(scale + 3 * 8), _mm256_loadu_ps(bias + 3 * 8), dNorm, dZero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void SynetQuantizedScaleLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            Array32f defaultBias;
            if (bias == NULL)
            {
                defaultBias.Resize(channels, true);
                bias = defaultBias.data;
            }
            __m256i sBias = _mm256_set1_epi32(-srcZero), dZero = _mm256_set1_epi32(dstZero);
            __m256 sNorm = _mm256_set1_ps(srcScale[0]), dNorm = _mm256_set1_ps(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels8 = AlignLo(channels, 8), channels32 = AlignLo(channels, 32);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels32; c += 32)
                        QuantizedScale32(src + c, sBias, sNorm, scale + c, bias + c, dst + c, dNorm, dZero);
                    for (; c < channels8; c += 8)
                        QuantizedScale8(src + c, sBias, sNorm, _mm256_load_ps(scale + c), _mm256_load_ps(bias + c), dst + c, dNorm, dZero);
                    for (; c < channels; ++c)
                        QuantizedScale1(src + c, sBias, sNorm, _mm256_castps128_ps256(_mm_load_ss(scale + c)), _mm256_castps128_ps256(_mm_load_ss(bias + c)), dst + c, dNorm, dZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial8 = AlignLo(spatial, 8), spatial32 = AlignLo(spatial, 32);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _bias = _mm256_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatial32; s += 32)
                        QuantizedScale32(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    for (; s < spatial8; s += 8)
                        QuantizedScale8(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    for (; s < spatial; ++s)
                        QuantizedScale1(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
