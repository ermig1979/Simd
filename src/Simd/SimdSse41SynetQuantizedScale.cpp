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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        SIMD_INLINE __m128i QuantizedScale(const __m128i& src, const __m128i& sBias, const __m128& sNorm, const __m128& scale, const __m128& bias, const __m128& dNorm, const __m128i& dZero)
        {
            __m128 _src = DequantizeLinear(src, sBias, sNorm);
            __m128 _dst = Fmadd<false>(_src, scale, bias);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedScale1(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& scale, const __m128& bias, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_set1_epi32(src[0]);
            __m128i d0 = QuantizedScale(_src, sBias, sNorm, scale, bias, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedScale4(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& scale, const __m128& bias, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)src)[0]));
            __m128i d0 = QuantizedScale(_src, sBias, sNorm, scale, bias, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedScale16(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& scale, const __m128& bias, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i d0 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m128i d1 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m128i d2 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m128i d3 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), sBias, sNorm, scale, bias, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        SIMD_INLINE void QuantizedScale16(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const float* scale, const float* bias, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i d0 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), sBias, sNorm, _mm_loadu_ps(scale + 0 * 4), _mm_loadu_ps(bias + 0 * 4), dNorm, dZero);
            __m128i d1 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), sBias, sNorm, _mm_loadu_ps(scale + 1 * 4), _mm_loadu_ps(bias + 1 * 4), dNorm, dZero);
            __m128i d2 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), sBias, sNorm, _mm_loadu_ps(scale + 2 * 4), _mm_loadu_ps(bias + 2 * 4), dNorm, dZero);
            __m128i d3 = QuantizedScale(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), sBias, sNorm, _mm_loadu_ps(scale + 3 * 4), _mm_loadu_ps(bias + 3 * 4), dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        void SynetQuantizedScaleLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            Array32f defaultBias;
            if (bias == NULL)
            {
                defaultBias.Resize(channels, true);
                bias = defaultBias.data;
            }
            __m128i sBias = _mm_set1_epi32(-srcZero), dZero = _mm_set1_epi32(dstZero);
            __m128 sNorm = _mm_set1_ps(srcScale[0]), dNorm = _mm_set1_ps(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels4 = AlignLo(channels, 4), channels16 = AlignLo(channels, 16);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels16; c += 16)
                        QuantizedScale16(src + c, sBias, sNorm, scale + c, bias + c, dst + c, dNorm, dZero);
                    for (; c < channels4; c += 4)
                        QuantizedScale4(src + c, sBias, sNorm, _mm_load_ps(scale + c), _mm_load_ps(bias + c), dst + c, dNorm, dZero);
                    for (; c < channels; ++c)
                        QuantizedScale1(src + c, sBias, sNorm, _mm_load_ss(scale + c), _mm_load_ss(bias + c), dst + c, dNorm, dZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _bias = _mm_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatial16; s += 16)
                        QuantizedScale16(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    for (; s < spatial4; s += 4)
                        QuantizedScale4(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
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
