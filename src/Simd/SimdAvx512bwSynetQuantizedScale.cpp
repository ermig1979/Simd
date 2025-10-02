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
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE __m512i QuantizedScale(const __m512i& src, const __m512i& sBias, const __m512& sNorm, const __m512& scale, const __m512& bias, const __m512& dNorm, const __m512i& dZero)
        {
            __m512 _src = DequantizeLinear(src, sBias, sNorm);
            __m512 _dst = Fmadd<false>(_src, scale, bias);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedScale16(const uint8_t* src, const __m512i& sBias, const __m512& sNorm, const __m512& scale, const __m512& bias, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i _src = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src));
            __m512i d0 = QuantizedScale(_src, sBias, sNorm, scale, bias, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedScale64(const uint8_t* src, const __m512i& sBias, const __m512& sNorm, const __m512& scale, const __m512& bias, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 0)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m512i d1 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 1)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m512i d2 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 2)), sBias, sNorm, scale, bias, dNorm, dZero);
            __m512i d3 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 3)), sBias, sNorm, scale, bias, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE void QuantizedScale64(const uint8_t* src, const __m512i& sBias, const __m512& sNorm, const float* scale, const float* bias, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 0)), sBias, sNorm, _mm512_loadu_ps(scale + 0 * F), _mm512_loadu_ps(bias + 0 * F), dNorm, dZero);
            __m512i d1 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 1)), sBias, sNorm, _mm512_loadu_ps(scale + 1 * F), _mm512_loadu_ps(bias + 1 * F), dNorm, dZero);
            __m512i d2 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 2)), sBias, sNorm, _mm512_loadu_ps(scale + 2 * F), _mm512_loadu_ps(bias + 2 * F), dNorm, dZero);
            __m512i d3 = QuantizedScale(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)src + 3)), sBias, sNorm, _mm512_loadu_ps(scale + 3 * F), _mm512_loadu_ps(bias + 3 * F), dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void SynetQuantizedScaleLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            Array32f defaultBias;
            if (bias == NULL)
            {
                defaultBias.Resize(channels, true);
                bias = defaultBias.data;
            }
            __m512i sBias = _mm512_set1_epi32(-srcZero), dZero = _mm512_set1_epi32(dstZero);
            __m512 sNorm = _mm512_set1_ps(srcScale[0]), dNorm = _mm512_set1_ps(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels16 = AlignLo(channels, 16), channels64 = AlignLo(channels, 64);
                __mmask16 tail = TailMask16(channels - channels16);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels64; c += 64)
                        QuantizedScale64(src + c, sBias, sNorm, scale + c, bias + c, dst + c, dNorm, dZero);
                    for (; c < channels16; c += 16)
                        QuantizedScale16(src + c, sBias, sNorm, _mm512_load_ps(scale + c), _mm512_load_ps(bias + c), dst + c, dNorm, dZero);
                    if(tail)
                        QuantizedScale16(src + c, sBias, sNorm, _mm512_maskz_load_ps(tail, scale + c), _mm512_maskz_load_ps(tail, bias + c), dst + c, dNorm, dZero, tail);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial16 = AlignLo(spatial, 16), spatial64 = AlignLo(spatial, 64);
                __mmask16 tail = TailMask16(spatial - spatial16);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _bias = _mm512_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatial64; s += 64)
                        QuantizedScale64(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    for (; s < spatial16; s += 16)
                        QuantizedScale16(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero);
                    if (tail)
                        QuantizedScale16(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero, tail);
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
