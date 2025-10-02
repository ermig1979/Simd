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
        SIMD_INLINE __m128i QuantizedPrelu(const __m128i& src, const __m128i& sBias, const __m128& sNorm, const __m128& slope, const __m128& dNorm, const __m128i& dZero)
        {
            __m128 _src = DequantizeLinear(src, sBias, sNorm);
            __m128 pos = _mm_max_ps(_mm_setzero_ps(), _src);
            __m128 neg = _mm_min_ps(_mm_setzero_ps(), _src);
            __m128 _dst = Fmadd<false>(slope, neg, pos);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedPrelu1(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& slope, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_set1_epi32(src[0]);
            __m128i d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedPrelu4(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& slope, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)src)[0]));
            __m128i d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedPrelu16(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const __m128& slope, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i d0 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), sBias, sNorm, slope, dNorm, dZero);
            __m128i d1 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), sBias, sNorm, slope, dNorm, dZero);
            __m128i d2 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), sBias, sNorm, slope, dNorm, dZero);
            __m128i d3 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), sBias, sNorm, slope, dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        SIMD_INLINE void QuantizedPrelu16(const uint8_t* src, const __m128i& sBias, const __m128& sNorm, const float* slope, uint8_t* dst, const __m128& dNorm, const __m128i& dZero)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i d0 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), sBias, sNorm, _mm_loadu_ps(slope + 0 * 4), dNorm, dZero);
            __m128i d1 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), sBias, sNorm, _mm_loadu_ps(slope + 1 * 4), dNorm, dZero);
            __m128i d2 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), sBias, sNorm, _mm_loadu_ps(slope + 2 * 4), dNorm, dZero);
            __m128i d3 = QuantizedPrelu(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), sBias, sNorm, _mm_loadu_ps(slope + 3 * 4), dNorm, dZero);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        void SynetQuantizedPreluLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            __m128i sBias = _mm_set1_epi32(-srcZero), dZero = _mm_set1_epi32(dstZero);
            __m128 sNorm = _mm_set1_ps(srcScale[0]), dNorm = _mm_set1_ps(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels4 = AlignLo(channels, 4), channels16 = AlignLo(channels, 16);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels16; c += 16)
                        QuantizedPrelu16(src + c, sBias, sNorm, slope + c, dst + c, dNorm, dZero);
                    for (; c < channels4; c += 4)
                        QuantizedPrelu4(src + c, sBias, sNorm, _mm_load_ps(slope + c), dst + c, dNorm, dZero);
                    for (; c < channels; ++c)
                        QuantizedPrelu1(src + c, sBias, sNorm, _mm_load_ss(slope + c), dst + c, dNorm, dZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _slope = _mm_set1_ps(slope[c]);
                    size_t s = 0;
                    for (; s < spatial16; s += 16)
                        QuantizedPrelu16(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    for (; s < spatial4; s += 4)
                        QuantizedPrelu4(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
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
