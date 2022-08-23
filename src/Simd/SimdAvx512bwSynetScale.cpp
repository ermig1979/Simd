/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx512bw
    {
        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            __m512 _bias = Load<align, mask>(bias + offset, tail);
            Store<align, mask>(dst + offset, Fmadd<nofma>(_src, _scale, _bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            Store<align, mask>(dst + offset, _mm512_mul_ps(_src, _scale), tail);
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m512& scale, const __m512& bias, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            Store<align, mask>(dst + offset, Fmadd<nofma>(_src, scale, bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m512& scale, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            Store<align, mask>(dst + offset, _mm512_mul_ps(_src, scale), tail);
        }

        template <bool align, bool nofma, bool notail> void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(width, F) && Aligned(dst));

            size_t widthQF = AlignLo(width, QF);
            size_t widthF = AlignLo(width, F);
            __mmask16 tail = TailMask16(width - widthF);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _bias = _mm512_set1_ps(bias[c]);
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        for (; w < widthQF; w += QF)
                        {
                            SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, w + F * 0);
                            SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, w + F * 1);
                            SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, w + F * 2);
                            SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, w + F * 3);
                        }
                        for (; w < widthF; w += F)
                            SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, w);
                        if (w < width)
                            SynetScaleLayerForward<align, true, notail>(src, _scale, _bias, dst, w, tail);
                        src += width;
                        dst += width;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        for (; w < widthQF; w += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, _scale, dst, w + F * 0);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, w + F * 1);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, w + F * 2);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, w + F * 3);
                        }
                        for (; w < widthF; w += F)
                            SynetScaleLayerForward<align, false>(src, _scale, dst, w);
                        if (w < width)
                            SynetScaleLayerForward<align, true>(src, _scale, dst, w, tail);
                        src += width;
                        dst += width;
                    }
                }
            }
        }

        template <bool nofma, bool notail> SIMD_INLINE void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
            if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                SynetScaleLayerForwardNchw<true, nofma, notail>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNchw<false, nofma, notail>(src, scale, bias, channels, height, width, dst);
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility) && bias)
                SynetScaleLayerForwardNchw<true, true>(src, scale, bias, channels, height, width, dst);
            else if (Base::FmaNoTail(compatibility) && bias)
                SynetScaleLayerForwardNchw<false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNchw<false, false>(src, scale, bias, channels, 1, height * width, dst);
        }

        template <bool align, bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t channelsQF = AlignLo(channels, QF);
            size_t channelsF = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - channelsF);
            if (bias)
            {
                size_t widthF = AlignLo(width, F);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, false, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, false, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, false, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, false, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, false, nofma>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<align, true, nofma>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, false, notail>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, false, notail>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, false, notail>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, false, notail>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, false, notail>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<align, true, notail>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align, false>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align, false>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align, false>(src, scale, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, false>(src, scale, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<align, true>(src, scale, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc3(const float* src, const float* scale, const float* bias, size_t height, size_t width, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(width));

            size_t width3 = width * 3;
            size_t widthF3 = AlignLo(width, F) * 3;
            if (bias)
            {
                float _scale[F * 3], _bias[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c], _bias[i * 3 + c] = bias[c];
                __m512 _scale0 = Load<false>(_scale + 0 * F);
                __m512 _scale1 = Load<false>(_scale + 1 * F);
                __m512 _scale2 = Load<false>(_scale + 2 * F);
                __m512 _bias0 = Load<false>(_bias + 0 * F);
                __m512 _bias1 = Load<false>(_bias + 1 * F);
                __m512 _bias2 = Load<false>(_bias + 2 * F);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align, false, nofma>(src, _scale0, _bias0, dst, w + F * 0);
                        SynetScaleLayerForward<align, false, nofma>(src, _scale1, _bias1, dst, w + F * 1);
                        SynetScaleLayerForward<align, false, nofma>(src, _scale2, _bias2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0] + bias[0];
                        dst[w + 1] = src[w + 1] * scale[1] + bias[1];
                        dst[w + 2] = src[w + 2] * scale[2] + bias[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
            else
            {
                float _scale[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c];
                __m512 _scale0 = Load<false>(_scale + 0 * F);
                __m512 _scale1 = Load<false>(_scale + 1 * F);
                __m512 _scale2 = Load<false>(_scale + 2 * F);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align, false>(src, _scale0, dst, w + F * 0);
                        SynetScaleLayerForward<align, false>(src, _scale1, dst, w + F * 1);
                        SynetScaleLayerForward<align, false>(src, _scale2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0];
                        dst[w + 1] = src[w + 1] * scale[1];
                        dst[w + 2] = src[w + 2] * scale[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
        }

        template<bool nofma, bool notail> SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
            if (channels == 3)
            {
                if (Aligned(src) && Aligned(dst) && Aligned(width))
                    SynetScaleLayerForwardNhwc3<true, nofma>(src, scale, bias, height, width, dst);
                else
                    SynetScaleLayerForwardNhwc3<false, nofma>(src, scale, bias, height, width, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true, nofma, notail>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNhwc<false, nofma, notail>(src, scale, bias, channels, height, width, dst);
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility) && bias)
                SynetScaleLayerForwardNhwc<true, true>(src, scale, bias, channels, 1, height * width, dst);
            else if (Base::FmaNoTail(compatibility) && bias)
                SynetScaleLayerForwardNhwc<false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNhwc<false, false>(src, scale, bias, channels, 1, height * width, dst);
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw16c(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            if (bias)
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m512 _scale = Load<false>(scale + c);
                    __m512 _bias = Load<false>(bias + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, s + F * 0);
                        SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, s + F * 1);
                        SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, s + F * 2);
                        SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align, false, nofma>(src, _scale, _bias, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m512 _scale = Load<false>(scale + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align, false>(src, _scale, dst, s + F * 0);
                        SynetScaleLayerForward<align, false>(src, _scale, dst, s + F * 1);
                        SynetScaleLayerForward<align, false>(src, _scale, dst, s + F * 2);
                        SynetScaleLayerForward<align, false>(src, _scale, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align, false>(src, _scale, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw16c(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility) && bias)
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw16c<true, true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw16c<false, true>(src, scale, bias, channels, spatial, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw16c<true, false>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw16c<false, false>(src, scale, bias, channels, spatial, dst);
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, height, width, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, height, width, dst, compatibility);
            else if (format == SimdTensorFormatNchw4c)
                Sse41::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
            else if (format == SimdTensorFormatNchw8c)
                Avx2::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
            else if (format == SimdTensorFormatNchw16c)
                SynetScaleLayerForwardNchw16c(src, scale, bias, channels, spatial, dst, compatibility);
            else
                Base::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
        }

        //-----------------------------------------------------------------------------------------

        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Avx2::SynetScale8i(p)
        {
        }

        //-----------------------------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, _upper, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            __m512 _shift = Load<align, mask>(shift + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, _scale, _shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, _upper, dst, c, tailF);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)(src + offset))));
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Sse41::Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper));
        }

        template <bool nofma> void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m512 _scale0 = Load<false>(_scale + 0 * F);
            __m512 _scale1 = Load<false>(_scale + 1 * F);
            __m512 _scale2 = Load<false>(_scale + 2 * F);
            __m512 _shift0 = Load<false>(_shift + 0 * F);
            __m512 _shift1 = Load<false>(_shift + 1 * F);
            __m512 _shift2 = Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial > HF)
            {
                if (nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m512 scale, __m512 shift, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            Store<false, mask>(dst + offset, Fmadd<nofma>(_src, scale, shift), tail);
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            __m512 _shift = Load<align, mask>(shift + offset, tail);
            Store<false, mask>(dst + offset, Fmadd<nofma>(_src, _scale, _shift), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, dst, c, tailF);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m512 scale, __m512 shift, float* dst, size_t offset)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)(src + offset))));
            Store<false>(dst + offset, Fmadd<nofma>(_src, scale, shift));
        }

        template <bool nofma> void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, float* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m512 _scale0 = Load<false>(_scale + 0 * F);
            __m512 _scale1 = Load<false>(_scale + 1 * F);
            __m512 _scale2 = Load<false>(_scale + 2 * F);
            __m512 _shift0 = Load<false>(_shift + 0 * F);
            __m512 _shift1 = Load<false>(_shift + 1 * F);
            __m512 _shift2 = Load<false>(_shift + 2 * F);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
            {
                if (nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const float* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<false, mask>(src + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool nofma> void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, _upper, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const float* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<false, mask>(src + offset, tail);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            __m512 _shift = Load<align, mask>(shift + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, _scale, _shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, _upper, dst, c, tailF);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const float* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m512 _src = Load<false>(src + offset);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Sse41::Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper));
        }

        template <bool nofma> void ScaleNhwc3(const float* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m512 _scale0 = Load<false>(_scale + 0 * F);
            __m512 _scale1 = Load<false>(_scale + 1 * F);
            __m512 _scale2 = Load<false>(_scale + 2 * F);
            __m512 _shift0 = Load<false>(_shift + 0 * F);
            __m512 _shift1 = Load<false>(_shift + 1 * F);
            __m512 _shift2 = Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
            {
                if (nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }

        //-----------------------------------------------------------------------------------------

        void SynetScale8i::Scale(const float* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                SynetScaleLayerForward(src, _scale.data, _shift.data, p.channels, 1, p.spatial, dst, p.format, p.compatibility);
                src += p.channels * p.spatial;
                dst += p.channels * p.spatial;
            }
        }

        //-----------------------------------------------------------------------------------------

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            Base::Scale8iParam param(batch, channels, spatial, srcType, dstType, format, compatibility);
            if (!param.Valid())
                return NULL;
            return new Avx512bw::SynetScale8i(param);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
