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
#include "Simd/SimdConversion.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdTranspose.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx512bw
    {
        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetConvert32fTo8u(const float* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, __mmask16 tail = -1)
        {
            __m512i i32 = _mm512_cvtps_epi32(Fmadd<nofma>(Load<align, mask>(src, tail), scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(i32, K_ZERO), K_ZERO));
            Store<align, mask>(dst, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetConvert32fTo8u(const float* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, __mmask16 tail = -1)
        {
            SynetConvert32fTo8u<align, mask, nofma>(src, Load<align, mask>(scale, tail), Load<align, mask>(shift, tail), upper, dst, tail);
        }

        template <bool align, bool nofma> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(width, A));

            size_t widthF = AlignLo(width, F);
            __mmask16 tailF = TailMask16(width - widthF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        for (; w < widthF; w += F)
                            SynetConvert32fTo8u<align, false, nofma>(src + w, _scale, _shift, _upper, dst + w);
                        if( w < width)
                            SynetConvert32fTo8u<align, true, nofma>(src + w, _scale, _shift, _upper, dst + w, tailF);
                        src += width;
                        dst += width;
                    }
                }
            }
        }

        template <bool nofma> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(width, A))
                SynetConvert32fTo8uNchw<true, nofma>(src, batch, channels, height, width, scale, shift, upper, dst);
            else
                SynetConvert32fTo8uNchw<false, nofma>(src, batch, channels, height, width, scale, shift, upper, dst);
        }

        template <bool align, bool nofma, bool notail> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t widthF = AlignLo(width, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsF; c += F)
                            SynetConvert32fTo8u<align, false, nofma>(src + c, scale + c, shift + c, _upper, dst + c);
                        if (c < channels)
                            SynetConvert32fTo8u<align, true, nofma>(src + c, scale + c, shift + c, _upper, dst + c, tailF);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsF; c += F)
                            SynetConvert32fTo8u<align, false, notail>(src + c, scale + c, shift + c, _upper, dst + c);
                        if (c < channels)
                            SynetConvert32fTo8u<align, true, notail>(src + c, scale + c, shift + c, _upper, dst + c, tailF);
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool nofma, bool notail> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift))
                SynetConvert32fTo8uNhwc<true, nofma, notail>(src, batch, channels, height, width, scale, shift, upper, dst);
            else
                SynetConvert32fTo8uNhwc<false, nofma, notail>(src, batch, channels, height, width, scale, shift, upper, dst);
        }

        template <bool align, bool nofma> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(width, A));

            size_t width3 = width * 3;
            size_t width3F = AlignLo(width, F) * 3;
            __m128i _upper = _mm_set1_epi8(upper);
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
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < width3F; w += 3 * F)
                    {
                        SynetConvert32fTo8u<align, false, nofma>(src + 0 * F, _scale0, _shift0, _upper, dst + 0 * F);
                        SynetConvert32fTo8u<align, false, nofma>(src + 1 * F, _scale1, _shift1, _upper, dst + 1 * F);
                        SynetConvert32fTo8u<align, false, nofma>(src + 2 * F, _scale2, _shift2, _upper, dst + 2 * F);
                        src += 3 * F;
                        dst += 3 * F;
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[0] = Base::SynetConvert32fTo8u(src[0], scale[0], shift[0], 0, upper);
                        dst[1] = Base::SynetConvert32fTo8u(src[1], scale[1], shift[1], 0, upper);
                        dst[2] = Base::SynetConvert32fTo8u(src[2], scale[2], shift[2], 0, upper);
                        src += 3;
                        dst += 3;
                    }
                }
            }
        }

        template <bool nofma> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t height, size_t width, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(width, A))
                SynetConvert32fTo8uNhwc3<true, nofma>(src, batch, height, width, scale, shift, upper, dst);
            else
                SynetConvert32fTo8uNhwc3<false, nofma>(src, batch, height, width, scale, shift, upper, dst);
        }

        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            if (!Base::FmaNoTail(compatibility))
            {
                width = height * width;
                height = 1;
            }
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
            {
                if (Base::FmaAvoid(compatibility))
                    SynetConvert32fTo8uNchw<true>(src, batch, channels, height, width, scale, shift, upper, dst);
                else
                    SynetConvert32fTo8uNchw<false>(src, batch, channels, height, width, scale, shift, upper, dst);
            }
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                {
                    if (Base::FmaAvoid(compatibility))
                        SynetConvert32fTo8uNhwc3<true>(src, batch, height, width, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc3<false>(src, batch, height, width, scale, shift, upper, dst);
                }
                else
                {
                    if (Base::FmaAvoid(compatibility))
                        SynetConvert32fTo8uNhwc<true, true>(src, batch, channels, height, width, scale, shift, upper, dst);
                    else if (Base::FmaNoTail(compatibility))
                        SynetConvert32fTo8uNhwc<false, true>(src, batch, channels, height, width, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc<false, false>(src, batch, channels, height, width, scale, shift, upper, dst);
                }
            }
            else
                assert(0);
        }

        //-----------------------------------------------------------------------------------------

        template <bool nofma> SIMD_INLINE void SynetConvert8uTo32f(const uint8_t* src, const float* scale, const float* shift, float* dst, __mmask16 tail = -1)
        {
            __m512 f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src)));
            _mm512_mask_storeu_ps(dst, tail, Fmadd<nofma>(f32, _mm512_maskz_loadu_ps(tail, scale), _mm512_maskz_loadu_ps(tail, shift)));
        }

        template <bool nofma> SIMD_INLINE void SynetConvert8uTo32f(const uint8_t* src, const __m512& scale, const __m512& shift, float* dst, __mmask16 tail = -1)
        {
            __m512 f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src)));
            _mm512_mask_storeu_ps(dst, tail, Fmadd<nofma>(f32, scale, shift));
        }

        template <bool nofma> void SynetConvert8uTo32fNchw(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, const float* scale, const float* shift, float* dst)
        {
            size_t spatial = height * width;
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - spatialF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        SynetConvert8uTo32f<nofma>(src + s, _scale, _shift, dst + s);
                    if(s < spatial)
                        SynetConvert8uTo32f<nofma>(src + s, _scale, _shift, dst + s, tail);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool nofma> void SynetConvert8uTo32fNhwc(const uint8_t* src, size_t spatial, size_t channels, const float* scale, const float* shift, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - channelsF);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < channelsF; c += F)
                    SynetConvert8uTo32f<nofma>(src + c, scale + c, shift + c, dst + c);
                if (c < channels)
                    SynetConvert8uTo32f<nofma>(src + c, scale + c, shift + c, dst + c, tail);
                src += channels;
                dst += channels;
            }
        }

        template <bool nofma> void SynetConvert8uTo32fNhwc3(const uint8_t* src, size_t spatial, const float* scale, const float* shift, float* dst)
        {
            size_t spatial3 = spatial * 3;
            size_t spatial3F = AlignLo(spatial, F) * 3;

            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];

            __m512 _scale0 = _mm512_loadu_ps(_scale + 0 * F);
            __m512 _scale1 = _mm512_loadu_ps(_scale + 1 * F);
            __m512 _scale2 = _mm512_loadu_ps(_scale + 2 * F);
            __m512 _shift0 = _mm512_loadu_ps(_shift + 0 * F);
            __m512 _shift1 = _mm512_loadu_ps(_shift + 1 * F);
            __m512 _shift2 = _mm512_loadu_ps(_shift + 2 * F);

            size_t s = 0;
            for (; s < spatial3F; s += 3 * F)
            {
                SynetConvert8uTo32f<nofma>(src + 0 * F, _scale0, _shift0, dst + 0 * F);
                SynetConvert8uTo32f<nofma>(src + 1 * F, _scale1, _shift1, dst + 1 * F);
                SynetConvert8uTo32f<nofma>(src + 2 * F, _scale2, _shift2, dst + 2 * F);
                src += 3 * F;
                dst += 3 * F;
            }
            if (s < spatial3)
            {
                SynetConvert8uTo32f<nofma>(src + 0 * F, _scale0, _shift0, dst + 0 * F, TailMask16(spatial3 - spatial3F - 0 * F));
                SynetConvert8uTo32f<nofma>(src + 1 * F, _scale1, _shift1, dst + 1 * F, TailMask16(spatial3 - spatial3F - 1 * F));
                SynetConvert8uTo32f<nofma>(src + 2 * F, _scale2, _shift2, dst + 2 * F, TailMask16(spatial3 - spatial3F - 2 * F));
            }
        }

        void SynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility)
        {
            bool nofma = Base::FmaAvoid(compatibility);
            if (format == SimdTensorFormatNchw)
            {
                if (nofma)
                    SynetConvert8uTo32fNchw<true>(src, batch, channels, height, width, scale, shift, dst);
                else
                    SynetConvert8uTo32fNchw<false>(src, batch, channels, height, width, scale, shift, dst);
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t spatial = batch * height * width;
                if (channels == 3)
                {
                    if (nofma)
                        SynetConvert8uTo32fNhwc3<true>(src, spatial, scale, shift, dst);
                    else
                        SynetConvert8uTo32fNhwc3<false>(src, spatial, scale, shift, dst);
                }
                else
                {
                    if (nofma)
                        SynetConvert8uTo32fNhwc<true>(src, spatial, channels, scale, shift, dst);
                    else
                        SynetConvert8uTo32fNhwc<false>(src, spatial, channels, scale, shift, dst);
                }
            }
            else
                assert(0);
        }

        //-----------------------------------------------------------------------------------------

        template<bool align> void SynetReorderImage_Chw_Hwc(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channels8 = AlignLo(channels, 8);
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t s = 0;
            for (; s < spatial16; s += 16, src += 16, dst += 16 * channels)
            {
                size_t c = 0;
                const float* ps = src;
                float* pd = dst;
                for (; c < channels16; c += 16, ps += 16 * spatial, pd += 16)
                    Transpose16x16<align>(ps, spatial, pd, channels);
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Transpose16x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0x0 * channels] = ps[0x0];
                    pd[0x1 * channels] = ps[0x1];
                    pd[0x2 * channels] = ps[0x2];
                    pd[0x3 * channels] = ps[0x3];
                    pd[0x4 * channels] = ps[0x4];
                    pd[0x5 * channels] = ps[0x5];
                    pd[0x6 * channels] = ps[0x6];
                    pd[0x7 * channels] = ps[0x7];
                    pd[0x8 * channels] = ps[0x8];
                    pd[0x9 * channels] = ps[0x9];
                    pd[0xA * channels] = ps[0xA];
                    pd[0xB * channels] = ps[0xB];
                    pd[0xC * channels] = ps[0xC];
                    pd[0xD * channels] = ps[0xD];
                    pd[0xE * channels] = ps[0xE];
                    pd[0xF * channels] = ps[0xF];
                }
            }
            for (; s < spatial8; s += 8, src += 8, dst += 8 * channels)
            {
                size_t c = 0;
                const float* ps = src;
                float* pd = dst;
                for (; c < channels16; c += 16, ps += 16 * spatial, pd += 16)
                    Transpose8x16<align>(ps, spatial, pd, channels);
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Avx::Transpose8x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0x0 * channels] = ps[0x0];
                    pd[0x1 * channels] = ps[0x1];
                    pd[0x2 * channels] = ps[0x2];
                    pd[0x3 * channels] = ps[0x3];
                    pd[0x4 * channels] = ps[0x4];
                    pd[0x5 * channels] = ps[0x5];
                    pd[0x6 * channels] = ps[0x6];
                    pd[0x7 * channels] = ps[0x7];
                }
            }
            for (; s < spatial; ++s, src += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c * spatial];
        }

        template<bool align> void SynetReorderImage_Chw_Chw16c(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t tail = channels - channels16;
            size_t c = 0;
            for (; c < channels16; c += 16, src += 16 * spatial)
            {
                size_t s = 0;
                const float* ps = src;
                for (; s < spatial16; s += 16, dst += 16 * F, ps += 16)
                    Transpose16x16<align>(ps, spatial, dst, 16);
                for (; s < spatial8; s += 8, dst += 8 * F, ps += 8)
                    Transpose8x16<align>(ps, spatial, dst, 16);
                for (; s < spatial; ++s, dst += F, ps += 1)
                {
                    dst[0x0] = ps[0x0 * spatial];
                    dst[0x1] = ps[0x1 * spatial];
                    dst[0x2] = ps[0x2 * spatial];
                    dst[0x3] = ps[0x3 * spatial];
                    dst[0x4] = ps[0x4 * spatial];
                    dst[0x5] = ps[0x5 * spatial];
                    dst[0x6] = ps[0x6 * spatial];
                    dst[0x7] = ps[0x7 * spatial];
                    dst[0x8] = ps[0x8 * spatial];
                    dst[0x9] = ps[0x9 * spatial];
                    dst[0xA] = ps[0xA * spatial];
                    dst[0xB] = ps[0xB * spatial];
                    dst[0xC] = ps[0xC * spatial];
                    dst[0xD] = ps[0xD * spatial];
                    dst[0xE] = ps[0xE * spatial];
                    dst[0xF] = ps[0xF * spatial];
                }
            }
            if (tail)
            {
                const float* ps = src;
                for (size_t s = 0; s < spatial; ++s, dst += F, ps += 1)
                {
                    size_t i = 0;
                    for (; i < tail; ++i)
                        dst[i] = ps[i * spatial];
                    for (; i < F; ++i)
                        dst[i] = 0;
                }
            }
        }

        template<bool align> void SynetReorderImage_Hwc_Chw(size_t channels, size_t spatial, const float* src, float* dst)
        {
            SynetReorderImage_Chw_Hwc<align>(spatial, channels, src, dst);
        }

        template<bool align> void SynetReorderImage_Hwc_Chw16c(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            size_t spatial4 = AlignLo(spatial, 4);
            size_t stride = spatial * F;
            size_t c = 0;
            for (; c < channelsF4; c += 4 * F, src += 4 * F)
            {
                const float* ps = src;
                float* pd = dst;
                size_t i = 0;
                for (; i < spatial4; i += 4, pd += 4 * F, ps += 4 * channels)
                    Transpose4x4xF<align>(ps, channels, pd, stride);
                for (; i < spatial; ++i, pd += F, ps += channels)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * stride);
                    Copy<align>(ps + 1 * F, pd + 1 * stride);
                    Copy<align>(ps + 2 * F, pd + 2 * stride);
                    Copy<align>(ps + 3 * F, pd + 3 * stride);
                }
                dst += 4 * stride;
            }
            for (; c < channelsF; c += F, src += F)
            {
                const float* ps = src;
                for (size_t s = 0; s < spatial; ++s, ps += channels, dst += F)
                    Copy<align>(ps, dst);
            }
            if (tail)
            {
                __mmask16 mask = TailMask16(tail);
                const float* ps = src;
                for (size_t s = 0; s < spatial; ++s, ps += channels, dst += F)
                    CopyZP<align>(ps, dst, mask);
            }
        }

        template<bool align> void SynetReorderImage_Chw16c_Chw(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t tail = channels - channels16;
            size_t c = 0;
            for (; c < channels16; c += 16, dst += 16 * spatial, src += 16 * spatial)
            {
                const float* ps = src;
                size_t s = 0;
                for (; s < spatial16; s += 16, ps += 16 * F)
                    Transpose16x16<align>(ps, 16, dst + s, spatial);
                for (; s < spatial8; s += 8, ps += 8 * F)
                    Transpose16x8<align>(ps, 16, dst + s, spatial);
                for (; s < spatial; ++s, ps += 16)
                {
                    dst[s + 0x0 * spatial] = ps[0x0];
                    dst[s + 0x1 * spatial] = ps[0x1];
                    dst[s + 0x2 * spatial] = ps[0x2];
                    dst[s + 0x3 * spatial] = ps[0x3];
                    dst[s + 0x4 * spatial] = ps[0x4];
                    dst[s + 0x5 * spatial] = ps[0x5];
                    dst[s + 0x6 * spatial] = ps[0x6];
                    dst[s + 0x7 * spatial] = ps[0x7];
                    dst[s + 0x8 * spatial] = ps[0x8];
                    dst[s + 0x9 * spatial] = ps[0x9];
                    dst[s + 0xA * spatial] = ps[0xA];
                    dst[s + 0xB * spatial] = ps[0xB];
                    dst[s + 0xC * spatial] = ps[0xC];
                    dst[s + 0xD * spatial] = ps[0xD];
                    dst[s + 0xE * spatial] = ps[0xE];
                    dst[s + 0xF * spatial] = ps[0xF];
                }
            }
            if (tail)
            {
                const float* ps = src;
                for (size_t i = 0; i < tail; ++i, ps += 1, dst += spatial)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = ps[s * F];
                }
            }
        }

        template<bool align> void SynetReorderImage_Chw16c_Hwc(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t stride = F * spatial;
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            __mmask16 mask = TailMask16(tail);
            size_t spatial4 = AlignLo(spatial, 4);
            size_t s = 0;
            for (; s < spatial4; s += 4, src += 4 * F, dst += 4 * channels)
            {
                const float* ps = src;
                float* pd = dst;
                size_t c = 0;
                for (; c < channelsF4; c += 4 * F, ps += 4 * stride, pd += 4 * F)
                    Transpose4x4xF<align>(ps, stride, pd, channels);
                for (; c < channelsF; c += F, ps += stride, pd += F)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * channels);
                    Copy<align>(ps + 1 * F, pd + 1 * channels);
                    Copy<align>(ps + 2 * F, pd + 2 * channels);
                    Copy<align>(ps + 3 * F, pd + 3 * channels);
                }
                if (tail)
                {
                    Copy<align, true>(ps + 0 * F, pd + 0 * channels, mask);
                    Copy<align, true>(ps + 1 * F, pd + 1 * channels, mask);
                    Copy<align, true>(ps + 2 * F, pd + 2 * channels, mask);
                    Copy<align, true>(ps + 3 * F, pd + 3 * channels, mask);
                }
            }
            for (; s < spatial; ++s, src += F)
            {
                const float* ps = src;
                for (size_t c = 0; c < channelsF; c += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                    Copy<align, true>(ps, dst, mask), dst += tail;
            }
        }

        typedef void(*SynetImageConverterPtr)(size_t channels, size_t spatial, const float* src, float* dst);
        SynetImageConverterPtr GetImageConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatNchw)
            {
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw_Hwc<false>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Chw_Chw16c<false>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Hwc_Chw<false>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Hwc_Chw16c<false>;
            }
            if (src == SimdTensorFormatNchw16c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Chw16c_Chw<false>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw16c_Hwc<false>;
            }
            return NULL;
        }

        void SynetReorderImage(size_t batch, size_t channels, size_t spatial, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
        {
            SynetImageConverterPtr imageConverter = GetImageConverter(srcFormat, dstFormat);
            if (imageConverter)
            {
                size_t srcStride = AlignHi(channels, Base::SynetTensorAlignment(srcFormat)) * spatial;
                size_t dstStride = AlignHi(channels, Base::SynetTensorAlignment(dstFormat)) * spatial;
                for (size_t n = 0; n < batch; ++n)
                {
                    imageConverter(channels, spatial, src, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else
                return Avx::SynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
        }

        //-----------------------------------------------------------------------------------------

        template<bool align> void SynetReorderFilter_Oiyx_Yxio(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(output, input, src, dst);
                return;
            }
            size_t output8 = AlignLo(output, 8);
            size_t kernel8 = AlignLo(kernel, 8);
            size_t output16 = AlignLo(output, 16);
            size_t kernel16 = AlignLo(kernel, 16);
            size_t ik = input * kernel, oi = output * input;
            for (size_t i = 0; i < input; ++i, src += kernel, dst += output)
            {
                const float* ps = src;
                float* pd = dst;
                size_t k = 0;
                for (; k < kernel16; k += 16, ps += 16, pd += 16 * oi)
                {
                    size_t o = 0;
                    for (; o < output16; o += 16)
                        Transpose16x16<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output8; o += 8)
                        Transpose16x8<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0x0 * oi + o] = ps[o * ik + 0x0];
                        pd[0x1 * oi + o] = ps[o * ik + 0x1];
                        pd[0x2 * oi + o] = ps[o * ik + 0x2];
                        pd[0x3 * oi + o] = ps[o * ik + 0x3];
                        pd[0x4 * oi + o] = ps[o * ik + 0x4];
                        pd[0x5 * oi + o] = ps[o * ik + 0x5];
                        pd[0x6 * oi + o] = ps[o * ik + 0x6];
                        pd[0x7 * oi + o] = ps[o * ik + 0x7];
                        pd[0x8 * oi + o] = ps[o * ik + 0x8];
                        pd[0x9 * oi + o] = ps[o * ik + 0x9];
                        pd[0xA * oi + o] = ps[o * ik + 0xA];
                        pd[0xB * oi + o] = ps[o * ik + 0xB];
                        pd[0xC * oi + o] = ps[o * ik + 0xC];
                        pd[0xD * oi + o] = ps[o * ik + 0xD];
                        pd[0xE * oi + o] = ps[o * ik + 0xE];
                        pd[0xF * oi + o] = ps[o * ik + 0xF];
                    }
                }
                for (; k < kernel8; k += 8, ps += 8, pd += 8 * oi)
                {
                    size_t o = 0;
                    for (; o < output16; o += 16)
                        Transpose8x16<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output8; o += 8)
                        Avx::Transpose8x8<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0x0 * oi + o] = ps[o * ik + 0x0];
                        pd[0x1 * oi + o] = ps[o * ik + 0x1];
                        pd[0x2 * oi + o] = ps[o * ik + 0x2];
                        pd[0x3 * oi + o] = ps[o * ik + 0x3];
                        pd[0x4 * oi + o] = ps[o * ik + 0x4];
                        pd[0x5 * oi + o] = ps[o * ik + 0x5];
                        pd[0x6 * oi + o] = ps[o * ik + 0x6];
                        pd[0x7 * oi + o] = ps[o * ik + 0x7];
                    }
                }
                for (; k < kernel; ++k, ps += 1, pd += oi)
                    for (size_t o = 0; o < output; ++o)
                        pd[o] = ps[o * ik];
            }
        }

        template<bool align> void SynetReorderFilter_Oiyx_Oyxi16o(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Chw16c<align>(output, input, src, dst);
                return;
            }
            size_t output16 = AlignLo(output, 16);
            size_t kernel8 = AlignLo(kernel, 8);
            size_t tail = output - output16;
            size_t ik = input * kernel;
            size_t stride = input * 16;
            for (size_t o = 0; o < output16; o += F)
            {
                for (size_t i = 0; i < input; ++i)
                {
                    const float* ps = src + o * ik + i * kernel;
                    float* pd = dst + o * ik + i * 16;
                    size_t k = 0;
                    for (; k < kernel8; k += 8, ps += 8, pd += 8 * stride)
                        Transpose8x16<align>(ps, ik, pd, stride);
                    for (; k < kernel; ++k, ps += 1, pd += stride)
                        for (size_t j = 0; j < 16; ++j)
                            pd[j] = ps[j * ik];
                }
            }
            if (tail)
            {

                __mmask16 mask = TailMask16(tail);
                for (size_t i = 0; i < input; ++i)
                {
                    const float* ps = src + output16 * ik + i * kernel;
                    float* pd = dst + output16 * ik + i * 16;
                    for (size_t k = 0; k < kernel; ++k, ps += 1, pd += stride)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            pd[j] = ps[j * ik];
                        for (; j < 16; ++j)
                            pd[j] = 0;
                    }
                }
            }
        }

        template<bool align> void SynetReorderFilter_Yxio_Oiyx(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(input, output, src, dst);
                return;
            }
            SynetReorderFilter_Oiyx_Yxio<align>(kernel, input, output, src, dst);
        }

        template<bool align> void SynetReorderFilter_Yxio_Oyxi16o(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t ki = kernel * input;
            size_t stride = ki * F;
            size_t ki4 = AlignLo(ki, 4);
            size_t o = 0;
            for (; o < outputF4; o += 4 * F, src += 4 * F)
            {
                const float* ps = src;
                float* pd = dst;
                size_t i = 0;
                for (; i < ki4; i += 4, pd += 4 * F, ps += 4 * output)
                    Transpose4x4xF<align>(ps, output, pd, stride);
                for (; i < ki; ++i, pd += F, ps += output)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * stride);
                    Copy<align>(ps + 1 * F, pd + 1 * stride);
                    Copy<align>(ps + 2 * F, pd + 2 * stride);
                    Copy<align>(ps + 3 * F, pd + 3 * stride);
                }
                dst += 4 * stride;
            }
            for (; o < outputF; o += F, src += F)
            {
                const float* ps = src;
                float* pd = dst;
                size_t i = 0;
                for (; i < ki; ++i, pd += F, ps += output)
                    Copy<align>(ps, pd);
                dst += stride;
            }
            if (outputF < output)
            {
                size_t tail = output - outputF;
                __mmask16 mask = TailMask16(tail);
                for (size_t k = 0; k < kernel; ++k)
                    for (size_t i = 0; i < input; ++i, src += output, dst += F)
                        CopyZP<align>(src, dst, mask);
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi16o_Oiyx(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw16c_Chw<align>(output, input, src, dst);
                return;
            }
            size_t output16 = AlignLo(output, 16);
            size_t tail = output - output16;
            size_t kernel8 = AlignLo(kernel, 8);
            size_t ik = input * kernel;
            size_t stride = 16 * input;
            size_t o = 0;
            for (; o < output16; o += 16, src += 16 * ik)
            {
                const float* ps = src;
                float* pd = dst;
                for (size_t i = 0; i < input; ++i, ps += 16)
                {
                    size_t k = 0;
                    for (; k < kernel8; k += 8, pd += 8)
                        Transpose16x8<align>(ps + k * stride, stride, pd, ik);
                    for (; k < kernel; ++k, pd++)
                    {
                        pd[0x0 * ik] = ps[k * stride + 0x0];
                        pd[0x1 * ik] = ps[k * stride + 0x1];
                        pd[0x2 * ik] = ps[k * stride + 0x2];
                        pd[0x3 * ik] = ps[k * stride + 0x3];
                        pd[0x4 * ik] = ps[k * stride + 0x4];
                        pd[0x5 * ik] = ps[k * stride + 0x5];
                        pd[0x6 * ik] = ps[k * stride + 0x6];
                        pd[0x7 * ik] = ps[k * stride + 0x7];
                        pd[0x8 * ik] = ps[k * stride + 0x8];
                        pd[0x9 * ik] = ps[k * stride + 0x9];
                        pd[0xA * ik] = ps[k * stride + 0xA];
                        pd[0xB * ik] = ps[k * stride + 0xB];
                        pd[0xC * ik] = ps[k * stride + 0xC];
                        pd[0xD * ik] = ps[k * stride + 0xD];
                        pd[0xE * ik] = ps[k * stride + 0xE];
                        pd[0xF * ik] = ps[k * stride + 0xF];
                    }
                }
                dst += 16 * ik;
            }
            if (tail)
            {
                for (size_t j = 0; j < tail; ++j)
                {
                    const float* ps = src + j;
                    for (size_t i = 0; i < input; ++i, ps += 16)
                        for (size_t k = 0; k < kernel; ++k)
                            *(dst++) = ps[k * stride];
                }
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi16o_Yxio(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, 4 * F);
            size_t tail = output - outputF;
            __mmask16 mask = TailMask16(tail);
            size_t ki = kernel * input;
            size_t ki4 = AlignLo(ki, 4);
            size_t stride = ki * F;
            size_t i = 0;
            for (; i < ki4; i += 4, src += 4 * F)
            {
                const float* ps = src;
                float* pd = dst;
                size_t o = 0;
                for (; o < outputF4; o += 4 * F, ps += 4 * stride, pd += 4 * F)
                    Transpose4x4xF<align>(ps, stride, pd, output);
                for (; o < outputF; o += F, ps += stride, pd += F)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * output);
                    Copy<align>(ps + 1 * F, pd + 1 * output);
                    Copy<align>(ps + 2 * F, pd + 2 * output);
                    Copy<align>(ps + 3 * F, pd + 3 * output);
                }
                if (tail)
                {
                    Copy<align, true>(ps + 0 * F, pd + 0 * output, mask);
                    Copy<align, true>(ps + 1 * F, pd + 1 * output, mask);
                    Copy<align, true>(ps + 2 * F, pd + 2 * output, mask);
                    Copy<align, true>(ps + 3 * F, pd + 3 * output, mask);
                }
                dst += 4 * output;
            }
            for (; i < ki; ++i, src += F)
            {
                const float* ps = src;
                for (size_t o = 0; o < outputF; o += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                    Copy<align, true>(ps, dst, mask), dst += tail;
            }
        }

        typedef void(*SynetFilterConverterPtr)(size_t output, size_t input, size_t kernel, const float* src, float* dst);
        SynetFilterConverterPtr GetFilterConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatOiyx)
            {
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oiyx_Yxio<false>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Oiyx_Oyxi16o<false>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Yxio_Oiyx<false>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Yxio_Oyxi16o<false>;
            }
            if (src == SimdTensorFormatOyxi16o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Oyxi16o_Oiyx<false>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oyxi16o_Yxio<false>;
            }
            return NULL;
        }

        void SynetReorderFilter(size_t output, size_t input, size_t kernel, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
        {
            SynetFilterConverterPtr filterConverter = GetFilterConverter(srcFormat, dstFormat);
            if (filterConverter)
                filterConverter(output, input, kernel, src, dst);
            else
                Avx::SynetReorderFilter(output, input, kernel, src, srcFormat, dst, dstFormat);
        }

        //-----------------------------------------------------------------------------------------
 
        template <bool align> SIMD_INLINE void StoreScaled(float * ptr, __m512i value32, __m512 scale, __m512 shift)
        {
            Store<align>(ptr, _mm512_fmadd_ps(_mm512_cvtepi32_ps(value32), scale, shift));
        }

        const __m512i K16_BLUE_RED = SIMD_MM512_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m512i K16_GREEN_0000 = SIMD_MM512_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m512i K32_ROUND_TERM = SIMD_MM512_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m512i BgraToGray32(__m512i bgra)
        {
            const __m512i g0a0 = _mm512_shuffle_epi8(bgra, K8_SUFFLE_BGRA_TO_G0A0);
            const __m512i b0r0 = _mm512_and_si512(bgra, K16_00FF);
            const __m512i weightedSum = _mm512_add_epi32(_mm512_madd_epi16(g0a0, K16_GREEN_0000), _mm512_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm512_srli_epi32(_mm512_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInput1(const uint8_t * src, __m512 scale, __m512 shift, float * dst);

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatGray8>(const uint8_t * src, __m512 scale, __m512 shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 0)), scale, shift);
            StoreScaled<false>(dst + 1 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 1)), scale, shift);
            StoreScaled<false>(dst + 2 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 2)), scale, shift);
            StoreScaled<false>(dst + 3 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 3)), scale, shift);
        }

        const __m512i K8_SHUFFLE_BGR_TO_BGRA = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgr24>(const uint8_t * src, __m512 scale, __m512 shift, float * dst)
        {
            __m512i bgr0 = Load<false>(src + 0 * A);
            __m512i bgr1 = Load<false>(src + 1 * A);
            __m512i bgr2 = Load<false>(src + 2 * A);
            const __m512i bgra0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            const __m512i bgra1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            const __m512i bgra2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            const __m512i bgra3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);
            StoreScaled<false>(dst + 0 * F, BgraToGray32(_mm512_shuffle_epi8(bgra0, K8_SHUFFLE_BGR_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(_mm512_shuffle_epi8(bgra1, K8_SHUFFLE_BGR_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(_mm512_shuffle_epi8(bgra2, K8_SHUFFLE_BGR_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(_mm512_shuffle_epi8(bgra3, K8_SHUFFLE_BGR_TO_BGRA)), scale, shift);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgra32>(const uint8_t * src, __m512 scale, __m512 shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, BgraToGray32(Load<false>((__m512i*)src + 0)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(Load<false>((__m512i*)src + 1)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(Load<false>((__m512i*)src + 2)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(Load<false>((__m512i*)src + 3)), scale, shift);
        }

        const __m512i K8_SHUFFLE_RGB_TO_BGRA = SIMD_MM512_SETR_EPI8(
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1);

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatRgb24>(const uint8_t * src, __m512 scale, __m512 shift, float * dst)
        {
            __m512i bgr0 = Load<false>(src + 0 * A);
            __m512i bgr1 = Load<false>(src + 1 * A);
            __m512i bgr2 = Load<false>(src + 2 * A);
            const __m512i bgra0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            const __m512i bgra1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            const __m512i bgra2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            const __m512i bgra3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);
            StoreScaled<false>(dst + 0 * F, BgraToGray32(_mm512_shuffle_epi8(bgra0, K8_SHUFFLE_RGB_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(_mm512_shuffle_epi8(bgra1, K8_SHUFFLE_RGB_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(_mm512_shuffle_epi8(bgra2, K8_SHUFFLE_RGB_TO_BGRA)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(_mm512_shuffle_epi8(bgra3, K8_SHUFFLE_RGB_TO_BGRA)), scale, shift);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInput1(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            __m512 _scale = _mm512_set1_ps(scale[0]);
            __m512 _shift = _mm512_set1_ps(shift[0]);
            size_t aligned = AlignLo(width, A);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInput1<format>(src + step * x, _scale, _shift, dst + x);
                if(aligned < width)
                    SynetSetInput1<format>(src + step * (width - A), _scale, _shift, dst + width - A);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNchw3A(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel);

        template<> SIMD_INLINE void SynetSetInputNchw3A<SimdPixelFormatGray8>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            __m512i gray0 = _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 0));
            __m512i gray1 = _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 1));
            __m512i gray2 = _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 2));
            __m512i gray3 = _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 3));
            StoreScaled<false>(dst + 0 * F, gray0, scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[0], shift[0]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[0], shift[0]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[0], shift[0]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[1], shift[1]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[1], shift[1]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[1], shift[1]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[2], shift[2]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[2], shift[2]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[2], shift[2]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[2], shift[2]);
        }

        const __m512i K8_SHUFFLE_BGR_TO_B32 = SIMD_MM512_SETR_EPI8(
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1);

        const __m512i K8_SHUFFLE_BGR_TO_G32 = SIMD_MM512_SETR_EPI8(
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1);

        const __m512i K8_SHUFFLE_BGR_TO_R32 = SIMD_MM512_SETR_EPI8(
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1);

        SIMD_INLINE void SynetSetInputNchw3Bgr(__m512i bgra, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            StoreScaled<false>(dst + 0 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_B32), scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_G32), scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_R32), scale[2], shift[2]);
        }  

        const __m512i K32_PERMUTE_BGR_TO_BGRA_BEG = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);
        const __m512i K32_PERMUTE_BGR_TO_BGRA_END = SIMD_MM512_SETR_EPI32(0x4, 0x5, 0x6, -1, 0x7, 0x8, 0x9, -1, 0xA, 0xB, 0xC, -1, 0xD, 0xE, 0xF, -1);

        template<> SIMD_INLINE void SynetSetInputNchw3A<SimdPixelFormatBgr24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Bgr(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 0 * F)), scale, shift, dst + 0 * F, channel);
            SynetSetInputNchw3Bgr(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 3 * F)), scale, shift, dst + 1 * F, channel);
            SynetSetInputNchw3Bgr(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 6 * F)), scale, shift, dst + 2 * F, channel);
            SynetSetInputNchw3Bgr(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src + 8 * F)), scale, shift, dst + 3 * F, channel);
        }

        const __m512i K8_SHUFFLE_BGRA_TO_B32 = SIMD_MM512_SETR_EPI8(
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1);

        const __m512i K8_SHUFFLE_BGRA_TO_G32 = SIMD_MM512_SETR_EPI8(
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1);

        const __m512i K8_SHUFFLE_BGRA_TO_R32 = SIMD_MM512_SETR_EPI8(
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1);

        SIMD_INLINE void SynetSetInputNchw3Bgra(__m512i bgra, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            StoreScaled<false>(dst + 0 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGRA_TO_B32), scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGRA_TO_G32), scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGRA_TO_R32), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3A<SimdPixelFormatBgra32>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Bgra(Load<false>(src + 0 * A), scale, shift, dst + 0 * F, channel);
            SynetSetInputNchw3Bgra(Load<false>(src + 1 * A), scale, shift, dst + 1 * F, channel);
            SynetSetInputNchw3Bgra(Load<false>(src + 2 * A), scale, shift, dst + 2 * F, channel);
            SynetSetInputNchw3Bgra(Load<false>(src + 3 * A), scale, shift, dst + 3 * F, channel);
        }

        SIMD_INLINE void SynetSetInputNchw3Rgb(__m512i bgra, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            StoreScaled<false>(dst + 0 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_R32), scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_G32), scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * channel, _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGR_TO_B32), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3A<SimdPixelFormatRgb24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Rgb(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 0 * F)), scale, shift, dst + 0 * F, channel);
            SynetSetInputNchw3Rgb(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 3 * F)), scale, shift, dst + 1 * F, channel);
            SynetSetInputNchw3Rgb(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 6 * F)), scale, shift, dst + 2 * F, channel);
            SynetSetInputNchw3Rgb(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src + 8 * F)), scale, shift, dst + 3 * F, channel);
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNchw3F(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel);

        template<> SIMD_INLINE void SynetSetInputNchw3F<SimdPixelFormatGray8>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            __m512i gray = _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src));
            StoreScaled<false>(dst + 0 * channel, gray, scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * channel, gray, scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * channel, gray, scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3F<SimdPixelFormatBgr24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Bgr(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src - F)), scale, shift, dst, channel);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3F<SimdPixelFormatBgra32>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Bgra(Load<false>(src), scale, shift, dst, channel);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3F<SimdPixelFormatRgb24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Rgb(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src - F)), scale, shift, dst, channel);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t widthF = AlignLo(width, F), widthA = AlignLo(width, A), channel = width * height;
            __m512 _scale[3], _shift[3];
            for (size_t i = 0; i < 3; ++i)
            {
                _scale[i] = _mm512_set1_ps(scale[i]);
                _shift[i] = _mm512_set1_ps(shift[i]);
            }
            for (size_t y = 0; y < height; ++y)
            {
                size_t x = 0;
                for (; x < widthA; x += A)
                    SynetSetInputNchw3A<format>(src + step * x, _scale, _shift, dst + x, channel);
                for (; x < widthF; x += F)
                    SynetSetInputNchw3F<format>(src + step * x, _scale, _shift, dst + x, channel);
                if (widthF < width)
                    SynetSetInputNchw3F<format>(src + step * (width - F), _scale, _shift, dst + width - F, channel);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNhwc3A(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst);

        template<> SIMD_INLINE void SynetSetInputNhwc3A<SimdPixelFormatGray8>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m128i gray0 = Sse41::Load<false>((__m128i*)src + 0);
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR2)), scale[2], shift[2]);
            __m128i gray1 = Sse41::Load<false>((__m128i*)src + 1);
            StoreScaled<false>(dst + 0x3 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray1, Sse41::K8_SHUFFLE_GRAY_TO_BGR0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x4 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray1, Sse41::K8_SHUFFLE_GRAY_TO_BGR1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray1, Sse41::K8_SHUFFLE_GRAY_TO_BGR2)), scale[2], shift[2]);
            __m128i gray2 = Sse41::Load<false>((__m128i*)src + 2);
            StoreScaled<false>(dst + 0x6 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray2, Sse41::K8_SHUFFLE_GRAY_TO_BGR0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray2, Sse41::K8_SHUFFLE_GRAY_TO_BGR1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x8 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray2, Sse41::K8_SHUFFLE_GRAY_TO_BGR2)), scale[2], shift[2]);
            __m128i gray3 = Sse41::Load<false>((__m128i*)src + 3);
            StoreScaled<false>(dst + 0x9 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray3, Sse41::K8_SHUFFLE_GRAY_TO_BGR0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray3, Sse41::K8_SHUFFLE_GRAY_TO_BGR1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray3, Sse41::K8_SHUFFLE_GRAY_TO_BGR2)), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3A<SimdPixelFormatBgr24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m512i src0 = Load<false>((__m512i*)src + 0);
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src0, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src0, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src0, 2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src0, 3)), scale[0], shift[0]);
            __m512i src1 = Load<false>((__m512i*)src + 1);
            StoreScaled<false>(dst + 0x4 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src1, 0)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src1, 1)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x6 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src1, 2)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src1, 3)), scale[1], shift[1]);
            __m512i src2 = Load<false>((__m512i*)src + 2);
            StoreScaled<false>(dst + 0x8 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src2, 0)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src2, 1)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src2, 2)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(src2, 3)), scale[2], shift[2]);
        }

        const __m512i K8_SUFFLE_BGRA_TO_BGR = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGRA_TO_BGR = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        template<> SIMD_INLINE void SynetSetInputNhwc3A<SimdPixelFormatBgra32>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m512i bgr0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(Load<false>(src + 0 * A), K8_SUFFLE_BGRA_TO_BGR));
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 2)), scale[2], shift[2]);
            __m512i bgr1 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(Load<false>(src + 1 * A), K8_SUFFLE_BGRA_TO_BGR));
            StoreScaled<false>(dst + 0x3 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x4 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 2)), scale[2], shift[2]);
            __m512i bgr2 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(Load<false>(src + 2 * A), K8_SUFFLE_BGRA_TO_BGR));
            StoreScaled<false>(dst + 0x6 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x8 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 2)), scale[2], shift[2]);
            __m512i bgr3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(Load<false>(src + 3 * A), K8_SUFFLE_BGRA_TO_BGR));
            StoreScaled<false>(dst + 0x9 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 2)), scale[2], shift[2]);
        }

        const __m512i K8_SUFFLE_RGB_TO_BGR = SIMD_MM512_SETR_EPI8(
            0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB, 0xA, 0x9, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB, 0xA, 0x9, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB, 0xA, 0x9, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB, 0xA, 0x9, -1, -1, -1, -1);

        template<> SIMD_INLINE void SynetSetInputNhwc3A<SimdPixelFormatRgb24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m512i bgr0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 0*F)), K8_SUFFLE_RGB_TO_BGR));
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr0, 2)), scale[2], shift[2]);
            __m512i bgr1 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 3*F)), K8_SUFFLE_RGB_TO_BGR));
            StoreScaled<false>(dst + 0x3 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x4 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr1, 2)), scale[2], shift[2]);
            __m512i bgr2 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_BEG, Load<false>(src + 6*F)), K8_SUFFLE_RGB_TO_BGR));
            StoreScaled<false>(dst + 0x6 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x8 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr2, 2)), scale[2], shift[2]);
            __m512i bgr3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src + 8*F)), K8_SUFFLE_RGB_TO_BGR));
            StoreScaled<false>(dst + 0x9 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr3, 2)), scale[2], shift[2]);
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNhwc3F(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst);

        template<> SIMD_INLINE void SynetSetInputNhwc3F<SimdPixelFormatGray8>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m128i gray0 = Sse41::Load<false>((__m128i*)src + 0);
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm_shuffle_epi8(gray0, Sse41::K8_SHUFFLE_GRAY_TO_BGR2)), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3F<SimdPixelFormatBgr24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 0x0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 0x1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(Sse41::Load<false>((__m128i*)src + 0x2)), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3F<SimdPixelFormatBgra32>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m512i bgr = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(Load<false>(src), K8_SUFFLE_BGRA_TO_BGR));
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 2)), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3F<SimdPixelFormatRgb24>(const uint8_t * src, const __m512 * scale, const __m512 * shift, float * dst)
        {
            __m512i bgr = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_END, Load<false>(src - F)), K8_SUFFLE_RGB_TO_BGR));
            StoreScaled<false>(dst + 0x0 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bgr, 2)), scale[2], shift[2]);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t widthF = AlignLo(width, F);
            size_t widthA = AlignLo(width, A);
            __m512 _scale[3], _shift[3];
            for (float *sc = (float*)_scale, *sh = (float*)_shift, *end = sc + 48; sc < end; sc += 3, sh += 3)
            {
                sc[0] = scale[0]; sc[1] = scale[1]; sc[2] = scale[2]; 
                sh[0] = shift[0]; sh[1] = shift[1]; sh[2] = shift[2];
            }
            for (size_t y = 0; y < height; ++y)
            {
                size_t x = 0;
                for (; x < widthA; x += A)
                    SynetSetInputNhwc3A<format>(src + step * x, _scale, _shift, dst + 3 * x);
                for (; x < widthF; x += F)
                    SynetSetInputNhwc3F<format>(src + step * x, _scale, _shift, dst + 3 * x);
                if (widthF < width)
                    SynetSetInputNhwc3F<format>(src + step * (width - F), _scale, _shift, dst + 3 * (width - F));
                src += stride;
                dst += 3*width;
            }
        }

        void SynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            assert(width >= A);

            float scale[3];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SynetSetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SynetSetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SynetSetInput1<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SynetSetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
            default: assert(0);
            }
        }
    }
#endif//SIMD_AVX512BW_ENABLE
}
