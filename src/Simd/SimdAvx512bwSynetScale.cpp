/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx512bw
    {
        template <bool mask, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _scale = (mask ? _mm512_maskz_loadu_ps(tail, scale + offset) : _mm512_loadu_ps(scale + offset));
            __m512 _bias = (mask ? _mm512_maskz_loadu_ps(tail, bias + offset) : _mm512_loadu_ps(bias + offset));
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, Fmadd<nofma>(_src, _scale, _bias));
            else
                _mm512_storeu_ps(dst + offset, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool mask> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _scale = (mask ? _mm512_maskz_loadu_ps(tail, scale + offset) : _mm512_loadu_ps(scale + offset));
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _mm512_mul_ps(_src, _scale));
            else
                _mm512_storeu_ps(dst + offset, _mm512_mul_ps(_src, _scale));
        }

        template <bool mask, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m512& scale, const __m512& bias, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, Fmadd<nofma>(_src, scale, bias));
            else
                _mm512_storeu_ps(dst + offset, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool mask> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m512& scale, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _mm512_mul_ps(_src, scale));
            else
                _mm512_storeu_ps(dst + offset, _mm512_mul_ps(_src, scale));
        }

        template <bool nofma, bool notail> void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
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
                            SynetScaleLayerForward<false, nofma>(src, _scale, _bias, dst, w + F * 0);
                            SynetScaleLayerForward<false, nofma>(src, _scale, _bias, dst, w + F * 1);
                            SynetScaleLayerForward<false, nofma>(src, _scale, _bias, dst, w + F * 2);
                            SynetScaleLayerForward<false, nofma>(src, _scale, _bias, dst, w + F * 3);
                        }
                        for (; w < widthF; w += F)
                            SynetScaleLayerForward<false, nofma>(src, _scale, _bias, dst, w);
                        if (w < width)
                            SynetScaleLayerForward<true, notail>(src, _scale, _bias, dst, w, tail);
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
                            SynetScaleLayerForward<false>(src, _scale, dst, w + F * 0);
                            SynetScaleLayerForward<false>(src, _scale, dst, w + F * 1);
                            SynetScaleLayerForward<false>(src, _scale, dst, w + F * 2);
                            SynetScaleLayerForward<false>(src, _scale, dst, w + F * 3);
                        }
                        for (; w < widthF; w += F)
                            SynetScaleLayerForward<false>(src, _scale, dst, w);
                        if (w < width)
                            SynetScaleLayerForward<true>(src, _scale, dst, w, tail);
                        src += width;
                        dst += width;
                    }
                }
            }
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

        template <bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst)
        {
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
                            SynetScaleLayerForward<false, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<false, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<false, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<false, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<false, nofma>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<true, nofma>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<false, notail>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<false, notail>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<false, notail>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<false, notail>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<false, notail>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<true, notail>(src, scale, bias, dst, c, tail);
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
                            SynetScaleLayerForward<false>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<false>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<false>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<false>(src, scale, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<false>(src, scale, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<true>(src, scale, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool nofma> void SynetScaleLayerForwardNhwc3(const float* src, const float* scale, const float* bias, size_t height, size_t width, float* dst)
        {
            size_t width3 = width * 3;
            size_t widthF3 = AlignLo(width, F) * 3;
            if (bias)
            {
                float _scale[F * 3], _bias[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c], _bias[i * 3 + c] = bias[c];
                __m512 _scale0 = _mm512_loadu_ps(_scale + 0 * F);
                __m512 _scale1 = _mm512_loadu_ps(_scale + 1 * F);
                __m512 _scale2 = _mm512_loadu_ps(_scale + 2 * F);
                __m512 _bias0 = _mm512_loadu_ps(_bias + 0 * F);
                __m512 _bias1 = _mm512_loadu_ps(_bias + 1 * F);
                __m512 _bias2 = _mm512_loadu_ps(_bias + 2 * F);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<false, nofma>(src, _scale0, _bias0, dst, w + F * 0);
                        SynetScaleLayerForward<false, nofma>(src, _scale1, _bias1, dst, w + F * 1);
                        SynetScaleLayerForward<false, nofma>(src, _scale2, _bias2, dst, w + F * 2);
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
                __m512 _scale0 = _mm512_loadu_ps(_scale + 0 * F);
                __m512 _scale1 = _mm512_loadu_ps(_scale + 1 * F);
                __m512 _scale2 = _mm512_loadu_ps(_scale + 2 * F);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<false>(src, _scale0, dst, w + F * 0);
                        SynetScaleLayerForward<false>(src, _scale1, dst, w + F * 1);
                        SynetScaleLayerForward<false>(src, _scale2, dst, w + F * 2);
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

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (channels == 3)
            {
                size_t h = height, w = width;
                if (!(Base::FmaNoTail(compatibility) && bias))
                {
                    w = height * width;
                    h = 1;
                }
                if (Base::FmaAvoid(compatibility) && bias)
                    SynetScaleLayerForwardNhwc3<true>(src, scale, bias, h, w, dst);
                else
                    SynetScaleLayerForwardNhwc3<false>(src, scale, bias, h, w, dst);
            }
            else if (Base::FmaAvoid(compatibility) && bias)
                SynetScaleLayerForwardNhwc<true, true>(src, scale, bias, channels, 1, height * width, dst);
            else if (Base::FmaNoTail(compatibility) && bias)
                SynetScaleLayerForwardNhwc<false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNhwc<false, false>(src, scale, bias, channels, 1, height * width, dst);
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, height, width, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, height, width, dst, compatibility);
            else
                assert(0);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
