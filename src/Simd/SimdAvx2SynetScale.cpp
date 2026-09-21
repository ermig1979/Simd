/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _scale = Load<align>(scale + offset);
            __m256 _bias = Load<align>(bias + offset);
            Store<align>(dst + offset, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            __m256 _scale = _mm256_maskload_ps(scale + offset, tail);
            __m256 _bias = _mm256_maskload_ps(bias + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m256& scale, const __m256& bias, float* dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            Store<align>(dst + offset, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), scale));
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(width, F) && Aligned(dst));

            size_t widthQF = AlignLo(width, QF);
            size_t widthF = AlignLo(width, F);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            __m256 _bias = _mm256_set1_ps(bias[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 0);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 1);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 2);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c] + bias[c];
                        src += width;
                        dst += width;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 0);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 1);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 2);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align>(src, _scale, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c];
                        src += width;
                        dst += width;
                    }
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (!(Base::FmaAvoid(compatibility) && bias))
            {
                width = height * width;
                height = 1;
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, false>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, false>(src, scale, bias, channels, height, width, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, true>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, true>(src, scale, bias, channels, height, width, dst);
            }
        }

        template <bool align, bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsQF = AlignLo(channels, QF);
            if (bias)
            {
                size_t widthF = AlignLo(width, F);
                __m256i tail = LeftNotZero32i(channels - channelsF);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<nofma>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<notail>(src, scale, bias, dst, c, tail);
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
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align>(src, scale, dst, c);
                        for (; c < channels; ++c)
                            dst[c] = src[c] * scale[c];
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility) && bias)
                SynetScaleLayerForwardNhwc<align, true, true>(src, scale, bias, channels, height, width, dst);
            else if(Base::FmaNoTail(compatibility) && bias)
                SynetScaleLayerForwardNhwc<align, false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNhwc<align, false, false>(src, scale, bias, channels, height, width, dst);
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc3(const float * src, const float * scale, const float * bias, size_t height, size_t width, float * dst)
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
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);
                __m256 _bias0 = Load<false>(_bias + 0 * F);
                __m256 _bias1 = Load<false>(_bias + 1 * F);
                __m256 _bias2 = Load<false>(_bias + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale0, _bias0, dst, w + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale1, _bias1, dst, w + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale2, _bias2, dst, w + F * 2);
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
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, dst, w + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, dst, w + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, dst, w + F * 2);
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

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst, SimdSynetCompatibilityType compatibility)
        {
            if (!(Base::FmaNoTail(compatibility) && bias))
            {
                width = height * width;
                height = 1;
            }
            if (channels == 3)
            {
                if (Base::FmaAvoid(compatibility) && bias)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, true>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, true>(src, scale, bias, height, width, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, false>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, false>(src, scale, bias, height, width, dst);
                }
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true>(src, scale, bias, channels, height, width, dst, compatibility);
                else
                    SynetScaleLayerForwardNhwc<false>(src, scale, bias, channels, height, width, dst, compatibility);
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, width, height, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, height, width, dst, compatibility);
            else
                assert(0);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
