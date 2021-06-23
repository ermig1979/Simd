/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx
    {
        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)), Load<align>(bias + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), scale), bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), scale));
        }

        template <bool align> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[c]);
                        __m256 _bias = _mm256_set1_ps(bias[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 3);
                        }
                        for (; s < partial; s += F)
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, s);
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * scale[c] + bias[c];
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                        }
                        for (; s < partial; s += F)
                            SynetScaleLayerForward<align>(src, _scale, dst, s);
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * scale[c];
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetScaleLayerForwardNchw<true>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNchw<false>(src, scale, bias, channels, spatial, dst);
        }

        template <bool align> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    if (partial)
                    {
                        for (; c < aligned; c += QF)
                        {
                            SynetScaleLayerForward<align>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < partial; c += F)
                            SynetScaleLayerForward<align>(src, scale, bias, dst, c);
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] + bias[c];
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    if (partial)
                    {
                        for (; c < aligned; c += QF)
                        {
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 3);
                        }
                        for (; c < partial; c += F)
                            SynetScaleLayerForward<align>(src, scale, dst, c);
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool align> void SynetScaleLayerForwardNhwc3(const float * src, const float * scale, const float * bias, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            if (bias)
            {
                size_t s = 0;
                if (spatialF3)
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
                    for (; s < spatialF3; s += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, _bias0, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, _bias1, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, _bias2, dst, s + F * 2);
                    }
                }
                for (; s < spatial3; s += 3)
                {
                    dst[s + 0] = src[s + 0] * scale[0] + bias[0];
                    dst[s + 1] = src[s + 1] * scale[1] + bias[1];
                    dst[s + 2] = src[s + 2] * scale[2] + bias[2];
                }
            }
            else
            {
                size_t s = 0;
                if (spatialF3)
                {
                    float _scale[F * 3];
                    for (size_t i = 0; i < F; ++i)
                        for (size_t c = 0; c < 3; ++c)
                            _scale[i * 3 + c] = scale[c];
                    __m256 _scale0 = Load<false>(_scale + 0 * F);
                    __m256 _scale1 = Load<false>(_scale + 1 * F);
                    __m256 _scale2 = Load<false>(_scale + 2 * F);
                    for (; s < spatialF3; s += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, dst, s + F * 2);
                    }
                }
                for (; s < spatial3; s += 3)
                {
                    dst[s + 0] = src[s + 0] * scale[0];
                    dst[s + 1] = src[s + 1] * scale[1];
                    dst[s + 2] = src[s + 2] * scale[2];
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (channels == 3)
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNhwc3<true>(src, scale, bias, spatial, dst);
                else
                    SynetScaleLayerForwardNhwc3<false>(src, scale, bias, spatial, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNhwc<false>(src, scale, bias, channels, spatial, dst);
            }
        }

        template <bool align> void SynetScaleLayerForwardNchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            if (bias)
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    __m256 _bias = Load<false>(bias + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 2);
                        SynetScaleLayerForward<align>(src, _scale, _bias, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align>(src, _scale, _bias, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align>(src, _scale, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetScaleLayerForwardNchw8c<true>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNchw8c<false>(src, scale, bias, channels, spatial, dst);
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
            else if (format == SimdTensorFormatNchw8c)
                SynetScaleLayerForwardNchw8c(src, scale, bias, channels, spatial, dst);
            else
                Base::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
        }
    }
#endif// SIMD_AVX_ENABLE
}
