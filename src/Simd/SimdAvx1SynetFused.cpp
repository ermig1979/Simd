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
        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias);
            __m256 _scale = Load<align>(scale + offset);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(x, _mm256_andnot_ps(sign, x)), _scale), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, __m256 bias, __m256 scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(x, _mm256_andnot_ps(sign, x)), scale), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward0Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __m256 sign = _mm256_set1_ps(-0.0f);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _bias = _mm256_set1_ps(bias[c]);
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 0);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 1);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 2);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetFusedLayerForward0(src[s] + bias[c], scale[c]);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward0Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward0Nchw<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward0Nchw<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward0Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            __m256 sign = _mm256_set1_ps(-0.0f);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, c + F * 0);
                        SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, c + F * 1);
                        SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, c + F * 2);
                        SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetFusedLayerForward0(src[c] + bias[c], scale[c]);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward0Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward0Nhwc<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward0Nhwc<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward0Nchw8c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            __m256 sign = _mm256_set1_ps(-0.0f);
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _bias = Load<false>(bias + c);
                __m256 _scale = Load<false>(scale + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 0);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 1);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 2);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward0Nchw8c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward0Nchw8c<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward0Nchw8c<false>(src, bias, scale, channels, spatial, dst);
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward0(src, bias, scale, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward0Nchw8c(src, bias, scale, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward0(src, bias, scale, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float *  bias1, float * dst, size_t offset)
        {
            __m256 _bias0 = Load<align>(bias0 + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias0);
            __m256 _scale1 = Load<align>(scale1 + offset);
            __m256 _bias1 = Load<align>(bias1 + offset);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(_mm256_setzero_ps(), x)), _scale1), _bias1), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, __m256 bias0, __m256 scale1, __m256 bias1, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias0);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(_mm256_setzero_ps(), x)), scale1), bias1), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward1Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _bias0 = _mm256_set1_ps(bias0[c]);
                    __m256 _scale1 = _mm256_set1_ps(scale1[c]);
                    __m256 _bias1 = _mm256_set1_ps(bias1[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 0);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 1);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 2);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetFusedLayerForward1(src[s] + bias0[c], scale1[c], bias1[c]);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward1Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward1Nchw<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward1Nchw<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward1Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, c + F * 0);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, c + F * 1);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, c + F * 2);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetFusedLayerForward1(src[c] + bias0[c], scale1[c], bias1[c]);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward1Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward1Nhwc<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward1Nhwc<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward1Nchw8c(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _bias0 = Load<false>(bias0 + c);
                __m256 _scale1 = Load<false>(scale1 + c);
                __m256 _bias1 = Load<false>(bias1 + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 0);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 1);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 2);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward1Nchw8c(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward1Nchw8c<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward1Nchw8c<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward1(src, bias0, scale1, bias1, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward1Nchw8c(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward1(src, bias0, scale1, bias1, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, __m256 slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _scale = Load<align>(scale + offset);
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(_mm256_mul_ps(_src, _scale), _bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), x), _mm256_mul_ps(_mm256_min_ps(_mm256_setzero_ps(), x), slope)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, __m256 scale, __m256 bias, __m256 slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 x = _mm256_add_ps(_mm256_mul_ps(_src, scale), bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), x), _mm256_mul_ps(_mm256_min_ps(_mm256_setzero_ps(), x), slope)));
        }

        template <bool align> void SynetFusedLayerForward2Nchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            __m256 _slope = _mm256_set1_ps(slope[0]);
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _bias = _mm256_set1_ps(bias[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 0);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 1);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 2);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetFusedLayerForward2(src[s], scale[c], bias[c], slope[0]);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward2Nchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward2Nchw<true>(src, scale, bias, channels, spatial, slope, dst);
            else
                SynetFusedLayerForward2Nchw<false>(src, scale, bias, channels, spatial, slope, dst);
        }

        template <bool align> void SynetFusedLayerForward2Nhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            __m256 _slope = _mm256_set1_ps(slope[0]);
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, c + F * 0);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, c + F * 1);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, c + F * 2);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetFusedLayerForward2(src[c], scale[c], bias[c], slope[0]);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward2Nhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward2Nhwc<true>(src, scale, bias, channels, spatial, slope, dst);
            else
                SynetFusedLayerForward2Nhwc<false>(src, scale, bias, channels, spatial, slope, dst);
        }

        template <bool align> void SynetFusedLayerForward2Nchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m256 _slope = _mm256_set1_ps(slope[0]);
            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _scale = Load<false>(scale + c);
                __m256 _bias = Load<false>(bias + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 0);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 1);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 2);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward2Nchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward2Nchw8c<true>(src, scale, bias, channels, spatial, slope, dst);
            else
                SynetFusedLayerForward2Nchw8c<false>(src, scale, bias, channels, spatial, slope, dst);
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nchw(src, scale, bias, channels, spatial, slope, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nhwc(src, scale, bias, channels, spatial, slope, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward2(src, scale, bias, channels, spatial, slope, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward2Nchw8c(src, scale, bias, channels, spatial, slope, dst);
            else
                Base::SynetFusedLayerForward2(src, scale, bias, channels, spatial, slope, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, float * dst, size_t offset)
        {
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias);
            __m256 _scale = Load<align>(scale + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), x);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), x);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(_scale, neg)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, __m256 bias, __m256 scale, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), x);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), x);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(scale, neg)));
        }

        template <bool align> void SynetFusedLayerForward3Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _bias = _mm256_set1_ps(bias[c]);
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 0);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 1);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 2);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetFusedLayerForward3(src[s] + bias[c], scale[c]);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward3Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward3Nchw<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward3Nchw<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward3Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward3<align>(src, bias, scale, dst, c + F * 0);
                        SynetFusedLayerForward3<align>(src, bias, scale, dst, c + F * 1);
                        SynetFusedLayerForward3<align>(src, bias, scale, dst, c + F * 2);
                        SynetFusedLayerForward3<align>(src, bias, scale, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward3<align>(src, bias, scale, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetFusedLayerForward3(src[c] + bias[c], scale[c]);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward3Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward3Nhwc<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward3Nhwc<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward3Nchw8c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _bias = Load<false>(bias + c);
                __m256 _scale = Load<false>(scale + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 0);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 1);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 2);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward3<align>(src, _bias, _scale, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward3Nchw8c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward3Nchw8c<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward3Nchw8c<false>(src, bias, scale, channels, spatial, dst);
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward3(src, bias, scale, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward3Nchw8c(src, bias, scale, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward3(src, bias, scale, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, const float * bias0, __m256 scale1, __m256 bias1, float * dst0, float * dst1, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), Load<align>(bias0 + offset));
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), x));
            Store<align>(dst1 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(bias1, _mm256_mul_ps(scale1, x))));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, __m256 bias0, __m256 scale1, __m256 bias1, float * dst0, float * dst1, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias0);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), x));
            Store<align>(dst1 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(bias1, _mm256_mul_ps(scale1, x))));
        }

        template <bool align> void SynetFusedLayerForward4Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst0));

            __m256 _bias1 = _mm256_set1_ps(bias1[0]);
            __m256 _scale1 = _mm256_set1_ps(scale1[0]);
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            float * dst1 = dst0 + channels * spatial;
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _bias0 = _mm256_set1_ps(bias0[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 0);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 1);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 2);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s);
                }
                for (; s < spatial; ++s)
                    Base::SynetFusedLayerForward4(src[s], bias0[c], scale1[0], bias1[0], dst0 + s, dst1 + s);
                src += spatial;
                dst0 += spatial;
                dst1 += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward4Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward4Nchw<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nchw<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward4Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias0) && Aligned(channels, F) && Aligned(dst0));

            __m256 _bias1 = _mm256_set1_ps(bias1[0]);
            __m256 _scale1 = _mm256_set1_ps(scale1[0]);
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            float * dst1 = dst0 + channels;
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, c + F * 0);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, c + F * 1);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, c + F * 2);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, c);
                }
                for (; c < channels; ++c)
                    Base::SynetFusedLayerForward4(src[c], bias0[c], scale1[0], bias1[0], dst0 + c, dst1 + c);
                src += channels;
                dst0 += 2 * channels;
                dst1 += 2 * channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward4Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(bias0) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward4Nhwc<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nhwc<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward4Nchw8cA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst0));

            __m256 _bias1 = _mm256_set1_ps(bias1[0]);
            __m256 _scale1 = _mm256_set1_ps(scale1[0]);
            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            float * dst1 = dst0 + channels * spatial;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _bias0 = Load<false>(bias0 + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 0);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 1);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 2);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, s);
                src += spatialF;
                dst0 += spatialF;
                dst1 += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward4Nchw8cA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            assert(Aligned(channels, F));
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward4Nchw8cA<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nchw8cA<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward4(src, bias0, scale1, bias1, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c && Aligned(channels, F))
                SynetFusedLayerForward4Nchw8cA(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward4(src, bias0, scale1, bias1, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(Load<align>(src0 + offset), _mm256_mul_ps(Load<align>(src1 + offset), Load<align>(src2 + offset))));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward8(const float * src0, const float * src1, const __m256 & src2, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(Load<align>(src0 + offset), _mm256_mul_ps(Load<align>(src1 + offset), src2)));
        }

        template <bool align> void SynetFusedLayerForward8Nchw(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _src2 = _mm256_set1_ps(src2[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 0);
                        SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 1);
                        SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 2);
                        SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetFusedLayerForward8(src0[s], src1[s], src2[c]);
                src0 += spatial;
                src1 += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward8Nchw(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(spatial, F) && Aligned(dst))
                SynetFusedLayerForward8Nchw<true>(src0, src1, src2, channels, spatial, dst);
            else
                SynetFusedLayerForward8Nchw<false>(src0, src1, src2, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward8Nhwc(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(src2) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward8<align>(src0, src1, src2, dst, c + F * 0);
                        SynetFusedLayerForward8<align>(src0, src1, src2, dst, c + F * 1);
                        SynetFusedLayerForward8<align>(src0, src1, src2, dst, c + F * 2);
                        SynetFusedLayerForward8<align>(src0, src1, src2, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward8<align>(src0, src1, src2, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetFusedLayerForward8(src0[c], src1[c], src2[c]);
                src0 += channels;
                src1 += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward8Nhwc(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(src2) && Aligned(channels, F) && Aligned(dst))
                SynetFusedLayerForward8Nhwc<true>(src0, src1, src2, channels, spatial, dst);
            else
                SynetFusedLayerForward8Nhwc<false>(src0, src1, src2, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward8Nchw8c(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _src2 = Load<false>(src2 + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 0);
                    SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 1);
                    SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 2);
                    SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward8<align>(src0, src1, _src2, dst, s);
                src0 += spatialF;
                src1 += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward8Nchw8c(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(dst))
                SynetFusedLayerForward8Nchw8c<true>(src0, src1, src2, channels, spatial, dst);
            else
                SynetFusedLayerForward8Nchw8c<false>(src0, src1, src2, channels, spatial, dst);
        }

        void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nchw(src0, src1, src2, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nhwc(src0, src1, src2, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward8(src0, src1, src2, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward8Nchw8c(src0, src1, src2, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward8(src0, src1, src2, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float * scale, const float * bias, float * dst0, float * dst1, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(_mm256_mul_ps(_src, Load<align>(scale + offset)), Load<align>(bias + offset))));
            Store<align>(dst1 + offset, _src);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float * scale, const float * bias, float * dst0, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(_mm256_mul_ps(_src, Load<align>(scale + offset)), Load<align>(bias + offset))));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const __m256 & scale, const __m256 & bias, float * dst0, float * dst1, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(_mm256_mul_ps(_src, scale), bias)));
            Store<align>(dst1 + offset, _src);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const __m256 & scale, const __m256 & bias, float * dst0, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(_mm256_mul_ps(_src, scale), bias)));
        }

        template<bool align> void SynetFusedLayerForward9Nchw(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(spatial, F) && Aligned(dst0) && Aligned(dst1));
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            if (dst1)
            {
                for (size_t c = 0; c < channels0; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale0 = _mm256_set1_ps(scale0[c]);
                        __m256 _bias0 = _mm256_set1_ps(bias0[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + 0 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + 1 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + 2 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + 3 * F);
                        }
                        for (; s < partial; s += F)
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = Base::SynetFusedLayerForward9(src0[s], scale0[c], bias0[c]), dst1[s] = src0[s];
                    src0 += spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
                for (size_t c = 0; c < channels1; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale1 = _mm256_set1_ps(scale1[c]);
                        __m256 _bias1 = _mm256_set1_ps(bias1[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + 0 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + 1 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + 2 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + 3 * F);
                        }
                        for (; s < partial; s += F)
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = Base::SynetFusedLayerForward9(src1[s], scale1[c], bias1[c]), dst1[s] = src1[s];
                    src1 += spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels0; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale0 = _mm256_set1_ps(scale0[c]);
                        __m256 _bias0 = _mm256_set1_ps(bias0[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + 0 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + 1 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + 2 * F);
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + 3 * F);
                        }
                        for (; s < partial; s += F)
                            SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = Base::SynetFusedLayerForward9(src0[s], scale0[c], bias0[c]);
                    src0 += spatial;
                    dst0 += spatial;
                }
                for (size_t c = 0; c < channels1; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        __m256 _scale1 = _mm256_set1_ps(scale1[c]);
                        __m256 _bias1 = _mm256_set1_ps(bias1[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + 0 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + 1 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + 2 * F);
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + 3 * F);
                        }
                        for (; s < partial; s += F)
                            SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = Base::SynetFusedLayerForward9(src1[s], scale1[c], bias1[c]);
                    src1 += spatial;
                    dst0 += spatial;
                }
            }
        }

        SIMD_INLINE void SynetFusedLayerForward9Nchw(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(spatial, F) && Aligned(dst0) && Aligned(dst1))
                SynetFusedLayerForward9Nchw<true>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                SynetFusedLayerForward9Nchw<false>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
        }

        template<bool align> void SynetFusedLayerForward9Nhwc(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(scale0) && Aligned(bias0) && Aligned(channels0, F) && Aligned(channels1, F) && Aligned(dst0) && Aligned(dst1));
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            size_t aligned0 = AlignLo(channels0, QF);
            size_t partial0 = AlignLo(channels0, F);
            size_t aligned1 = AlignLo(channels1, QF);
            size_t partial1 = AlignLo(channels1, F);
            if (dst1)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned0; c += QF)
                    {
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, c + 0 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, c + 1 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, c + 2 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, c + 3 * F);
                    }
                    for (; c < partial0; c += F)
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, c);
                    for (; c < channels0; ++c)
                        dst0[c] = Base::SynetFusedLayerForward9(src0[c], scale0[c], bias0[c]), dst1[c] = src0[c];
                    src0 += channels0;
                    dst0 += channels0;
                    dst1 += channels0;
                    c = 0;
                    for (; c < aligned1; c += QF)
                    {
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, c + 0 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, c + 1 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, c + 2 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, c + 3 * F);
                    }
                    for (; c < partial1; c += F)
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, c);
                    for (; c < channels1; ++c)
                        dst0[c] = Base::SynetFusedLayerForward9(src1[c], scale1[c], bias1[c]), dst1[c] = src1[c];
                    src1 += channels1;
                    dst0 += channels1;
                    dst1 += channels1;
                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned0; c += QF)
                    {
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, c + 0 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, c + 1 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, c + 2 * F);
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, c + 3 * F);
                    }
                    for (; c < partial0; c += F)
                        SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, c);
                    for (; c < channels0; ++c)
                        dst0[c] = Base::SynetFusedLayerForward9(src0[c], scale0[c], bias0[c]);
                    src0 += channels0;
                    dst0 += channels0;
                    c = 0;
                    for (; c < aligned1; c += QF)
                    {
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, c + 0 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, c + 1 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, c + 2 * F);
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, c + 3 * F);
                    }
                    for (; c < partial1; c += F)
                        SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, c);
                    for (; c < channels1; ++c)
                        dst0[c] = Base::SynetFusedLayerForward9(src1[c], scale1[c], bias1[c]);
                    src1 += channels1;
                    dst0 += channels1;
                }
            }
        }

        SIMD_INLINE void SynetFusedLayerForward9Nhwc(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(scale) && Aligned(bias) && Aligned(channels0, F) && Aligned(channels1, F) && Aligned(dst0) && Aligned(dst1))
                SynetFusedLayerForward9Nhwc<true>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                SynetFusedLayerForward9Nhwc<false>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
        }

        template <bool align> void SynetFusedLayerForward9Nchw8cA(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(dst0) && Aligned(dst1));
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            if (dst1)
            {
                for (size_t c = 0; c < channels0; c += F)
                {
                    __m256 _scale0 = Load<false>(scale0 + c);
                    __m256 _bias0 = Load<false>(bias0 + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + F * 0);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + F * 1);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + F * 2);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, s);
                    src0 += spatialF;
                    dst0 += spatialF;
                    dst1 += spatialF;
                }
                for (size_t c = 0; c < channels1; c += F)
                {
                    __m256 _scale1 = Load<false>(scale1 + c);
                    __m256 _bias1 = Load<false>(bias1 + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + F * 0);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + F * 1);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + F * 2);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, s);
                    src1 += spatialF;
                    dst0 += spatialF;
                    dst1 += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels0; c += F)
                {
                    __m256 _scale0 = Load<false>(scale0 + c);
                    __m256 _bias0 = Load<false>(bias0 + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + F * 0);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + F * 1);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + F * 2);
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, s);
                    src0 += spatialF;
                    dst0 += spatialF;
                }
                for (size_t c = 0; c < channels1; c += F)
                {
                    __m256 _scale1 = Load<false>(scale1 + c);
                    __m256 _bias1 = Load<false>(bias1 + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + F * 0);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + F * 1);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + F * 2);
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, s);
                    src1 += spatialF;
                    dst0 += spatialF;
                }
            }
        }

        SIMD_INLINE void SynetFusedLayerForward9Nchw8cA(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            assert(Aligned(channels0, F));
            if (Aligned(src0) && Aligned(src1) && Aligned(dst0) && Aligned(dst1))
                SynetFusedLayerForward9Nchw8cA<true>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                SynetFusedLayerForward9Nchw8cA<false>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
        }

        void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nchw(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (Base::NhwcCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nhwc(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetFusedLayerForward9(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1, format);
            else if (format == SimdTensorFormatNchw8c && Aligned(channels0, F))
                SynetFusedLayerForward9Nchw8cA(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                Base::SynetFusedLayerForward9(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1, format);
        }
    }
#endif// SIMD_AVX_ENABLE
}
