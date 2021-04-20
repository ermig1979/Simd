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
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, float * dst, size_t offset)
        {
            float32x4_t _bias = Load<align>(bias + offset);
            float32x4_t x = vaddq_f32(Load<align>(src + offset), _bias);
            float32x4_t _scale = Load<align>(scale + offset);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), x), vsubq_f32(x, vabsq_f32(x)), _scale));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, float32x4_t bias, float32x4_t scale, float * dst, size_t offset)
        {
            float32x4_t x = vaddq_f32(Load<align>(src + offset), bias);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), x), vsubq_f32(x, vabsq_f32(x)), scale));
        }


        template <bool align> void SynetFusedLayerForward0Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
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
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 0);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 1);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 2);
                        SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s);
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
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward0<align>(src, bias, scale, dst, c + F * 0);
                        SynetFusedLayerForward0<align>(src, bias, scale, dst, c + F * 1);
                        SynetFusedLayerForward0<align>(src, bias, scale, dst, c + F * 2);
                        SynetFusedLayerForward0<align>(src, bias, scale, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward0<align>(src, bias, scale, dst, c);
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

        template <bool align> void SynetFusedLayerForward0Nchw4c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _bias = Load<false>(bias + c);
                float32x4_t _scale = Load<false>(scale + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 0);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 1);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 2);
                    SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward0<align>(src, _bias, _scale, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward0Nchw4c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward0Nchw4c<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward0Nchw4c<false>(src, bias, scale, channels, spatial, dst);
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward0Nchw4c(src, bias, scale, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward0(src, bias, scale, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float *  bias1, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _bias0 = Load<align>(bias0 + offset);
            float32x4_t x = vaddq_f32(Load<align>(src + offset), _bias0);
            float32x4_t _scale1 = Load<align>(scale1 + offset);
            float32x4_t _bias1 = Load<align>(bias1 + offset);
            Store<align>(dst + offset, vaddq_f32(vmlaq_f32(_bias1, vmaxq_f32(_0, vnegq_f32(x)), _scale1), vmaxq_f32(_0, x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, float32x4_t bias0, float32x4_t scale1, float32x4_t bias1, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t x = vaddq_f32(Load<align>(src + offset), bias0);
            Store<align>(dst + offset, vaddq_f32(vmlaq_f32(bias1, vmaxq_f32(_0, vnegq_f32(x)), scale1), vmaxq_f32(_0, x)));
        }

        template <bool align> void SynetFusedLayerForward1Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _bias0 = vdupq_n_f32(bias0[c]);
                    float32x4_t _scale1 = vdupq_n_f32(scale1[c]);
                    float32x4_t _bias1 = vdupq_n_f32(bias1[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 0);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 1);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 2);
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s);
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
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, c + F * 0);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, c + F * 1);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, c + F * 2);
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, c);
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

        template <bool align> void SynetFusedLayerForward1Nchw4c(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _bias0 = Load<false>(bias0 + c);
                float32x4_t _scale1 = Load<false>(scale1 + c);
                float32x4_t _bias1 = Load<false>(bias1 + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 0);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 1);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 2);
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward1Nchw4c(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward1Nchw4c<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward1Nchw4c<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward1Nchw4c(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward1(src, bias0, scale1, bias1, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, float32x4_t slope, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _scale = Load<align>(scale + offset);
            float32x4_t _bias = Load<align>(bias + offset);
            float32x4_t x = vmlaq_f32(_bias, _src, _scale);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(_0, x), vminq_f32(_0, x), slope));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, float32x4_t scale, float32x4_t bias, float32x4_t slope, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t x = vmlaq_f32(bias, _src, scale);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(_0, x), vminq_f32(_0, x), slope));
        }

        template <bool align> void SynetFusedLayerForward2Nchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            float32x4_t _slope = vdupq_n_f32(slope[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 0);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 1);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 2);
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s);
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

            float32x4_t _slope = vdupq_n_f32(slope[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, c + F * 0);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, c + F * 1);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, c + F * 2);
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, c);
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

        template <bool align> void SynetFusedLayerForward2Nchw4c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float32x4_t _slope = vdupq_n_f32(slope[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _scale = Load<false>(scale + c);
                float32x4_t _bias = Load<false>(bias + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 0);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 1);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 2);
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward2Nchw4c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward2Nchw4c<true>(src, scale, bias, channels, spatial, slope, dst);
            else
                SynetFusedLayerForward2Nchw4c<false>(src, scale, bias, channels, spatial, slope, dst);
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nchw(src, scale, bias, channels, spatial, slope, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nhwc(src, scale, bias, channels, spatial, slope, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward2Nchw4c(src, scale, bias, channels, spatial, slope, dst);
            else
                Base::SynetFusedLayerForward2(src, scale, bias, channels, spatial, slope, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _bias = Load<align>(bias + offset);
            float32x4_t x = vaddq_f32(Load<align>(src + offset), _bias);
            float32x4_t _scale = Load<align>(scale + offset);
            float32x4_t pos = vmaxq_f32(_0, x);
            float32x4_t neg = vminq_f32(_0, x);
            Store<align>(dst + offset, vmlaq_f32(pos, _scale, neg));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, float32x4_t bias, float32x4_t scale, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t x = vaddq_f32(Load<align>(src + offset), bias);
            float32x4_t pos = vmaxq_f32(_0, x);
            float32x4_t neg = vminq_f32(_0, x);
            Store<align>(dst + offset, vmlaq_f32(pos, scale, neg));
        }

        template <bool align> void SynetFusedLayerForward3Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 0);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 1);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 2);
                        SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s);
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
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, c + F * 0);
                        SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, c + F * 1);
                        SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, c + F * 2);
                        SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, c);
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

        template <bool align> void SynetFusedLayerForward3Nchw4c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _bias = Load<false>(bias + c);
                float32x4_t _scale = Load<false>(scale + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 0);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 1);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 2);
                    SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward3Nchw4c(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward3Nchw4c<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward3Nchw4c<false>(src, bias, scale, channels, spatial, dst);
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward3Nchw4c(src, bias, scale, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward3(src, bias, scale, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, const float * bias0, float32x4_t scale1, float32x4_t bias1, float32x4_t _0, float * dst0, float * dst1, size_t offset)
        {
            float32x4_t x = vaddq_f32(Load<align>(src + offset), Load<align>(bias0 + offset));
            Store<align>(dst0 + offset, vmaxq_f32(_0, x));
            Store<align>(dst1 + offset, vmaxq_f32(_0, vmlaq_f32(bias1, scale1, x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, float32x4_t bias0, float32x4_t scale1, float32x4_t bias1, float32x4_t _0, float * dst0, float * dst1, size_t offset)
        {
            float32x4_t x = vaddq_f32(Load<align>(src + offset), bias0);
            Store<align>(dst0 + offset, vmaxq_f32(_0, x));
            Store<align>(dst1 + offset, vmaxq_f32(_0, vmlaq_f32(bias1, scale1, x)));
        }

        template <bool align> void SynetFusedLayerForward4Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst0));

            float32x4_t _bias1 = vdupq_n_f32(bias1[0]);
            float32x4_t _scale1 = vdupq_n_f32(scale1[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            float * dst1 = dst0 + channels * spatial;
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _bias0 = vdupq_n_f32(bias0[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 0);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 1);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 2);
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s);
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

            float32x4_t _bias1 = vdupq_n_f32(bias1[0]);
            float32x4_t _scale1 = vdupq_n_f32(scale1[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
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
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, c + F * 0);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, c + F * 1);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, c + F * 2);
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, c);
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

        template <bool align> void SynetFusedLayerForward4Nchw4cA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst0));

            float32x4_t _bias1 = vdupq_n_f32(bias1[0]);
            float32x4_t _scale1 = vdupq_n_f32(scale1[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            float * dst1 = dst0 + channels * spatial;
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _bias0 = Load<false>(bias0 + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 0);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 1);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 2);
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, s);
                src += spatialF;
                dst0 += spatialF;
                dst1 += spatialF;
            }
        }

        SIMD_INLINE void SynetFusedLayerForward4Nchw4cA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            assert(Aligned(channels, F));
            if (Aligned(src) && Aligned(dst))
                SynetFusedLayerForward4Nchw4cA<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nchw4cA<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c && Aligned(channels, F))
                SynetFusedLayerForward4Nchw4cA(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward4(src, bias0, scale1, bias1, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(Load<align>(src0 + offset), Load<align>(src1 + offset), Load<align>(src2 + offset)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward8(const float * src0, const float * src1, const float32x4_t & src2, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(Load<align>(src0 + offset), Load<align>(src1 + offset), src2));
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
                    float32x4_t _src2 = vdupq_n_f32(src2[c]);
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

        template <bool align> void SynetFusedLayerForward8Nchw4c(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _src2 = Load<false>(src2 + c);
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

        SIMD_INLINE void SynetFusedLayerForward8Nchw4c(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src0) && Aligned(src1) && Aligned(dst))
                SynetFusedLayerForward8Nchw4c<true>(src0, src1, src2, channels, spatial, dst);
            else
                SynetFusedLayerForward8Nchw4c<false>(src0, src1, src2, channels, spatial, dst);
        }

        void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nchw(src0, src1, src2, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nhwc(src0, src1, src2, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward8Nchw4c(src0, src1, src2, channels, spatial, dst);
            else
                Base::SynetFusedLayerForward8(src0, src1, src2, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float * scale, const float * bias, float * dst0, float * dst1, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, vmaxq_f32(vdupq_n_f32(0.0f), vmlaq_f32(Load<align>(bias + offset), _src, Load<align>(scale + offset))));
            Store<align>(dst1 + offset, _src);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float * scale, const float * bias, float * dst0, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, vmaxq_f32(vdupq_n_f32(0.0f), vmlaq_f32(Load<align>(bias + offset), _src, Load<align>(scale + offset))));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float32x4_t & scale, const float32x4_t & bias, float * dst0, float * dst1, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, vmaxq_f32(vdupq_n_f32(0.0f), vmlaq_f32(bias, _src, scale)));
            Store<align>(dst1 + offset, _src);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward9(const float * src, const float32x4_t & scale, const float32x4_t & bias, float * dst0, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst0 + offset, vmaxq_f32(vdupq_n_f32(0.0f), vmlaq_f32(bias, _src, scale)));
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
                        float32x4_t _scale0 = vdupq_n_f32(scale0[c]);
                        float32x4_t _bias0 = vdupq_n_f32(bias0[c]);
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
                        float32x4_t _scale1 = vdupq_n_f32(scale1[c]);
                        float32x4_t _bias1 = vdupq_n_f32(bias1[c]);
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
                        float32x4_t _scale0 = vdupq_n_f32(scale0[c]);
                        float32x4_t _bias0 = vdupq_n_f32(bias0[c]);
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
                        float32x4_t _scale1 = vdupq_n_f32(scale1[c]);
                        float32x4_t _bias1 = vdupq_n_f32(bias1[c]);
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

        template <bool align> void SynetFusedLayerForward9Nchw4cA(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
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
                    float32x4_t _scale0 = Load<false>(scale0 + c);
                    float32x4_t _bias0 = Load<false>(bias0 + c);
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
                    float32x4_t _scale1 = Load<false>(scale1 + c);
                    float32x4_t _bias1 = Load<false>(bias1 + c);
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
                    float32x4_t _scale0 = Load<false>(scale0 + c);
                    float32x4_t _bias0 = Load<false>(bias0 + c);
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
                    float32x4_t _scale1 = Load<false>(scale1 + c);
                    float32x4_t _bias1 = Load<false>(bias1 + c);
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

        SIMD_INLINE void SynetFusedLayerForward9Nchw4cA(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            assert(Aligned(channels0, F));
            if (Aligned(src0) && Aligned(src1) && Aligned(dst0) && Aligned(dst1))
                SynetFusedLayerForward9Nchw4cA<true>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                SynetFusedLayerForward9Nchw4cA<false>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
        }

        void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nchw(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (Base::NhwcCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nhwc(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (format == SimdTensorFormatNchw4c && Aligned(channels0, F))
                SynetFusedLayerForward9Nchw4cA(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                Base::SynetFusedLayerForward9(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1, format);
        }
    }
#endif// SIMD_NEON_ENABLE
}
