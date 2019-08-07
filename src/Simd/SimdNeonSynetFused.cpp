/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifdef SIMD_NEON_ENABLE  
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
                assert(Aligned(src) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward0Nchw<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward0Nchw<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward0Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels) && Aligned(dst))
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
                assert(Aligned(src) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward1Nchw<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward1Nchw<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward1Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) && Aligned(channels) && Aligned(dst))
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
                assert(Aligned(src) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward2Nchw<true>(src, scale, bias, channels, spatial, slope, dst);
            else
                SynetFusedLayerForward2Nchw<false>(src, scale, bias, channels, spatial, slope, dst);
        }

        template <bool align> void SynetFusedLayerForward2Nhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels) && Aligned(dst))
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
                assert(Aligned(src) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward3Nchw<true>(src, bias, scale, channels, spatial, dst);
            else
                SynetFusedLayerForward3Nchw<false>(src, bias, scale, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward3Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src) && Aligned(bias) && Aligned(scale) && Aligned(channels) && Aligned(dst))
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
                assert(Aligned(src) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward4Nchw<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nchw<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward4Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(bias0) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src) && Aligned(bias0) && Aligned(channels) && Aligned(dst))
                SynetFusedLayerForward4Nhwc<true>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4Nhwc<false>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward4Nchw4cA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

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
            assert(Aligned(channels));
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
            else if (format == SimdTensorFormatNchw4c && Aligned(channels))
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
                assert(Aligned(src0) && Aligned(src1) && Aligned(spatial) && Aligned(dst));

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
            if (Aligned(src0) && Aligned(src1) && Aligned(spatial) && Aligned(dst))
                SynetFusedLayerForward8Nchw<true>(src0, src1, src2, channels, spatial, dst);
            else
                SynetFusedLayerForward8Nchw<false>(src0, src1, src2, channels, spatial, dst);
        }

        template <bool align> void SynetFusedLayerForward8Nhwc(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src0) && Aligned(src1) && Aligned(src2) && Aligned(channels) && Aligned(dst));

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
            if (Aligned(src0) && Aligned(src1) && Aligned(src2) && Aligned(channels) && Aligned(dst))
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

        template<bool align> void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t count0, size_t count1, size_t size, float * dst0, float * dst1, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count0) && Aligned(count1) && Aligned(scale0) && Aligned(bias0) : Aligned(size)) && Aligned(src0) && Aligned(src1) && Aligned(dst0) && Aligned(dst1));
            const float * scale1 = scale0 + count0;
            const float * bias1 = bias0 + count0;
            if (trans || size == 1)
            {
                size_t aligned0 = AlignLo(count0, QF);
                size_t partial0 = AlignLo(count0, F);
                size_t aligned1 = AlignLo(count1, QF);
                size_t partial1 = AlignLo(count1, F);
                if (dst1)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned0; i += QF)
                        {
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, i + 0 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, i + 1 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, i + 2 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, i + 3 * F);
                        }
                        for (; i < partial0; i += F)
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, dst1, i);
                        for (; i < count0; ++i)
                            dst0[i] = Base::SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]), dst1[i] = src0[i];
                        src0 += count0;
                        dst0 += count0;
                        dst1 += count0;
                        i = 0;
                        for (; i < aligned1; i += QF)
                        {
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, i + 0 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, i + 1 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, i + 2 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, i + 3 * F);
                        }
                        for (; i < partial1; i += F)
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, dst1, i);
                        for (; i < count1; ++i)
                            dst0[i] = Base::SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]), dst1[i] = src1[i];
                        src1 += count1;
                        dst0 += count1;
                        dst1 += count1;
                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned0; i += QF)
                        {
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, i + 0 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, i + 1 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, i + 2 * F);
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, i + 3 * F);
                        }
                        for (; i < partial0; i += F)
                            SynetFusedLayerForward9<align>(src0, scale0, bias0, dst0, i);
                        for (; i < count0; ++i)
                            dst0[i] = Base::SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]);
                        src0 += count0;
                        dst0 += count0;
                        i = 0;
                        for (; i < aligned1; i += QF)
                        {
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, i + 0 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, i + 1 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, i + 2 * F);
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, i + 3 * F);
                        }
                        for (; i < partial1; i += F)
                            SynetFusedLayerForward9<align>(src1, scale1, bias1, dst0, i);
                        for (; i < count1; ++i)
                            dst0[i] = Base::SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]);
                        src1 += count1;
                        dst0 += count1;
                    }
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                if (dst1)
                {
                    for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size, dst1 += size)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            float32x4_t _scale0 = vdupq_n_f32(scale0[i]);
                            float32x4_t _bias0 = vdupq_n_f32(bias0[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, j + 0 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, j + 1 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, j + 2 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, j + 3 * F);
                            }
                            for (; j < partial; j += F)
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, dst1, j);
                        }
                        for (; j < size; ++j)
                            dst0[j] = Base::SynetFusedLayerForward9(src0[j], scale0[i], bias0[i]), dst1[j] = src0[j];
                    }
                    for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size, dst1 += size)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            float32x4_t _scale1 = vdupq_n_f32(scale1[i]);
                            float32x4_t _bias1 = vdupq_n_f32(bias1[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, j + 0 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, j + 1 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, j + 2 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, j + 3 * F);
                            }
                            for (; j < partial; j += F)
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, dst1, j);
                        }
                        for (; j < size; ++j)
                            dst0[j] = Base::SynetFusedLayerForward9(src1[j], scale1[i], bias1[i]), dst1[j] = src1[j];
                    }
                }
                else
                {
                    for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            float32x4_t _scale0 = vdupq_n_f32(scale0[i]);
                            float32x4_t _bias0 = vdupq_n_f32(bias0[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, j + 0 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, j + 1 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, j + 2 * F);
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, j + 3 * F);
                            }
                            for (; j < partial; j += F)
                                SynetFusedLayerForward9<align>(src0, _scale0, _bias0, dst0, j);
                        }
                        for (; j < size; ++j)
                            dst0[j] = Base::SynetFusedLayerForward9(src0[j], scale0[i], bias0[i]);
                    }
                    for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            float32x4_t _scale1 = vdupq_n_f32(scale1[i]);
                            float32x4_t _bias1 = vdupq_n_f32(bias1[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, j + 0 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, j + 1 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, j + 2 * F);
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, j + 3 * F);
                            }
                            for (; j < partial; j += F)
                                SynetFusedLayerForward9<align>(src1, _scale1, _bias1, dst0, j);
                        }
                        for (; j < size; ++j)
                            dst0[j] = Base::SynetFusedLayerForward9(src1[j], scale1[i], bias1[i]);
                    }
                }
            }
        }

        void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t count0, size_t count1, size_t size, float * dst0, float * dst1, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count0) && Aligned(count1) && Aligned(scale0) && Aligned(bias0) : Aligned(size)) && Aligned(src0) && Aligned(src1) && Aligned(dst0) && Aligned(dst1))
                SynetFusedLayerForward9<true>(src0, src1, scale0, bias0, count0, count1, size, dst0, dst1, trans);
            else
                SynetFusedLayerForward9<false>(src0, src1, scale0, bias0, count0, count1, size, dst0, dst1, trans);
        }
    }
#endif// SIMD_NEON_ENABLE
}
