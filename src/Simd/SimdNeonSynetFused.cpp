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

        template <bool align> void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst));
            float32x4_t _0 = vdupq_n_f32(0.0f);
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, i + 0 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, i + 1 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, i + 2 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, _0, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward1(src[i] + bias0[i], scale1[i], bias1[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _bias0 = vdupq_n_f32(bias0[i]);
                        float32x4_t _scale1 = vdupq_n_f32(scale1[i]);
                        float32x4_t _bias1 = vdupq_n_f32(bias1[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, j + 0 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, j + 1 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, j + 2 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, _0, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward1(src[j] + bias0[i], scale1[i], bias1[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward1<true>(src, bias0, scale1, bias1, count, size, dst, trans);
            else
                SynetFusedLayerForward1<false>(src, bias0, scale1, bias1, count, size, dst, trans);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, float32x4_t slope, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _scale = Load<align>(scale + offset);
            float32x4_t _bias = Load<align>(bias + offset);
            float32x4_t x = vmlaq_f32(_bias, _src, _scale);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(_0, x), vminq_f32(_0, x), slope));
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, float32x4_t scale, float32x4_t bias, float32x4_t slope, float32x4_t _0, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t x = vmlaq_f32(bias, _src, scale);
            Store<align>(dst + offset, vmlaq_f32(vmaxq_f32(_0, x), vminq_f32(_0, x), slope));
        }

        template <bool align> void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            float32x4_t _slope = vdupq_n_f32(slope[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, i + 0 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, i + 1 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, i + 2 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, _0, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward2(src[i], scale[i], bias[i], slope[0]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _scale = vdupq_n_f32(scale[i]);
                        float32x4_t _bias = vdupq_n_f32(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, j + 0 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, j + 1 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, j + 2 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, _0, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward2(src[j], scale[i], bias[i], slope[0]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward2<true>(src, scale, bias, count, size, slope, dst, trans);
            else
                SynetFusedLayerForward2<false>(src, scale, bias, count, size, slope, dst, trans);
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

        template <bool align> void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            float32x4_t _0 = vdupq_n_f32(0.0f);
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, i + 0 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, i + 1 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, i + 2 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward3<align>(src, bias, scale, _0, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward3(src[i] + bias[i], scale[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _bias = vdupq_n_f32(bias[i]);
                        float32x4_t _scale = vdupq_n_f32(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, j + 0 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, j + 1 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, j + 2 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward3<align>(src, _bias, _scale, _0, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward3(src[j] + bias[i], scale[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward3<true>(src, bias, scale, count, size, dst, trans);
            else
                SynetFusedLayerForward3<false>(src, bias, scale, count, size, dst, trans);
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

        template<bool align> void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) : Aligned(size)) && Aligned(src) && Aligned(dst));
            float32x4_t _scale1 = vdupq_n_f32(scale1[0]);
            float32x4_t _bias1 = vdupq_n_f32(bias1[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            if ((trans || size == 1) && count != 1)
            {
                float * dst0 = dst, *dst1 = dst + count;
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, i + 0 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, i + 1 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, i + 2 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, _0, dst0, dst1, i);
                    }
                    for (; i < count; ++i)
                        Base::SynetFusedLayerForward4(src[i], bias0[i], scale1[0], bias1[0], dst0 + i, dst1 + i);
                    src += count;
                    dst0 += 2 * count;
                    dst1 += 2 * count;
                }
            }
            else
            {
                float * dst0 = dst, *dst1 = dst + count * size;
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _bias0 = vdupq_n_f32(bias0[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, j + 0 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, j + 1 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, j + 2 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, _0, dst0, dst1, j);
                    }
                    for (; j < size; ++j)
                        Base::SynetFusedLayerForward4(src[j], bias0[i], scale1[0], bias1[0], dst0 + j, dst1 + j);
                    src += size;
                    dst0 += size;
                    dst1 += size;
                }
            }
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward4<true>(src, bias0, scale1, bias1, count, size, dst, trans);
            else
                SynetFusedLayerForward4<false>(src, bias0, scale1, bias1, count, size, dst, trans);
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

        template <bool align> void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(src2) : Aligned(size)) && Aligned(src0) && Aligned(src1) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward8<align>(src0, src1, src2, dst, i + 0 * F);
                            SynetFusedLayerForward8<align>(src0, src1, src2, dst, i + 1 * F);
                            SynetFusedLayerForward8<align>(src0, src1, src2, dst, i + 2 * F);
                            SynetFusedLayerForward8<align>(src0, src1, src2, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward8<align>(src0, src1, src2, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward8(src0[i], src1[i], src2[i]);
                    src0 += count;
                    src1 += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _src2 = vdupq_n_f32(src2[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward8<align>(src0, src1, _src2, dst, j + 0 * F);
                            SynetFusedLayerForward8<align>(src0, src1, _src2, dst, j + 1 * F);
                            SynetFusedLayerForward8<align>(src0, src1, _src2, dst, j + 2 * F);
                            SynetFusedLayerForward8<align>(src0, src1, _src2, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward8<align>(src0, src1, _src2, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward8(src0[j], src1[j], src2[i]);
                    src0 += size;
                    src1 += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(src2) : Aligned(size)) && Aligned(src0) && Aligned(src1) && Aligned(dst))
                SynetFusedLayerForward8<true>(src0, src1, src2, count, size, dst, trans);
            else
                SynetFusedLayerForward8<false>(src0, src1, src2, count, size, dst, trans);
        }

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
