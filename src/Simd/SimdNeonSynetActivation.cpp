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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Neon
    {
        template<bool align> SIMD_INLINE void SynetElu32f(const float * src, const Neon::Exp & exp, float32x4_t alpha, float * dst, size_t offset)
        {
            Store<align>(dst + offset, exp.Elu(Load<align>(src + offset), alpha));
        }

        template<bool align> void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            float32x4_t _alpha = vdupq_n_f32(alpha[0]);
            Neon::Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<align>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<align>(src, exp, _alpha, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetElu32f(src[i], alpha[0]);
        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetElu32f<true>(src, size, alpha, dst);
            else
                SynetElu32f<false>(src, size, alpha, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetHardSigmoid32f(const float* src, float32x4_t scale, float32x4_t shift, float* dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _dst = SynetHardSigmoid32f(_src, scale, shift);
            Store<align>(dst + offset, _dst);
        }

        template<bool align> void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            float32x4_t _scale = vdupq_n_f32(scale[0]);
            float32x4_t _shift = vdupq_n_f32(shift[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 0 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 1 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 2 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHardSigmoid32f(src[i], scale[0], shift[0]);
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHardSigmoid32f<true>(src, size, scale, shift, dst);
            else
                SynetHardSigmoid32f<false>(src, size, scale, shift, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetHswish32f(const float * src, float32x4_t shift, float32x4_t scale, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _dst = SynetHswish32f(_src, shift, scale);
            Store<align>(dst + offset, _dst);
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            float32x4_t _shift = vdupq_n_f32(shift[0]);
            float32x4_t _scale = vdupq_n_f32(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align>(src, _shift, _scale, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHswish32f(src[i], shift[0], scale[0]);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetMish32f(const float* src, float32x4_t threshold, float* dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst + offset, Mish<1>(_src, threshold));
        }

        template<bool align> void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            float32x4_t _threshold = vdupq_n_f32(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetMish32f<align>(src, _threshold, dst, i + 0 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 1 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 2 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetMish32f<align>(src, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetMish32f(src[i], threshold[0]);
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetMish32f<true>(src, size, threshold, dst);
            else
                SynetMish32f<false>(src, size, threshold, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float* src, const float* slope, float32x4_t _0, float* dst, size_t offset)
        {
            Store<align>(dst + offset, SynetRelu32f(Load<align>(src + offset), Load<align>(slope + offset), _0));
        }

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float* src, float32x4_t slope, float32x4_t _0, float* dst, size_t offset)
        {
            Store<align>(dst + offset, SynetRelu32f(Load<align>(src + offset), slope, _0));
        }

        template <bool align> void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
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
                    float32x4_t _slope = vdupq_n_f32(slope[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 0);
                        SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 1);
                        SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 2);
                        SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetPreluLayerForward<align>(src, _slope, _0, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetRelu32f(src[s], slope[c]);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetPreluLayerForwardNchw<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNchw<false>(src, slope, channels, spatial, dst);
        }

        template <bool align> void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(slope) && Aligned(channels, F) && Aligned(dst));

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
                        SynetPreluLayerForward<align>(src, slope, _0, dst, c + F * 0);
                        SynetPreluLayerForward<align>(src, slope, _0, dst, c + F * 1);
                        SynetPreluLayerForward<align>(src, slope, _0, dst, c + F * 2);
                        SynetPreluLayerForward<align>(src, slope, _0, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetPreluLayerForward<align>(src, slope, _0, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetRelu32f(src[c], slope[c]);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(slope) && Aligned(channels, F) && Aligned(dst))
                SynetPreluLayerForwardNhwc<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNhwc<false>(src, slope, channels, spatial, dst);
        }

        template <bool align> void SynetPreluLayerForwardNchw4c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            float32x4_t _0 = vdupq_n_f32(0.0f);
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _slope = Load<false>(slope + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 0);
                    SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 1);
                    SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 2);
                    SynetPreluLayerForward<align>(src, _slope, _0, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetPreluLayerForward<align>(src, _slope, _0, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNchw4c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetPreluLayerForwardNchw4c<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNchw4c<false>(src, slope, channels, spatial, dst);
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetPreluLayerForwardNchw(src, slope, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetPreluLayerForwardNhwc(src, slope, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetPreluLayerForwardNchw4c(src, slope, channels, spatial, dst);
            else
                Base::SynetPreluLayerForward(src, slope, channels, spatial, dst, format);
        }

        //-------------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetRelu32f(const float* src, float32x4_t slope, float32x4_t zero, float* dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _dst = SynetRelu32f(_src, slope, zero);
            Store<align>(dst + offset, _dst);
        }

        template<bool align> void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float32x4_t _slope = vdupq_n_f32(slope[0]);
            float32x4_t _0 = vdupq_n_f32(0.0f);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetRelu32f<align>(src, _slope, _0, dst, i + 0 * F);
                SynetRelu32f<align>(src, _slope, _0, dst, i + 1 * F);
                SynetRelu32f<align>(src, _slope, _0, dst, i + 2 * F);
                SynetRelu32f<align>(src, _slope, _0, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetRelu32f<align>(src, _slope, _0, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetRelu32f(src[i], slope[0]);
        }

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRelu32f<true>(src, size, slope, dst);
            else
                SynetRelu32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            float32x4_t _min = vdupq_n_f32(min);
            float32x4_t _max = vdupq_n_f32(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, vminq_f32(vmaxq_f32(_min, Load<align>(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange32f<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange32f<false>(src, size, lower, upper, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSigmoid32f(const float* src, const Neon::Exp& exp, float* dst, size_t offset)
        {
            Store<align>(dst + offset, exp.Sigmoid<1>(Load<align>(src + offset)));
        }

        template<bool align> void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSigmoid32f<align>(src, exp, dst, i + 0 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 1 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 2 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSigmoid32f<align>(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSigmoid32f(src[i], slope[0]);
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSigmoid32f<true>(src, size, slope, dst);
            else
                SynetSigmoid32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSoftplus32f(const float* src, float32x4_t beta, float32x4_t threshold, float* dst, size_t offset)
        {
            Store<align>(dst + offset, Softplus<1>(Load<align>(src + offset), beta, threshold));
        }

        template<bool align> void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            float32x4_t _beta = vdupq_n_f32(beta[0]);
            float32x4_t _threshold = vdupq_n_f32(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 0 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 1 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 2 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSoftplus32f(src[i], beta[0], threshold[0]);
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSoftplus32f<true>(src, size, beta, threshold, dst);
            else
                SynetSoftplus32f<false>(src, size, beta, threshold, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSwish32f(const float* src, float32x4_t slope, float* dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            Store<align>(dst + offset, Swish<1>(_src, slope));
        }

        template<bool align> void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float32x4_t _slope = vdupq_n_f32(slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSwish32f<align>(src, _slope, dst, i + 0 * F);
                SynetSwish32f<align>(src, _slope, dst, i + 1 * F);
                SynetSwish32f<align>(src, _slope, dst, i + 2 * F);
                SynetSwish32f<align>(src, _slope, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSwish32f<align>(src, _slope, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSwish32f(src[i], slope[0]);
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSwish32f<true>(src, size, slope, dst);
            else
                SynetSwish32f<false>(src, size, slope, dst);
        }
        
        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetTanh32f(const float* src, const Neon::Exp& exp, float* dst, size_t offset)
        {
            Store<align>(dst + offset, exp.Tanh<1>(Load<align>(src + offset)));
        }

        template<bool align> void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-2.0f*slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetTanh32f<align>(src, exp, dst, i + 0 * F);
                SynetTanh32f<align>(src, exp, dst, i + 1 * F);
                SynetTanh32f<align>(src, exp, dst, i + 2 * F);
                SynetTanh32f<align>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetTanh32f<align>(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetTanh32f(src[i], slope[0]);
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetTanh32f<true>(src, size, slope, dst);
            else
                SynetTanh32f<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}
