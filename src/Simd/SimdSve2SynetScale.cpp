/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        template<bool nofma> SIMD_INLINE svfloat32_t SynetScaleLayerForward(const svfloat32_t& src, const svfloat32_t& scale, const svfloat32_t& bias, const svbool_t& mask)
        {
            if (nofma)
                return svadd_f32_x(mask, svmul_f32_x(mask, src, scale), bias);
            else
                return svmla_f32_x(mask, bias, src, scale);
        }

        template<bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const svfloat32_t& scale, const svfloat32_t& bias, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, SynetScaleLayerForward<nofma>(svld1_f32(mask, src), scale, bias, mask));
        }

        template<bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, const svbool_t& mask)
        {
            SynetScaleLayerForward<nofma>(src, svld1_f32(mask, scale), svld1_f32(mask, bias), dst, mask);
        }

        SIMD_INLINE void SynetScaleLayerForward(const float* src, const svfloat32_t& scale, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svmul_f32_x(mask, svld1_f32(mask, src), scale));
        }

        SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, const svbool_t& mask)
        {
            SynetScaleLayerForward(src, svld1_f32(mask, scale), dst, mask);
        }

        template<bool nofma, bool notail> void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    svfloat32_t _bias = svdup_n_f32(bias[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        SynetScaleLayerForward<nofma>(src + s + 0 * F, _scale, _bias, dst + s + 0 * F, full);
                        SynetScaleLayerForward<nofma>(src + s + 1 * F, _scale, _bias, dst + s + 1 * F, full);
                        SynetScaleLayerForward<nofma>(src + s + 2 * F, _scale, _bias, dst + s + 2 * F, full);
                        SynetScaleLayerForward<nofma>(src + s + 3 * F, _scale, _bias, dst + s + 3 * F, full);
                    }
                    for (; s + F <= spatial; s += F)
                        SynetScaleLayerForward<nofma>(src + s, _scale, _bias, dst + s, full);
                    if (s < spatial)
                    {
                        svbool_t tail = svwhilelt_b32(s, spatial);
                        SynetScaleLayerForward<nofma || notail>(src + s, _scale, _bias, dst + s, tail);
                    }
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        SynetScaleLayerForward(src + s + 0 * F, _scale, dst + s + 0 * F, full);
                        SynetScaleLayerForward(src + s + 1 * F, _scale, dst + s + 1 * F, full);
                        SynetScaleLayerForward(src + s + 2 * F, _scale, dst + s + 2 * F, full);
                        SynetScaleLayerForward(src + s + 3 * F, _scale, dst + s + 3 * F, full);
                    }
                    for (; s < spatial; s += F)
                        SynetScaleLayerForward(src + s, _scale, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility))
                SynetScaleLayerForwardNchw<true, true>(src, scale, bias, channels, spatial, dst);
            else if (Base::FmaNoTail(compatibility))
                SynetScaleLayerForwardNchw<false, true>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNchw<false, false>(src, scale, bias, channels, spatial, dst);
        }

        template<bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        SynetScaleLayerForward<nofma>(src + c + 0 * F, scale + c + 0 * F, bias + c + 0 * F, dst + c + 0 * F, full);
                        SynetScaleLayerForward<nofma>(src + c + 1 * F, scale + c + 1 * F, bias + c + 1 * F, dst + c + 1 * F, full);
                        SynetScaleLayerForward<nofma>(src + c + 2 * F, scale + c + 2 * F, bias + c + 2 * F, dst + c + 2 * F, full);
                        SynetScaleLayerForward<nofma>(src + c + 3 * F, scale + c + 3 * F, bias + c + 3 * F, dst + c + 3 * F, full);
                    }
                    for (; c + F <= channels; c += F)
                        SynetScaleLayerForward<nofma>(src + c, scale + c, bias + c, dst + c, full);
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        SynetScaleLayerForward<nofma || notail>(src + c, scale + c, bias + c, dst + c, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        SynetScaleLayerForward(src + c + 0 * F, scale + c + 0 * F, dst + c + 0 * F, full);
                        SynetScaleLayerForward(src + c + 1 * F, scale + c + 1 * F, dst + c + 1 * F, full);
                        SynetScaleLayerForward(src + c + 2 * F, scale + c + 2 * F, dst + c + 2 * F, full);
                        SynetScaleLayerForward(src + c + 3 * F, scale + c + 3 * F, dst + c + 3 * F, full);
                    }
                    for (; c < channels; c += F)
                        SynetScaleLayerForward(src + c, scale + c, dst + c, svwhilelt_b32(c, channels));
                    src += channels;
                    dst += channels;
                }
            }
        }

        void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility))
                SynetScaleLayerForwardNhwc<true, true>(src, scale, bias, channels, spatial, dst);
            else if (Base::FmaNoTail(compatibility))
                SynetScaleLayerForwardNhwc<false, true>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNhwc<false, false>(src, scale, bias, channels, spatial, dst);
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, spatial, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, spatial, dst, compatibility);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------
        // SynetScale8i
        //-------------------------------------------------------------------------------------------------

        template <bool nofma> SIMD_INLINE svfloat32_t Scale8iFmadd(const svfloat32_t& src, const svfloat32_t& scale, const svfloat32_t& shift, const svbool_t& mask)
        {
            if (nofma)
                return svadd_f32_x(mask, svmul_f32_x(mask, src, scale), shift);
            else
                return svmla_f32_x(mask, shift, src, scale);
        }

        SIMD_INLINE svint32_t Scale8iRound(const svfloat32_t& value, const svbool_t& mask)
        {
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, value, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round));
        }

        template <bool nofma> SIMD_INLINE svuint32_t Scale8iTo8u(const svfloat32_t& src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, const svbool_t& mask)
        {
            svint32_t dst = Scale8iRound(Scale8iFmadd<nofma>(src, scale, shift, mask), mask);
            return svreinterpret_u32_s32(svmin_n_s32_x(mask, svmax_n_s32_x(mask, dst, 0), upper));
        }

        template <bool nofma> SIMD_INLINE void Scale8i(const uint8_t* src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, uint8_t* dst, const svbool_t& mask)
        {
            svfloat32_t _src = svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
            svst1b_u32(mask, dst, Scale8iTo8u<nofma>(_src, scale, shift, upper, mask));
        }

        template <bool nofma> SIMD_INLINE void Scale8i(const uint8_t* src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, float* dst, const svbool_t& mask)
        {
            svfloat32_t _src = svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
            svst1_f32(mask, dst, Scale8iFmadd<nofma>(_src, scale, shift, mask));
        }

        template <bool nofma> SIMD_INLINE void Scale8i(const float* src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, uint8_t* dst, const svbool_t& mask)
        {
            svst1b_u32(mask, dst, Scale8iTo8u<nofma>(svld1_f32(mask, src), scale, shift, upper, mask));
        }

        template <bool nofma, class S, class D> void Scale8iNchw(const S* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, D* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    svfloat32_t _shift = svdup_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        Scale8i<nofma>(src + s + 0 * F, _scale, _shift, upper, dst + s + 0 * F, full);
                        Scale8i<nofma>(src + s + 1 * F, _scale, _shift, upper, dst + s + 1 * F, full);
                        Scale8i<nofma>(src + s + 2 * F, _scale, _shift, upper, dst + s + 2 * F, full);
                        Scale8i<nofma>(src + s + 3 * F, _scale, _shift, upper, dst + s + 3 * F, full);
                    }
                    for (; s + F <= spatial; s += F)
                        Scale8i<nofma>(src + s, _scale, _shift, upper, dst + s, full);
                    if (s < spatial)
                        Scale8i<nofma>(src + s, _scale, _shift, upper, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool nofma, class S, class D> void Scale8iNhwc(const S* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, D* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        Scale8i<nofma>(src + c + 0 * F, svld1_f32(full, scale + c + 0 * F), svld1_f32(full, shift + c + 0 * F), upper, dst + c + 0 * F, full);
                        Scale8i<nofma>(src + c + 1 * F, svld1_f32(full, scale + c + 1 * F), svld1_f32(full, shift + c + 1 * F), upper, dst + c + 1 * F, full);
                        Scale8i<nofma>(src + c + 2 * F, svld1_f32(full, scale + c + 2 * F), svld1_f32(full, shift + c + 2 * F), upper, dst + c + 2 * F, full);
                        Scale8i<nofma>(src + c + 3 * F, svld1_f32(full, scale + c + 3 * F), svld1_f32(full, shift + c + 3 * F), upper, dst + c + 3 * F, full);
                    }
                    for (; c + F <= channels; c += F)
                        Scale8i<nofma>(src + c, svld1_f32(full, scale + c), svld1_f32(full, shift + c), upper, dst + c, full);
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        Scale8i<nofma>(src + c, svld1_f32(tail, scale + c), svld1_f32(tail, shift + c), upper, dst + c, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma, class S, class D> void Scale8i(const S* src, const Base::Scale8iParam& p, const float* scale, const float* shift, int upper, D* dst)
        {
            if (p.format == SimdTensorFormatNchw)
                Scale8iNchw<nofma>(src, scale, shift, p.batch, p.channels, p.spatial, upper, dst);
            else if (p.format == SimdTensorFormatNhwc)
                Scale8iNhwc<nofma>(src, scale, shift, p.batch, p.channels, p.spatial, upper, dst);
            else
                assert(0);
        }

        template <class S, class D> void Scale8i(const S* src, const Base::Scale8iParam& p, const float* scale, const float* shift, int upper, D* dst)
        {
            if (Base::FmaAvoid(p.compatibility))
                Scale8i<true>(src, p, scale, shift, upper, dst);
            else
                Scale8i<false>(src, p, scale, shift, upper, dst);
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            Scale8i(src, _param, _scale.data, _shift.data, _dstCvt.uMax, dst);
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            Scale8i(src, _param, _scale.data, _shift.data, _dstCvt.uMax, dst);
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            Scale8i(src, _param, _scale.data, _shift.data, _dstCvt.uMax, dst);
        }

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

        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Base::SynetScale8i(p)
        {
        }

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            Base::Scale8iParam param(batch, channels, spatial, srcType, dstType, format, compatibility);
            if (!param.Valid())
                return NULL;
            return new Sve2::SynetScale8i(param);
        }
    }
#endif
}
