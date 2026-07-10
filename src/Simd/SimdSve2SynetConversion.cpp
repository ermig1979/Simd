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

#include "Simd/SimdSve2.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        template <bool nofma> SIMD_INLINE svfloat32_t Fmadd(const svfloat32_t& src, const svfloat32_t& scale, const svfloat32_t& shift, const svbool_t& mask)
        {
            if (nofma)
                return svadd_f32_x(mask, svmul_f32_x(mask, src, scale), shift);
            else
                return svmla_f32_x(mask, shift, src, scale);
        }

        SIMD_INLINE svint32_t Round(const svfloat32_t& value, const svbool_t& mask)
        {
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, value, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round));
        }

        template <bool nofma> SIMD_INLINE svuint32_t SynetConvert32fTo8u(const svfloat32_t& src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, const svbool_t& mask)
        {
            svint32_t dst = Round(Fmadd<nofma>(src, scale, shift, mask), mask);
            dst = svmin_n_s32_x(mask, svmax_n_s32_x(mask, dst, 0), upper);
            return svreinterpret_u32_s32(dst);
        }

        template <bool nofma> SIMD_INLINE void SynetConvert32fTo8u(const float* src, const svfloat32_t& scale, const svfloat32_t& shift, int upper, uint8_t* dst, const svbool_t& mask)
        {
            svst1b_u32(mask, dst, SynetConvert32fTo8u<nofma>(svld1_f32(mask, src), scale, shift, upper, mask));
        }

        template <bool nofma> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    svfloat32_t _shift = svdup_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        SynetConvert32fTo8u<nofma>(src + s + 0 * F, _scale, _shift, upper, dst + s + 0 * F, body);
                        SynetConvert32fTo8u<nofma>(src + s + 1 * F, _scale, _shift, upper, dst + s + 1 * F, body);
                        SynetConvert32fTo8u<nofma>(src + s + 2 * F, _scale, _shift, upper, dst + s + 2 * F, body);
                        SynetConvert32fTo8u<nofma>(src + s + 3 * F, _scale, _shift, upper, dst + s + 3 * F, body);
                    }
                    for (; s + F <= spatial; s += F)
                        SynetConvert32fTo8u<nofma>(src + s, _scale, _shift, upper, dst + s, body);
                    if (s < spatial)
                        SynetConvert32fTo8u<nofma>(src + s, _scale, _shift, upper, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool nofma> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        SynetConvert32fTo8u<nofma>(src + c + 0 * F, svld1_f32(body, scale + c + 0 * F), svld1_f32(body, shift + c + 0 * F), upper, dst + c + 0 * F, body);
                        SynetConvert32fTo8u<nofma>(src + c + 1 * F, svld1_f32(body, scale + c + 1 * F), svld1_f32(body, shift + c + 1 * F), upper, dst + c + 1 * F, body);
                        SynetConvert32fTo8u<nofma>(src + c + 2 * F, svld1_f32(body, scale + c + 2 * F), svld1_f32(body, shift + c + 2 * F), upper, dst + c + 2 * F, body);
                        SynetConvert32fTo8u<nofma>(src + c + 3 * F, svld1_f32(body, scale + c + 3 * F), svld1_f32(body, shift + c + 3 * F), upper, dst + c + 3 * F, body);
                    }
                    for (; c + F <= channels; c += F)
                        SynetConvert32fTo8u<nofma>(src + c, svld1_f32(body, scale + c), svld1_f32(body, shift + c), upper, dst + c, body);
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        SynetConvert32fTo8u<nofma>(src + c, svld1_f32(tail, scale + c), svld1_f32(tail, shift + c), upper, dst + c, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), 3);
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; s += F)
                {
                    svbool_t mask = svwhilelt_b32(s, spatial);
                    const float* ps = src + 3 * s;
                    uint8_t* pd = dst + 3 * s;
                    svst1b_scatter_u32offset_u32(mask, pd + 0, offsets, SynetConvert32fTo8u<nofma>(svld1_gather_u32index_f32(mask, ps + 0, offsets), scale0, shift0, upper, mask));
                    svst1b_scatter_u32offset_u32(mask, pd + 1, offsets, SynetConvert32fTo8u<nofma>(svld1_gather_u32index_f32(mask, ps + 1, offsets), scale1, shift1, upper, mask));
                    svst1b_scatter_u32offset_u32(mask, pd + 2, offsets, SynetConvert32fTo8u<nofma>(svld1_gather_u32index_f32(mask, ps + 2, offsets), scale2, shift2, upper, mask));
                }
                src += 3 * spatial;
                dst += 3 * spatial;
            }
        }

        template <bool nofma> void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetConvert32fTo8uNchw<nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                    SynetConvert32fTo8uNhwc3<nofma>(src, batch, spatial, scale, shift, upper, dst);
                else
                    SynetConvert32fTo8uNhwc<nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
            }
            else
                assert(0);
        }

        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            if (Base::FmaAvoid(compatibility))
                SynetConvert32fTo8u<true>(src, batch, channels, height, width, format, scale, shift, upper, dst);
            else
                SynetConvert32fTo8u<false>(src, batch, channels, height, width, format, scale, shift, upper, dst);
        }
    }
#endif
}
