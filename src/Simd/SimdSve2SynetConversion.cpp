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
#include "Simd/SimdConversion.h"

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

        SIMD_INLINE svfloat32_t RepeatNhwc3(const svfloat32_t& table, uint32_t start, const svbool_t& mask)
        {
            svuint32_t index = svindex_u32(start, 1);
            svuint32_t rem = svmls_n_u32_x(mask, index, svdiv_n_u32_x(mask, index, 3), 3);
            return svreinterpret_f32_u32(svtbl_u32(svreinterpret_u32_f32(table), rem));
        }

        template <bool nofma> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            const svbool_t channels = svwhilelt_b32((uint64_t)0, (uint64_t)3);
            svfloat32_t scaleTbl = svld1_f32(channels, scale);
            svfloat32_t shiftTbl = svld1_f32(channels, shift);
            svfloat32_t scale0 = RepeatNhwc3(scaleTbl, 0, body);
            svfloat32_t scale1 = RepeatNhwc3(scaleTbl, (uint32_t)F, body);
            svfloat32_t scale2 = RepeatNhwc3(scaleTbl, (uint32_t)(F * 2), body);
            svfloat32_t shift0 = RepeatNhwc3(shiftTbl, 0, body);
            svfloat32_t shift1 = RepeatNhwc3(shiftTbl, (uint32_t)F, body);
            svfloat32_t shift2 = RepeatNhwc3(shiftTbl, (uint32_t)(F * 2), body);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s + DF <= spatial; s += DF)
                {
                    const float* ps = src + s * 3;
                    uint8_t* pd = dst + s * 3;
                    SynetConvert32fTo8u<nofma>(ps + 0 * F, scale0, shift0, upper, pd + 0 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 1 * F, scale1, shift1, upper, pd + 1 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 2 * F, scale2, shift2, upper, pd + 2 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 3 * F, scale0, shift0, upper, pd + 3 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 4 * F, scale1, shift1, upper, pd + 4 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 5 * F, scale2, shift2, upper, pd + 5 * F, body);
                }
                for (; s + F <= spatial; s += F)
                {
                    const float* ps = src + s * 3;
                    uint8_t* pd = dst + s * 3;
                    SynetConvert32fTo8u<nofma>(ps + 0 * F, scale0, shift0, upper, pd + 0 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 1 * F, scale1, shift1, upper, pd + 1 * F, body);
                    SynetConvert32fTo8u<nofma>(ps + 2 * F, scale2, shift2, upper, pd + 2 * F, body);
                }
                if (s < spatial)
                {
                    size_t tail = (spatial - s) * 3;
                    const float* ps = src + s * 3;
                    uint8_t* pd = dst + s * 3;
                    SynetConvert32fTo8u<nofma>(ps + 0 * F, scale0, shift0, upper, pd + 0 * F, svwhilelt_b32((size_t)0, tail));
                    if (tail > F)
                        SynetConvert32fTo8u<nofma>(ps + 1 * F, scale1, shift1, upper, pd + 1 * F, svwhilelt_b32(F, tail));
                    if (tail > F * 2)
                        SynetConvert32fTo8u<nofma>(ps + 2 * F, scale2, shift2, upper, pd + 2 * F, svwhilelt_b32(F * 2, tail));
                }
                src += spatial * 3;
                dst += spatial * 3;
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

        //-------------------------------------------------------------------------------------------------

        template <bool nofma> SIMD_INLINE void SynetConvert8uTo32f(const uint8_t* src, const svfloat32_t& scale, const svfloat32_t& shift, float* dst, const svbool_t& mask)
        {
            svfloat32_t f32 = svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
            svst1_f32(mask, dst, Fmadd<nofma>(f32, scale, shift, mask));
        }

        template <bool nofma> SIMD_INLINE void SynetConvert8uTo32f(const uint8_t* src, const float* scale, const float* shift, float* dst, const svbool_t& mask)
        {
            SynetConvert8uTo32f<nofma>(src, svld1_f32(mask, scale), svld1_f32(mask, shift), dst, mask);
        }

        template <bool nofma> void SynetConvert8uTo32fNchw(const uint8_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float* dst)
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
                        SynetConvert8uTo32f<nofma>(src + s + 0 * F, _scale, _shift, dst + s + 0 * F, body);
                        SynetConvert8uTo32f<nofma>(src + s + 1 * F, _scale, _shift, dst + s + 1 * F, body);
                        SynetConvert8uTo32f<nofma>(src + s + 2 * F, _scale, _shift, dst + s + 2 * F, body);
                        SynetConvert8uTo32f<nofma>(src + s + 3 * F, _scale, _shift, dst + s + 3 * F, body);
                    }
                    for (; s + F <= spatial; s += F)
                        SynetConvert8uTo32f<nofma>(src + s, _scale, _shift, dst + s, body);
                    if (s < spatial)
                        SynetConvert8uTo32f<nofma>(src + s, _scale, _shift, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool nofma> void SynetConvert8uTo32fNhwc(const uint8_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float* dst)
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
                        SynetConvert8uTo32f<nofma>(src + c + 0 * F, scale + c + 0 * F, shift + c + 0 * F, dst + c + 0 * F, body);
                        SynetConvert8uTo32f<nofma>(src + c + 1 * F, scale + c + 1 * F, shift + c + 1 * F, dst + c + 1 * F, body);
                        SynetConvert8uTo32f<nofma>(src + c + 2 * F, scale + c + 2 * F, shift + c + 2 * F, dst + c + 2 * F, body);
                        SynetConvert8uTo32f<nofma>(src + c + 3 * F, scale + c + 3 * F, shift + c + 3 * F, dst + c + 3 * F, body);
                    }
                    for (; c + F <= channels; c += F)
                        SynetConvert8uTo32f<nofma>(src + c, scale + c, shift + c, dst + c, body);
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        SynetConvert8uTo32f<nofma>(src + c, scale + c, shift + c, dst + c, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> void SynetConvert8uTo32fNhwc3(const uint8_t* src, size_t batch, size_t spatial, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            const svbool_t channels = svwhilelt_b32((uint64_t)0, (uint64_t)3);
            svfloat32_t scaleTbl = svld1_f32(channels, scale);
            svfloat32_t shiftTbl = svld1_f32(channels, shift);
            svfloat32_t scale0 = RepeatNhwc3(scaleTbl, 0, body);
            svfloat32_t scale1 = RepeatNhwc3(scaleTbl, (uint32_t)F, body);
            svfloat32_t scale2 = RepeatNhwc3(scaleTbl, (uint32_t)(F * 2), body);
            svfloat32_t shift0 = RepeatNhwc3(shiftTbl, 0, body);
            svfloat32_t shift1 = RepeatNhwc3(shiftTbl, (uint32_t)F, body);
            svfloat32_t shift2 = RepeatNhwc3(shiftTbl, (uint32_t)(F * 2), body);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s + DF <= spatial; s += DF)
                {
                    const uint8_t* ps = src + s * 3;
                    float* pd = dst + s * 3;
                    SynetConvert8uTo32f<nofma>(ps + 0 * F, scale0, shift0, pd + 0 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 1 * F, scale1, shift1, pd + 1 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 2 * F, scale2, shift2, pd + 2 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 3 * F, scale0, shift0, pd + 3 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 4 * F, scale1, shift1, pd + 4 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 5 * F, scale2, shift2, pd + 5 * F, body);
                }
                for (; s + F <= spatial; s += F)
                {
                    const uint8_t* ps = src + s * 3;
                    float* pd = dst + s * 3;
                    SynetConvert8uTo32f<nofma>(ps + 0 * F, scale0, shift0, pd + 0 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 1 * F, scale1, shift1, pd + 1 * F, body);
                    SynetConvert8uTo32f<nofma>(ps + 2 * F, scale2, shift2, pd + 2 * F, body);
                }
                if (s < spatial)
                {
                    size_t tail = (spatial - s) * 3;
                    const uint8_t* ps = src + s * 3;
                    float* pd = dst + s * 3;
                    SynetConvert8uTo32f<nofma>(ps + 0 * F, scale0, shift0, pd + 0 * F, svwhilelt_b32((size_t)0, tail));
                    if (tail > F)
                        SynetConvert8uTo32f<nofma>(ps + 1 * F, scale1, shift1, pd + 1 * F, svwhilelt_b32(F, tail));
                    if (tail > F * 2)
                        SynetConvert8uTo32f<nofma>(ps + 2 * F, scale2, shift2, pd + 2 * F, svwhilelt_b32(F * 2, tail));
                }
                src += spatial * 3;
                dst += spatial * 3;
            }
        }

        template <bool nofma> void SynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetConvert8uTo32fNchw<nofma>(src, batch, channels, spatial, scale, shift, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                    SynetConvert8uTo32fNhwc3<nofma>(src, batch, spatial, scale, shift, dst);
                else
                    SynetConvert8uTo32fNhwc<nofma>(src, batch, channels, spatial, scale, shift, dst);
            }
            else
                assert(0);
        }

        void SynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility))
                SynetConvert8uTo32f<true>(src, batch, channels, height, width, format, scale, shift, dst);
            else
                SynetConvert8uTo32f<false>(src, batch, channels, height, width, format, scale, shift, dst);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void StoreScaled(const svuint32_t& value, const svfloat32_t& scale, const svfloat32_t& shift, float* dst, const svbool_t& mask)
        {
            svfloat32_t f32 = svcvt_f32_u32_x(mask, value);
            svst1_f32(mask, dst, Fmadd<false>(f32, scale, shift, mask));
        }

        SIMD_INLINE void StoreScaled(const svuint32_t& value, const svfloat32_t& scale, const svfloat32_t& shift, float* dst, const svuint32_t& offsets, const svbool_t& mask)
        {
            svfloat32_t f32 = svcvt_f32_u32_x(mask, value);
            svst1_scatter_u32offset_f32(mask, dst, offsets, Fmadd<false>(f32, scale, shift, mask));
        }

        template<SimdPixelFormatType format> SIMD_INLINE svuint32_t LoadBgr(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask);

        template<> SIMD_INLINE svuint32_t LoadBgr<SimdPixelFormatGray8>(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src, offsets);
        }

        template<> SIMD_INLINE svuint32_t LoadBgr<SimdPixelFormatBgr24>(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + channel, offsets);
        }

        template<> SIMD_INLINE svuint32_t LoadBgr<SimdPixelFormatBgra32>(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + channel, offsets);
        }

        template<> SIMD_INLINE svuint32_t LoadBgr<SimdPixelFormatRgb24>(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + 2 - channel, offsets);
        }

        template<> SIMD_INLINE svuint32_t LoadBgr<SimdPixelFormatRgba32>(const uint8_t* src, const svuint32_t& offsets, size_t channel, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + 2 - channel, offsets);
        }

        SIMD_INLINE svuint32_t BgrToGray(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red, const svbool_t& mask)
        {
            svuint32_t gray = svdup_n_u32(Base::BGR_TO_GRAY_ROUND_TERM);
            gray = svmla_n_u32_x(mask, gray, blue, Base::BLUE_TO_GRAY_WEIGHT);
            gray = svmla_n_u32_x(mask, gray, green, Base::GREEN_TO_GRAY_WEIGHT);
            gray = svmla_n_u32_x(mask, gray, red, Base::RED_TO_GRAY_WEIGHT);
            return svlsr_n_u32_x(mask, gray, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template<SimdPixelFormatType format> SIMD_INLINE svuint32_t LoadGray(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return BgrToGray(LoadBgr<format>(src, offsets, 0, mask), LoadBgr<format>(src, offsets, 1, mask), LoadBgr<format>(src, offsets, 2, mask), mask);
        }

        template<> SIMD_INLINE svuint32_t LoadGray<SimdPixelFormatGray8>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src, offsets);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInput1(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t _scale = svdup_n_f32(scale[0]);
            const svfloat32_t _shift = svdup_n_f32(shift[0]);
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), step);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    StoreScaled(LoadGray<format>(src + step * x, offsets, mask), _scale, _shift, dst + x, mask);
                }
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw(), channel = width * height;
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]);
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), step);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    const uint8_t* ps = src + step * x;
                    float* pd = dst + x;
                    StoreScaled(LoadBgr<format>(ps, offsets, 0, mask), scale0, shift0, pd + 0 * channel, mask);
                    StoreScaled(LoadBgr<format>(ps, offsets, 1, mask), scale1, shift1, pd + 1 * channel, mask);
                    StoreScaled(LoadBgr<format>(ps, offsets, 2, mask), scale2, shift2, pd + 2 * channel, mask);
                }
                src += stride;
                dst += width;
            }
        }

        template<> void SynetSetInputNchw3<SimdPixelFormatGray8, 1>(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw(), channel = width * height;
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svuint32_t gray = svld1ub_u32(mask, src + x);
                    float* pd = dst + x;
                    StoreScaled(gray, scale0, shift0, pd + 0 * channel, mask);
                    StoreScaled(gray, scale1, shift1, pd + 1 * channel, mask);
                    StoreScaled(gray, scale2, shift2, pd + 2 * channel, mask);
                }
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]);
            const svuint32_t srcOffsets = svmul_n_u32_x(body, svindex_u32(0, 1), step);
            const svuint32_t dstOffsets = svmul_n_u32_x(body, svindex_u32(0, 1), 3 * sizeof(float));
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    const uint8_t* ps = src + step * x;
                    float* pd = dst + 3 * x;
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 0, mask), scale0, shift0, pd + 0, dstOffsets, mask);
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 1, mask), scale1, shift1, pd + 1, dstOffsets, mask);
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 2, mask), scale2, shift2, pd + 2, dstOffsets, mask);
                }
                src += stride;
                dst += 3 * width;
            }
        }

        template<> void SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]);
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), 3 * sizeof(float));
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svuint32_t gray = svld1ub_u32(mask, src + x);
                    float* pd = dst + 3 * x;
                    StoreScaled(gray, scale0, shift0, pd + 0, offsets, mask);
                    StoreScaled(gray, scale1, shift1, pd + 1, offsets, mask);
                    StoreScaled(gray, scale2, shift2, pd + 2, offsets, mask);
                }
                src += stride;
                dst += 3 * width;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdPixelFormatType format> SIMD_INLINE svuint32_t LoadAlpha(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask);

        template<> SIMD_INLINE svuint32_t LoadAlpha<SimdPixelFormatGray8>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svdup_n_u32(0xFF);
        }

        template<> SIMD_INLINE svuint32_t LoadAlpha<SimdPixelFormatBgr24>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svdup_n_u32(0xFF);
        }

        template<> SIMD_INLINE svuint32_t LoadAlpha<SimdPixelFormatBgra32>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + 3, offsets);
        }

        template<> SIMD_INLINE svuint32_t LoadAlpha<SimdPixelFormatRgb24>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svdup_n_u32(0xFF);
        }

        template<> SIMD_INLINE svuint32_t LoadAlpha<SimdPixelFormatRgba32>(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src + 3, offsets);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw4(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw(), channel = width * height;
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]), scale3 = svdup_n_f32(scale[3]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]), shift3 = svdup_n_f32(shift[3]);
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), step);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    const uint8_t* ps = src + step * x;
                    float* pd = dst + x;
                    StoreScaled(LoadBgr<format>(ps, offsets, 0, mask), scale0, shift0, pd + 0 * channel, mask);
                    StoreScaled(LoadBgr<format>(ps, offsets, 1, mask), scale1, shift1, pd + 1 * channel, mask);
                    StoreScaled(LoadBgr<format>(ps, offsets, 2, mask), scale2, shift2, pd + 2 * channel, mask);
                    StoreScaled(LoadAlpha<format>(ps, offsets, mask), scale3, shift3, pd + 3 * channel, mask);
                }
                src += stride;
                dst += width;
            }
        }

        template<> void SynetSetInputNchw4<SimdPixelFormatGray8, 1>(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw(), channel = width * height;
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]), scale3 = svdup_n_f32(scale[3]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]), shift3 = svdup_n_f32(shift[3]);
            const svuint32_t alpha = svdup_n_u32(0xFF);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svuint32_t gray = svld1ub_u32(mask, src + x);
                    float* pd = dst + x;
                    StoreScaled(gray, scale0, shift0, pd + 0 * channel, mask);
                    StoreScaled(gray, scale1, shift1, pd + 1 * channel, mask);
                    StoreScaled(gray, scale2, shift2, pd + 2 * channel, mask);
                    StoreScaled(alpha, scale3, shift3, pd + 3 * channel, mask);
                }
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc4(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]), scale3 = svdup_n_f32(scale[3]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]), shift3 = svdup_n_f32(shift[3]);
            const svuint32_t srcOffsets = svmul_n_u32_x(body, svindex_u32(0, 1), step);
            const svuint32_t dstOffsets = svmul_n_u32_x(body, svindex_u32(0, 1), 4 * sizeof(float));
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    const uint8_t* ps = src + step * x;
                    float* pd = dst + 4 * x;
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 0, mask), scale0, shift0, pd + 0, dstOffsets, mask);
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 1, mask), scale1, shift1, pd + 1, dstOffsets, mask);
                    StoreScaled(LoadBgr<format>(ps, srcOffsets, 2, mask), scale2, shift2, pd + 2, dstOffsets, mask);
                    StoreScaled(LoadAlpha<format>(ps, srcOffsets, mask), scale3, shift3, pd + 3, dstOffsets, mask);
                }
                src += stride;
                dst += 4 * width;
            }
        }

        template<> void SynetSetInputNhwc4<SimdPixelFormatGray8, 1>(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t scale0 = svdup_n_f32(scale[0]), scale1 = svdup_n_f32(scale[1]), scale2 = svdup_n_f32(scale[2]), scale3 = svdup_n_f32(scale[3]);
            svfloat32_t shift0 = svdup_n_f32(shift[0]), shift1 = svdup_n_f32(shift[1]), shift2 = svdup_n_f32(shift[2]), shift3 = svdup_n_f32(shift[3]);
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), 4 * sizeof(float));
            const svuint32_t alpha = svdup_n_u32(0xFF);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svuint32_t gray = svld1ub_u32(mask, src + x);
                    float* pd = dst + 4 * x;
                    StoreScaled(gray, scale0, shift0, pd + 0, offsets, mask);
                    StoreScaled(gray, scale1, shift1, pd + 1, offsets, mask);
                    StoreScaled(gray, scale2, shift2, pd + 2, offsets, mask);
                    StoreScaled(alpha, scale3, shift3, pd + 3, offsets, mask);
                }
                src += stride;
                dst += 4 * width;
            }
        }

        void SynetSetInput(const uint8_t* src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float* lower, const float* upper, float* dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            float scale[4];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SynetSetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SynetSetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SynetSetInput1<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SynetSetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgba32: SynetSetInput1<SimdPixelFormatRgba32, 4>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgba32: SynetSetInputNchw3<SimdPixelFormatRgba32, 4>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgba32: SynetSetInputNhwc3<SimdPixelFormatRgba32, 4>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
                break;
            case 4:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw4<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw4<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw4<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw4<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgba32: SynetSetInputNchw4<SimdPixelFormatRgba32, 4>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc4<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc4<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc4<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc4<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgba32: SynetSetInputNhwc4<SimdPixelFormatRgba32, 4>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
                break;
            default: assert(0);
            }
        }
    }
#endif
}
