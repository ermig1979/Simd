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
#include "Simd/SimdMemory.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        //-------------------------------------------------------------------------------------------------
        // uint8 -> uint8
        //-------------------------------------------------------------------------------------------------

        template<int part> SIMD_INLINE float32x4_t Cvt8uTo32f(uint8x16_t src)
        {
            return vcvtq_f32_u32(vmovl_u16(Half<part % 2>(vmovl_u8(Half<part / 2>(src)))));
        }

        template <bool align> SIMD_INLINE void ScaleNchwA(const uint8_t* src, float32x4_t scale, float32x4_t shift, uint8x16_t upper, uint8_t* dst, size_t offset)
        {
            uint8x16_t _src = Load<align>(src + offset);
            int32x4_t d0 = Round(vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
            int32x4_t d1 = Round(vmlaq_f32(shift, Cvt8uTo32f<1>(_src), scale));
            int32x4_t d2 = Round(vmlaq_f32(shift, Cvt8uTo32f<2>(_src), scale));
            int32x4_t d3 = Round(vmlaq_f32(shift, Cvt8uTo32f<3>(_src), scale));
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(d0), vmovn_s32(d1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(d2), vmovn_s32(d3)));
            Store<align>(dst + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        SIMD_INLINE void ScaleNchwF(const uint8_t* src, float32x4_t scale, float32x4_t shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            int32x4_t d0 = Round(vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
            uint8x8_t result = vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(result), 0);
        }

        template <bool align> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            uint8x16_t _upper = vdupq_n_u8((uint8_t)upper);
            uint8x8_t _upper8 = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF(src, _scale, _shift, _upper8, dst, s);
                    if (s < spatial)
                        ScaleNchwF(src, _scale, _shift, _upper8, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template<int part, bool align> SIMD_INLINE int32x4_t ScaleNhwcFi(uint8x16_t value, const float* scale, const float* shift)
        {
            return Round(vmlaq_f32(Load<align>(shift + part * F), Cvt8uTo32f<part>(value), Load<align>(scale + part * F)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const uint8_t* src, const float* scale, const float* shift, uint8x16_t upper, uint8_t* dst, size_t offset)
        {
            uint8x16_t _src = vld1q_u8(src + offset);
            int32x4_t d0 = ScaleNhwcFi<0, align>(_src, scale + offset, shift + offset);
            int32x4_t d1 = ScaleNhwcFi<1, align>(_src, scale + offset, shift + offset);
            int32x4_t d2 = ScaleNhwcFi<2, align>(_src, scale + offset, shift + offset);
            int32x4_t d3 = ScaleNhwcFi<3, align>(_src, scale + offset, shift + offset);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(d0), vmovn_s32(d1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(d2), vmovn_s32(d3)));
            vst1q_u8(dst + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            int32x4_t d0 = ScaleNhwcFi<0, false>(_src, scale + offset, shift + offset);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper)), 0);
        }

        template <bool align> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            uint8x16_t _upper = vdupq_n_u8((uint8_t)upper);
            uint8x8_t _upper8 = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        ScaleNhwcA<align>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF(src, scale, shift, _upper8, dst, c);
                    if (c < channels)
                        ScaleNhwcF(src, scale, shift, _upper8, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const uint8_t* src, float32x4_t scale, float32x4_t shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            int32x4_t d0 = Round(vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
            uint8x8_t result = vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(result), 0);
        }

        void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            float32x4_t _scale0 = Load<false>(_scale + 0 * F);
            float32x4_t _scale1 = Load<false>(_scale + 1 * F);
            float32x4_t _scale2 = Load<false>(_scale + 2 * F);
            float32x4_t _shift0 = Load<false>(_shift + 0 * F);
            float32x4_t _shift1 = Load<false>(_shift + 1 * F);
            float32x4_t _shift2 = Load<false>(_shift + 2 * F);
            uint8x8_t _upper = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //-------------------------------------------------------------------------------------------------
        // uint8 -> float32
        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void ScaleNchwA(const uint8_t* src, float32x4_t scale, float32x4_t shift, float* dst, size_t offset)
        {
            uint8x16_t _src = Load<align>(src + offset);
            Store<align>(dst + offset + 0 * F, vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
            Store<align>(dst + offset + 1 * F, vmlaq_f32(shift, Cvt8uTo32f<1>(_src), scale));
            Store<align>(dst + offset + 2 * F, vmlaq_f32(shift, Cvt8uTo32f<2>(_src), scale));
            Store<align>(dst + offset + 3 * F, vmlaq_f32(shift, Cvt8uTo32f<3>(_src), scale));
        }

        SIMD_INLINE void ScaleNchwF(const uint8_t* src, float32x4_t scale, float32x4_t shift, float* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            Store<false>(dst + offset + 0 * F, vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
        }

        template <bool align> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF(src, _scale, _shift, dst, s);
                    if (s < spatial)
                        ScaleNchwF(src, _scale, _shift, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, dst);
        }

        template<int part, bool align> SIMD_INLINE void ScaleNhwcF(uint8x16_t value, const float* scale, const float* shift, float* dst)
        {
            Store<false>(dst + part * F, vmlaq_f32(Load<align>(shift + part * F), Cvt8uTo32f<part>(value), Load<align>(scale + part * F)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            uint8x16_t _src = vld1q_u8(src + offset);
            ScaleNhwcF<0, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<1, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<2, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<3, align>(_src, scale + offset, shift + offset, dst + offset);
        }

        SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            ScaleNhwcF<0, false>(_src, scale + offset, shift + offset, dst + offset);
        }

        template <bool align> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        ScaleNhwcA<align>(src, scale, shift, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF(src, scale, shift, dst, c);
                    if (c < channels)
                        ScaleNhwcF(src, scale, shift, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const uint8_t* src, float32x4_t scale, float32x4_t shift, float* dst, size_t offset)
        {
            uint8x16_t _src = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(src + offset)));
            Store<false>(dst + offset, vmlaq_f32(shift, Cvt8uTo32f<0>(_src), scale));
        }

        void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, float* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            float32x4_t _scale0 = Load<false>(_scale + 0 * F);
            float32x4_t _scale1 = Load<false>(_scale + 1 * F);
            float32x4_t _scale2 = Load<false>(_scale + 2 * F);
            float32x4_t _shift0 = Load<false>(_shift + 0 * F);
            float32x4_t _shift1 = Load<false>(_shift + 1 * F);
            float32x4_t _shift2 = Load<false>(_shift + 2 * F);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F && 0)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //-------------------------------------------------------------------------------------------------
        // float32 -> uint8
        //-------------------------------------------------------------------------------------------------

        template<bool align> SIMD_INLINE int32x4_t ScaleNhwcFf(const float* src, float32x4_t scale, float32x4_t shift, size_t offset)
        {
            return Round(vmlaq_f32(shift, Load<align>(src + offset), scale));
        }

        template <bool align> SIMD_INLINE void ScaleNchwA(const float* src, float32x4_t scale, float32x4_t shift, uint8x16_t upper, uint8_t* dst, size_t offset)
        {
            int32x4_t d0 = ScaleNhwcFf<align>(src, scale, shift, offset + 0 * F);
            int32x4_t d1 = ScaleNhwcFf<align>(src, scale, shift, offset + 1 * F);
            int32x4_t d2 = ScaleNhwcFf<align>(src, scale, shift, offset + 2 * F);
            int32x4_t d3 = ScaleNhwcFf<align>(src, scale, shift, offset + 3 * F);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(d0), vmovn_s32(d1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(d2), vmovn_s32(d3)));
            Store<align>(dst + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        template <bool align> SIMD_INLINE void ScaleNchwF(const float* src, float32x4_t scale, float32x4_t shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            int32x4_t d0 = ScaleNhwcFf<align>(src, scale, shift, offset);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper)), 0);
        }

        template <bool align> void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            uint8x16_t _upper = vdupq_n_u8((uint8_t)upper);
            uint8x8_t _upper8 = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF<align>(src, _scale, _shift, _upper8, dst, s);
                    if (s < spatial)
                        ScaleNchwF<false>(src, _scale, _shift, _upper8, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template<bool align> SIMD_INLINE int32x4_t ScaleNhwcFf(const float* src, const float* scale, const float* shift, size_t offset)
        {
            return Round(vmlaq_f32(Load<align>(shift + offset), Load<false>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const float* src, const float* scale, const float* shift, uint8x16_t upper, uint8_t* dst, size_t offset)
        {
            int32x4_t d0 = ScaleNhwcFf<align>(src, scale, shift, offset + 0 * F);
            int32x4_t d1 = ScaleNhwcFf<align>(src, scale, shift, offset + 1 * F);
            int32x4_t d2 = ScaleNhwcFf<align>(src, scale, shift, offset + 2 * F);
            int32x4_t d3 = ScaleNhwcFf<align>(src, scale, shift, offset + 3 * F);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(d0), vmovn_s32(d1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(d2), vmovn_s32(d3)));
            vst1q_u8(dst + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcF(const float* src, const float* scale, const float* shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            int32x4_t d0 = ScaleNhwcFf<align>(src, scale, shift, offset + 0 * F);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper)), 0);
        }

        template <bool align> void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            uint8x16_t _upper = vdupq_n_u8((uint8_t)upper);
            uint8x8_t _upper8 = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        ScaleNhwcA<align>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align>(src, scale, shift, _upper8, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false>(src, scale, shift, _upper8, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const float* src, float32x4_t scale, float32x4_t shift, uint8x8_t upper, uint8_t* dst, size_t offset)
        {
            int32x4_t d0 = ScaleNhwcFf<false>(src, scale, shift, offset + 0 * F);
            *(uint32_t*)(dst + offset) = vget_lane_u32(vreinterpret_u32_u8(vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(d0), vcreate_s16(0))), upper)), 0);
        }

        void ScaleNhwc3(const float* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            float32x4_t _scale0 = Load<false>(_scale + 0 * F);
            float32x4_t _scale1 = Load<false>(_scale + 1 * F);
            float32x4_t _scale2 = Load<false>(_scale + 2 * F);
            float32x4_t _shift0 = Load<false>(_shift + 0 * F);
            float32x4_t _shift1 = Load<false>(_shift + 1 * F);
            float32x4_t _shift2 = Load<false>(_shift + 2 * F);
            uint8x8_t _upper = vdup_n_u8((uint8_t)upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //-------------------------------------------------------------------------------------------------
        // float32 -> float32
        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Base::SynetScale8i(p)
        {
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            Base::Scale8iParam param(batch, channels, spatial, srcType, dstType, format, compatibility);
            if (!param.Valid())
                return NULL;
            return new Neon::SynetScale8i(param);
        }
    }
#endif// SIMD_NEON_ENABLE
}
