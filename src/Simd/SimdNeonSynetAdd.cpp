/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(float32x4_t bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), bias));
        }

        template <bool align> void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetAddBias<align>(_bias, dst + s + F * 0);
                        SynetAddBias<align>(_bias, dst + s + F * 1);
                        SynetAddBias<align>(_bias, dst + s + F * 2);
                        SynetAddBias<align>(_bias, dst + s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetAddBias<align>(_bias, dst + s);
                }
                for (; s < spatial; ++s)
                    dst[s] += bias[c];
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(spatial, F) && Aligned(dst))
                SynetAddBiasNchw<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(channels, F) && Aligned(bias) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetAddBias<align>(bias + c + F * 0, dst + c + F * 0);
                        SynetAddBias<align>(bias + c + F * 1, dst + c + F * 1);
                        SynetAddBias<align>(bias + c + F * 2, dst + c + F * 2);
                        SynetAddBias<align>(bias + c + F * 3, dst + c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetAddBias<align>(bias + c, dst + c);
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        SIMD_INLINE void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetAddBiasNhwc<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNhwc<false>(bias, channels, spatial, dst);
        }

        void SynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<int part> SIMD_INLINE float32x4_t Cvt8uTo32f(uint8x16_t src)
        {
            return vcvtq_f32_u32(vmovl_u16(Half<part % 2>(vmovl_u8(Half<part / 2>(src)))));
        }

        template<int part> SIMD_INLINE int32x4_t SynetAdd8iNchwCore(uint8x16_t a, uint8x16_t b, float32x4_t scale[3], float32x4_t shift[3])
        {
            float32x4_t _a = vmlaq_f32(shift[0], Cvt8uTo32f<part>(a), scale[0]);
            float32x4_t _b = vmlaq_f32(shift[1], Cvt8uTo32f<part>(b), scale[1]);
            return Round(vmlaq_f32(shift[2], vaddq_f32(_a, _b), scale[2]));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNchwA(const uint8_t* a, const uint8_t* b, float32x4_t scale[3], float32x4_t shift[3], uint8x16_t upper, uint8_t* c, size_t offset)
        {
            uint8x16_t _a = Load<align>(a + offset);
            uint8x16_t _b = Load<align>(b + offset);
            int32x4_t c0 = SynetAdd8iNchwCore<0>(_a, _b, scale, shift);
            int32x4_t c1 = SynetAdd8iNchwCore<1>(_a, _b, scale, shift);
            int32x4_t c2 = SynetAdd8iNchwCore<2>(_a, _b, scale, shift);
            int32x4_t c3 = SynetAdd8iNchwCore<3>(_a, _b, scale, shift);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(c0), vmovn_s32(c1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(c2), vmovn_s32(c3)));
            Store<align>(c + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b, float32x4_t scale[3], float32x4_t shift[3], uint8x8_t upper, uint8_t* c, size_t offset)
        {
            uint8x16_t _a = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(a + offset)));
            uint8x16_t _b = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(b + offset)));
            int32x4_t c0 = SynetAdd8iNchwCore<0>(_a, _b, scale, shift);
            uint8x8_t result = vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(c0), vcreate_s16(0))), upper);
            *(uint32_t*)(c + offset) = vget_lane_u32(vreinterpret_u32_u8(result), 0);
        }

        template <bool align> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(aData) && Aligned(bData) && Aligned(cData) && Aligned(spatial, A));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            uint8x16_t _upperA = vdupq_n_u8(upper);
            uint8x8_t _upperF = vdup_n_u8(upper);
            float32x4_t scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = vdupq_n_f32(aScale[c]);
                    shift[0] = vdupq_n_f32(aShift[c]);
                    scale[1] = vdupq_n_f32(bScale[c]);
                    shift[1] = vdupq_n_f32(bShift[c]);
                    scale[2] = vdupq_n_f32(cScale[c]);
                    shift[2] = vdupq_n_f32(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        SynetAdd8iNchwA<align>(aData, bData, scale, shift, _upperA, cData, s);
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF(aData, bData, scale, shift, _upperF, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF(aData, bData, scale, shift, _upperF, cData, spatial - F);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        SIMD_INLINE void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aData) && Aligned(bData) && Aligned(cData) && Aligned(spatial, A))
                SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        template<int part, bool align> SIMD_INLINE int32x4_t SynetAdd8iNhwcCore(uint8x16_t a, const float* aScale, const float* aShift,
            uint8x16_t b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, size_t offset)
        {
            float32x4_t _a = vmlaq_f32(Load<align>(aShift + offset), Cvt8uTo32f<part>(a), Load<align>(aScale + offset));
            float32x4_t _b = vmlaq_f32(Load<align>(bShift + offset), Cvt8uTo32f<part>(b), Load<align>(bScale + offset));
            return Round(vmlaq_f32(Load<align>(cShift + offset), vaddq_f32(_a, _b), Load<align>(cScale + offset)));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNhwcA(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, uint8x16_t upper, uint8_t* c, size_t offset)
        {
            uint8x16_t _a = Load<false>(a + offset);
            uint8x16_t _b = Load<false>(b + offset);
            int32x4_t c0 = SynetAdd8iNhwcCore<0, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            int32x4_t c1 = SynetAdd8iNhwcCore<1, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 1 * F);
            int32x4_t c2 = SynetAdd8iNhwcCore<2, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 2 * F);
            int32x4_t c3 = SynetAdd8iNhwcCore<3, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 3 * F);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vmovn_s32(c0), vmovn_s32(c1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vmovn_s32(c2), vmovn_s32(c3)));
            Store<false>(c + offset, vminq_u8(vcombine_u8(lo, hi), upper));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, uint8x8_t upper, uint8_t* c, size_t offset)
        {
            uint8x16_t _a = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(a + offset)));
            uint8x16_t _b = vreinterpretq_u8_u32(vdupq_n_u32(*(const uint32_t*)(b + offset)));
            int32x4_t c0 = SynetAdd8iNhwcCore<0, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            uint8x8_t result = vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(c0), vcreate_s16(0))), upper);
            *(uint32_t*)(c + offset) = vget_lane_u32(vreinterpret_u32_u8(result), 0);
        }

        template <bool align> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            uint8x16_t _upperA = vdupq_n_u8(upper);
            uint8x8_t _upperF = vdup_n_u8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        SynetAdd8iNhwcA<align>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upperA, cData, c);
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upperF, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<false>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upperF, cData, channels - F);
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        SIMD_INLINE void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift))
                SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            if (format == SimdTensorFormatNchw && spatial >= F)
                SynetAdd8iNchw(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else if (format == SimdTensorFormatNhwc && channels >= F)
                SynetAdd8iNhwc(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                Base::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }
    }
#endif// SIMD_NEON_ENABLE
}
