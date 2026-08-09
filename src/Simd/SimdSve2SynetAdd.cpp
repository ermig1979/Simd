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
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void SynetAddBias(const svfloat32_t& bias, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), bias));
        }

        SIMD_INLINE void SynetAddBias(const float* bias, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), svld1_f32(mask, bias)));
        }

        void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                svfloat32_t _bias = svdup_n_f32(bias[c]);
                for (; s + QF <= spatial; s += QF)
                {
                    SynetAddBias(_bias, dst + s + 0 * F, body);
                    SynetAddBias(_bias, dst + s + 1 * F, body);
                    SynetAddBias(_bias, dst + s + 2 * F, body);
                    SynetAddBias(_bias, dst + s + 3 * F, body);
                }
                for (; s + F <= spatial; s += F)
                    SynetAddBias(_bias, dst + s, body);
                if (s < spatial)
                    SynetAddBias(_bias, dst + s, svwhilelt_b32(s, spatial));
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c + QF <= channels; c += QF)
                {
                    SynetAddBias(bias + c + 0 * F, dst + c + 0 * F, body);
                    SynetAddBias(bias + c + 1 * F, dst + c + 1 * F, body);
                    SynetAddBias(bias + c + 2 * F, dst + c + 2 * F, body);
                    SynetAddBias(bias + c + 3 * F, dst + c + 3 * F, body);
                }
                for (; c + F <= channels; c += F)
                    SynetAddBias(bias + c, dst + c, body);
                if (c < channels)
                    SynetAddBias(bias + c, dst + c, svwhilelt_b32(c, channels));
                dst += channels;
            }
        }

        void SynetAddBias(const float* bias, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetAdd8iLoad(const uint8_t* src, const svbool_t& mask)
        {
            return svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
        }

        SIMD_INLINE svuint32_t SynetAdd8iConvert(const svfloat32_t& value, const svfloat32_t& scale,
            const svfloat32_t& shift, float upper, const svbool_t& mask)
        {
            svfloat32_t dst = svmla_f32_x(mask, shift, value, scale);
            dst = svmin_n_f32_x(mask, svmax_n_f32_x(mask, dst, 0.0f), upper);
            return svcvt_u32_f32_x(mask, svadd_n_f32_x(mask, dst, 0.5f));
        }

        SIMD_INLINE void SynetAdd8i(const uint8_t* a, const svfloat32_t& aScale, const svfloat32_t& aShift,
            const uint8_t* b, const svfloat32_t& bScale, const svfloat32_t& bShift, uint8_t* c,
            const svfloat32_t& cScale, const svfloat32_t& cShift, float upper, const svbool_t& mask)
        {
            svfloat32_t _a = svmla_f32_x(mask, aShift, SynetAdd8iLoad(a, mask), aScale);
            svfloat32_t _b = svmla_f32_x(mask, bShift, SynetAdd8iLoad(b, mask), bScale);
            svst1b_u32(mask, c, SynetAdd8iConvert(svadd_f32_x(mask, _a, _b), cScale, cShift, upper, mask));
        }

        void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, float upper)
        {
            const size_t F = svcntw();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _aScale = svdup_n_f32(aScale[c]);
                    svfloat32_t _aShift = svdup_n_f32(aShift[c]);
                    svfloat32_t _bScale = svdup_n_f32(bScale[c]);
                    svfloat32_t _bShift = svdup_n_f32(bShift[c]);
                    svfloat32_t _cScale = svdup_n_f32(cScale[c]);
                    svfloat32_t _cShift = svdup_n_f32(cShift[c]);
                    for (size_t s = 0; s < spatial; s += F)
                    {
                        svbool_t mask = svwhilelt_b32(s, spatial);
                        SynetAdd8i(aData + s, _aScale, _aShift, bData + s, _bScale, _bShift, cData + s, _cScale, _cShift, upper, mask);
                    }
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, float upper)
        {
            const size_t F = svcntw();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += F)
                    {
                        svbool_t mask = svwhilelt_b32(c, channels);
                        SynetAdd8i(aData + c, svld1_f32(mask, aScale + c), svld1_f32(mask, aShift + c),
                            bData + c, svld1_f32(mask, bScale + c), svld1_f32(mask, bShift + c),
                            cData + c, svld1_f32(mask, cScale + c), svld1_f32(mask, cShift + c), upper, mask);
                    }
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            float upper = (float)(Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX);
            if (format == SimdTensorFormatNchw)
                SynetAdd8iNchw(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else if (format == SimdTensorFormatNhwc)
                SynetAdd8iNhwc(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                Base::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }
    }
#endif
}
