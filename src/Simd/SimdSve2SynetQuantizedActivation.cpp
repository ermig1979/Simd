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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svfloat32_t BFloat16ToFloat32(svuint32_t value, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, value, Base::Bf16::SHIFT));
        }

        SIMD_INLINE void SynetRelu16b(const uint16_t* src, const svbool_t& mask, const svfloat32_t& slope, uint16_t* dst)
        {
            svfloat32_t _src = BFloat16ToFloat32(svld1uh_u32(mask, src), mask);
            svfloat32_t pos = svmax_n_f32_x(mask, _src, 0.0f);
            svfloat32_t neg = svmin_n_f32_x(mask, _src, 0.0f);
            svst1h_u32(mask, dst, Float32ToBFloat16(svmla_f32_x(mask, pos, slope, neg), mask));
        }

        void SynetRelu16b(const uint16_t* src, size_t size, const float* slope, uint16_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _slope = svdup_n_f32(slope[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                SynetRelu16b(src + i + 0 * F, body, _slope, dst + i + 0 * F);
                SynetRelu16b(src + i + 1 * F, body, _slope, dst + i + 1 * F);
                SynetRelu16b(src + i + 2 * F, body, _slope, dst + i + 2 * F);
                SynetRelu16b(src + i + 3 * F, body, _slope, dst + i + 3 * F);
            }
            for (; i < size; i += F)
                SynetRelu16b(src + i, svwhilelt_b32(i, size), _slope, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t Load8u(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE void Store8u(const svint32_t& value, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t lo = svmax_n_s32_x(mask, value, 0);
            svuint32_t u32 = svreinterpret_u32_s32(svmin_n_s32_x(mask, lo, 255));
            svst1b_u32(mask, dst, u32);
        }

        SIMD_INLINE svfloat32_t SynetHardSigmoid32f(const svfloat32_t& value, const svfloat32_t& scale, const svfloat32_t& shift, const svbool_t& mask)
        {
            return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, shift, value, scale), 1.0f), 0.0f);
        }

        SIMD_INLINE svint32_t QuantizedHardSigmoid(const svint32_t& src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& scale, const svfloat32_t& shift, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            svfloat32_t _src = DequantizeLinear(src, sBias, sNorm, mask);
            svfloat32_t _dst = SynetHardSigmoid32f(_src, scale, shift, mask);
            return svadd_s32_x(mask, Round(svmul_f32_x(mask, _dst, dNorm), mask), dZero);
        }

        SIMD_INLINE void QuantizedHardSigmoid(const uint8_t* src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& scale, const svfloat32_t& shift, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            Store8u(QuantizedHardSigmoid(Load8u(src, mask), sBias, sNorm, scale, shift, dNorm, dZero, mask), dst, mask);
        }

        void SynetQuantizedHardSigmoid(const uint8_t* src, const float* srcScale, int srcZero, size_t size, const float* scale, const float* shift, uint8_t* dst, const float* dstScale, int dstZero)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            const svint32_t sBias = svdup_n_s32(-srcZero), dZero = svdup_n_s32(dstZero);
            const svfloat32_t sNorm = svdup_n_f32(srcScale[0]), dNorm = svdup_n_f32(1.0f / dstScale[0]);
            const svfloat32_t _scale = svdup_n_f32(scale[0]), _shift = svdup_n_f32(shift[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                QuantizedHardSigmoid(src + i + 0 * F, sBias, sNorm, _scale, _shift, dst + i + 0 * F, dNorm, dZero, full);
                QuantizedHardSigmoid(src + i + 1 * F, sBias, sNorm, _scale, _shift, dst + i + 1 * F, dNorm, dZero, full);
                QuantizedHardSigmoid(src + i + 2 * F, sBias, sNorm, _scale, _shift, dst + i + 2 * F, dNorm, dZero, full);
                QuantizedHardSigmoid(src + i + 3 * F, sBias, sNorm, _scale, _shift, dst + i + 3 * F, dNorm, dZero, full);
            }
            for (; i < size; i += F)
                QuantizedHardSigmoid(src + i, sBias, sNorm, _scale, _shift, dst + i, dNorm, dZero, svwhilelt_b32(i, size));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetHswish32f(const svfloat32_t& value, const svfloat32_t& shift, const svfloat32_t& scale, const svbool_t& mask)
        {
            svfloat32_t upper = svmin_f32_x(mask, value, shift);
            svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, shift), 0.0f);
            return svmul_f32_x(mask, svmul_f32_x(mask, positive, scale), value);
        }

        SIMD_INLINE svint32_t QuantizedHswish(const svint32_t& src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& shift, const svfloat32_t& scale, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            svfloat32_t _src = DequantizeLinear(src, sBias, sNorm, mask);
            svfloat32_t _dst = SynetHswish32f(_src, shift, scale, mask);
            return svadd_s32_x(mask, Round(svmul_f32_x(mask, _dst, dNorm), mask), dZero);
        }

        SIMD_INLINE void QuantizedHswish(const uint8_t* src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& shift, const svfloat32_t& scale, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            Store8u(QuantizedHswish(Load8u(src, mask), sBias, sNorm, shift, scale, dNorm, dZero, mask), dst, mask);
        }

        void SynetQuantizedHswish(const uint8_t* src, const float* srcScale, int srcZero, size_t size, const float* shift, const float* scale, uint8_t* dst, const float* dstScale, int dstZero)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            const svint32_t sBias = svdup_n_s32(-srcZero), dZero = svdup_n_s32(dstZero);
            const svfloat32_t sNorm = svdup_n_f32(srcScale[0]), dNorm = svdup_n_f32(1.0f / dstScale[0]);
            const svfloat32_t _shift = svdup_n_f32(shift[0]), _scale = svdup_n_f32(scale[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                QuantizedHswish(src + i + 0 * F, sBias, sNorm, _shift, _scale, dst + i + 0 * F, dNorm, dZero, full);
                QuantizedHswish(src + i + 1 * F, sBias, sNorm, _shift, _scale, dst + i + 1 * F, dNorm, dZero, full);
                QuantizedHswish(src + i + 2 * F, sBias, sNorm, _shift, _scale, dst + i + 2 * F, dNorm, dZero, full);
                QuantizedHswish(src + i + 3 * F, sBias, sNorm, _shift, _scale, dst + i + 3 * F, dNorm, dZero, full);
            }
            for (; i < size; i += F)
                QuantizedHswish(src + i, sBias, sNorm, _shift, _scale, dst + i, dNorm, dZero, svwhilelt_b32(i, size));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t QuantizedPrelu(const svint32_t& src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& slope, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            svfloat32_t _src = DequantizeLinear(src, sBias, sNorm, mask);
            svfloat32_t pos = svmax_n_f32_x(mask, _src, 0.0f);
            svfloat32_t neg = svmin_n_f32_x(mask, _src, 0.0f);
            svfloat32_t _dst = svmla_f32_x(mask, pos, slope, neg);
            return svadd_s32_x(mask, Round(svmul_f32_x(mask, _dst, dNorm), mask), dZero);
        }

        SIMD_INLINE void QuantizedPrelu(const uint8_t* src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& slope, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            Store8u(QuantizedPrelu(Load8u(src, mask), sBias, sNorm, slope, dNorm, dZero, mask), dst, mask);
        }

        void SynetQuantizedPreluLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            const svint32_t sBias = svdup_n_s32(-srcZero), dZero = svdup_n_s32(dstZero);
            const svfloat32_t sNorm = svdup_n_f32(srcScale[0]), dNorm = svdup_n_f32(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        QuantizedPrelu(src + c + 0 * F, sBias, sNorm, svld1_f32(full, slope + c + 0 * F), dst + c + 0 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + c + 1 * F, sBias, sNorm, svld1_f32(full, slope + c + 1 * F), dst + c + 1 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + c + 2 * F, sBias, sNorm, svld1_f32(full, slope + c + 2 * F), dst + c + 2 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + c + 3 * F, sBias, sNorm, svld1_f32(full, slope + c + 3 * F), dst + c + 3 * F, dNorm, dZero, full);
                    }
                    for (; c < channels; c += F)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        QuantizedPrelu(src + c, sBias, sNorm, svld1_f32(tail, slope + c), dst + c, dNorm, dZero, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _slope = svdup_n_f32(slope[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        QuantizedPrelu(src + s + 0 * F, sBias, sNorm, _slope, dst + s + 0 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + s + 1 * F, sBias, sNorm, _slope, dst + s + 1 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + s + 2 * F, sBias, sNorm, _slope, dst + s + 2 * F, dNorm, dZero, full);
                        QuantizedPrelu(src + s + 3 * F, sBias, sNorm, _slope, dst + s + 3 * F, dNorm, dZero, full);
                    }
                    for (; s < spatial; s += F)
                        QuantizedPrelu(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
