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
#include "Simd/SimdArray.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t DequantizeLinear(const svint32_t& value, const svint32_t& bias, const svfloat32_t& norm, const svbool_t& mask)
        {
            return svmul_f32_x(mask, svcvt_f32_s32_x(mask, svadd_s32_x(mask, value, bias)), norm);
        }

        SIMD_INLINE svint32_t QuantizeLinear(const svfloat32_t& value, const svfloat32_t& norm, const svint32_t& zero, const svbool_t& mask)
        {
            svfloat32_t scaled = svmul_f32_x(mask, value, norm);
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, scaled, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svadd_s32_x(mask, svcvt_s32_f32_x(mask, svadd_f32_x(mask, scaled, round)), zero);
        }

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

        SIMD_INLINE svint32_t QuantizedScale(const svint32_t& src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& scale, const svfloat32_t& bias, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            svfloat32_t _src = DequantizeLinear(src, sBias, sNorm, mask);
            svfloat32_t _dst = svmla_f32_x(mask, bias, _src, scale);
            return QuantizeLinear(_dst, dNorm, dZero, mask);
        }

        SIMD_INLINE void QuantizedScale(const uint8_t* src, const svint32_t& sBias, const svfloat32_t& sNorm,
            const svfloat32_t& scale, const svfloat32_t& bias, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            Store8u(QuantizedScale(Load8u(src, mask), sBias, sNorm, scale, bias, dNorm, dZero, mask), dst, mask);
        }

        void SynetQuantizedScaleLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            Array32f defaultBias;
            if (bias == NULL)
            {
                defaultBias.Resize(channels, true);
                bias = defaultBias.data;
            }
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
                        QuantizedScale(src + c + 0 * F, sBias, sNorm, svld1_f32(full, scale + c + 0 * F), svld1_f32(full, bias + c + 0 * F), dst + c + 0 * F, dNorm, dZero, full);
                        QuantizedScale(src + c + 1 * F, sBias, sNorm, svld1_f32(full, scale + c + 1 * F), svld1_f32(full, bias + c + 1 * F), dst + c + 1 * F, dNorm, dZero, full);
                        QuantizedScale(src + c + 2 * F, sBias, sNorm, svld1_f32(full, scale + c + 2 * F), svld1_f32(full, bias + c + 2 * F), dst + c + 2 * F, dNorm, dZero, full);
                        QuantizedScale(src + c + 3 * F, sBias, sNorm, svld1_f32(full, scale + c + 3 * F), svld1_f32(full, bias + c + 3 * F), dst + c + 3 * F, dNorm, dZero, full);
                    }
                    for (; c < channels; c += F)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        QuantizedScale(src + c, sBias, sNorm, svld1_f32(tail, scale + c), svld1_f32(tail, bias + c), dst + c, dNorm, dZero, tail);
                    }
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    svfloat32_t _bias = svdup_n_f32(bias[c]);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        QuantizedScale(src + s + 0 * F, sBias, sNorm, _scale, _bias, dst + s + 0 * F, dNorm, dZero, full);
                        QuantizedScale(src + s + 1 * F, sBias, sNorm, _scale, _bias, dst + s + 1 * F, dNorm, dZero, full);
                        QuantizedScale(src + s + 2 * F, sBias, sNorm, _scale, _bias, dst + s + 2 * F, dNorm, dZero, full);
                        QuantizedScale(src + s + 3 * F, sBias, sNorm, _scale, _bias, dst + s + 3 * F, dNorm, dZero, full);
                    }
                    for (; s < spatial; s += F)
                        QuantizedScale(src + s, sBias, sNorm, _scale, _bias, dst + s, dNorm, dZero, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
