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

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svint32_t QuantizeLinear(const svfloat32_t& value, const svfloat32_t& norm, const svint32_t& zero, const svbool_t& mask)
        {
            svfloat32_t scaled = svmul_f32_x(mask, value, norm);
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, scaled, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svadd_s32_x(mask, svcvt_s32_f32_x(mask, svadd_f32_x(mask, scaled, round)), zero);
        }

        SIMD_INLINE svuint32_t DequantizeQuantizeLinear(const svuint32_t& src, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale, const svint32_t& zero, const svbool_t& mask)
        {
            svint32_t value = svadd_s32_x(mask, svreinterpret_s32_u32(src), bias);
            svint32_t i32 = QuantizeLinear(svmul_f32_x(mask, svcvt_f32_s32_x(mask, value), norm), scale, zero, mask);
            i32 = svmin_n_s32_x(mask, svmax_n_s32_x(mask, i32, 0), 255);
            return svreinterpret_u32_s32(i32);
        }

        SIMD_INLINE void DequantizeQuantizeLinear(const uint8_t* src, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale, const svint32_t& zero, uint8_t* dst, const svbool_t& mask)
        {
            svst1b_u32(mask, dst, DequantizeQuantizeLinear(svld1ub_u32(mask, src), bias, norm, scale, zero, mask));
        }

        SIMD_INLINE void DequantizeQuantizeLinear(const uint8_t* src, size_t size, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale, const svint32_t& zero, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                DequantizeQuantizeLinear(src + i + 0 * F, bias, norm, scale, zero, dst + i + 0 * F, body);
                DequantizeQuantizeLinear(src + i + 1 * F, bias, norm, scale, zero, dst + i + 1 * F, body);
                DequantizeQuantizeLinear(src + i + 2 * F, bias, norm, scale, zero, dst + i + 2 * F, body);
                DequantizeQuantizeLinear(src + i + 3 * F, bias, norm, scale, zero, dst + i + 3 * F, body);
            }
            for (; i < size; i += F)
                DequantizeQuantizeLinear(src + i, bias, norm, scale, zero, dst + i, svwhilelt_b32(i, size));
        }

        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            svint32_t _bias0 = svdup_n_s32(bias0), _bias1 = svdup_n_s32(bias1), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm0), _norm1 = svdup_n_f32(norm1), _scale = svdup_n_f32(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2)
            {
                DequantizeQuantizeLinear(src0, spatial, _bias0, _norm0, _scale, _zero, dst0);
                src0 += spatial;
                dst0 += spatial;
                DequantizeQuantizeLinear(src0, spatial, _bias0, _norm0, _scale, _zero, dst1);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2)
            {
                DequantizeQuantizeLinear(src1, spatial, _bias1, _norm1, _scale, _zero, dst0);
                src1 += spatial;
                dst0 += spatial;
                DequantizeQuantizeLinear(src1, spatial, _bias1, _norm1, _scale, _zero, dst1);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0(const uint8_t* src, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale,
            const svint32_t& zero, const svuint32_t& even, const svuint32_t& odd, uint8_t* dst0, uint8_t* dst1, const svbool_t& mask)
        {
            svst1b_u32(mask, dst0, DequantizeQuantizeLinear(svld1ub_gather_u32offset_u32(mask, src, even), bias, norm, scale, zero, mask));
            svst1b_u32(mask, dst1, DequantizeQuantizeLinear(svld1ub_gather_u32offset_u32(mask, src, odd), bias, norm, scale, zero, mask));
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            const size_t F = svcntw(), dstC = (srcC0 + srcC1) / 2;
            const svuint32_t even = svindex_u32(0, 2), odd = svindex_u32(1, 2);
            svint32_t _bias0 = svdup_n_s32(bias0), _bias1 = svdup_n_s32(bias1), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm0), _norm1 = svdup_n_f32(norm1), _scale = svdup_n_f32(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t srcC0H = srcC0 / 2, srcC1H = srcC1 / 2;
                for (size_t c0 = 0; c0 < srcC0H; c0 += F)
                    DequantizeQuantizeLinearNhwc0(src0 + 2 * c0, _bias0, _norm0, _scale, _zero, even, odd, dst0 + c0, dst1 + c0, svwhilelt_b32(c0, srcC0H));
                for (size_t c1 = 0; c1 < srcC1H; c1 += F)
                    DequantizeQuantizeLinearNhwc0(src1 + 2 * c1, _bias1, _norm1, _scale, _zero, even, odd, dst0 + srcC0H + c1, dst1 + srcC0H + c1, svwhilelt_b32(c1, srcC1H));
                src0 += srcC0;
                src1 += srcC1;
                dst0 += dstC;
                dst1 += dstC;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForwardNchw1(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            svint32_t _bias0 = svdup_n_s32(bias0), _bias1 = svdup_n_s32(bias1), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm0), _norm1 = svdup_n_f32(norm1), _scale = svdup_n_f32(scale);
            for (size_t cd = 0; cd < srcC0; cd += 2)
            {
                DequantizeQuantizeLinear(src0, spatial, _bias0, _norm0, _scale, _zero, dst0);
                src0 += spatial;
                dst0 += spatial;
                DequantizeQuantizeLinear(src1, spatial, _bias1, _norm1, _scale, _zero, dst0);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cd += 2)
            {
                DequantizeQuantizeLinear(src0, spatial, _bias0, _norm0, _scale, _zero, dst1);
                src0 += spatial;
                dst1 += spatial;
                DequantizeQuantizeLinear(src1, spatial, _bias1, _norm1, _scale, _zero, dst1);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1(const uint8_t* src0, const uint8_t* src1, const svint32_t& bias0, const svint32_t& bias1,
            const svfloat32_t& norm0, const svfloat32_t& norm1, const svfloat32_t& scale, const svint32_t& zero, const svuint32_t& even, const svuint32_t& odd, uint8_t* dst, const svbool_t& mask)
        {
            svst1b_scatter_u32offset_u32(mask, dst, even, DequantizeQuantizeLinear(svld1ub_u32(mask, src0), bias0, norm0, scale, zero, mask));
            svst1b_scatter_u32offset_u32(mask, dst, odd, DequantizeQuantizeLinear(svld1ub_u32(mask, src1), bias1, norm1, scale, zero, mask));
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            const size_t F = svcntw(), dstC = (srcC0 + srcC1) / 2;
            const svuint32_t even = svindex_u32(0, 2), odd = svindex_u32(1, 2);
            svint32_t _bias0 = svdup_n_s32(bias0), _bias1 = svdup_n_s32(bias1), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm0), _norm1 = svdup_n_f32(norm1), _scale = svdup_n_f32(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t dstC0H = srcC0 / 2, dstC1H = srcC1 / 2;
                for (size_t cd = 0; cd < dstC0H; cd += F)
                    DequantizeQuantizeLinearNhwc1(src0 + cd, src1 + cd, _bias0, _bias1, _norm0, _norm1, _scale, _zero, even, odd, dst0 + 2 * cd, svwhilelt_b32(cd, dstC0H));
                for (size_t cd = 0; cd < dstC1H; cd += F)
                    DequantizeQuantizeLinearNhwc1(src0 + dstC0H + cd, src1 + dstC0H + cd, _bias0, _bias1, _norm0, _norm1, _scale, _zero, even, odd, dst1 + 2 * cd, svwhilelt_b32(cd, dstC1H));
                src0 += dstC;
                src1 += dstC;
                dst0 += srcC0;
                dst1 += srcC1;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForward(const uint8_t* src0, int bias0, const float* norm0, size_t srcC0, const uint8_t* src1, int bias1, const float* norm1, size_t srcC1,
            size_t spatial, uint8_t* dst0, uint8_t* dst1, const float* scale, int zero, SimdTensorFormatType format, int shuffleType)
        {
            switch (shuffleType)
            {
            case 0:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            case 1:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            }
        }
    }
#endif
}
