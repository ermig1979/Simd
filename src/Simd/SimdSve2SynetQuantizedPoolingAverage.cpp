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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svint32_t QuantizeSumLinear(const svuint32_t& sum, const svint32_t& bias, const svfloat32_t& norm, const svint32_t& zero, const svbool_t& mask)
        {
            svint32_t value = svadd_s32_x(mask, svreinterpret_s32_u32(sum), bias);
            svfloat32_t scaled = svmul_f32_x(mask, svcvt_f32_s32_x(mask, value), norm);
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, scaled, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svadd_s32_x(mask, svcvt_s32_f32_x(mask, svadd_f32_x(mask, scaled, round)), zero);
        }

        SIMD_INLINE void Store8u(const svint32_t& value, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t lo = svmax_n_s32_x(mask, value, 0);
            svuint32_t u32 = svreinterpret_u32_s32(svmin_n_s32_x(mask, lo, 255));
            svst1b_u32(mask, dst, u32);
        }

        SIMD_INLINE void QuantizeSumLinear(const svuint32_t& sum, const svint32_t& bias, const svfloat32_t& norm, const svint32_t& zero, uint8_t* dst, const svbool_t& mask)
        {
            Store8u(QuantizeSumLinear(sum, bias, norm, zero, mask), dst, mask);
        }

        SIMD_INLINE int32_t Sum8u(const uint8_t* src, size_t size)
        {
            size_t i = 0, A = svcntb(), QA = 4 * A, sizeQA = AlignLo(size, QA), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            const svuint8_t one = svdup_n_u8(1), zero = svdup_n_u8(0);
            svuint32_t sum0 = svdup_n_u32(0), sum1 = svdup_n_u32(0), sum2 = svdup_n_u32(0), sum3 = svdup_n_u32(0);
            for (; i < sizeQA; i += QA)
            {
                sum0 = svdot_u32(sum0, svld1_u8(body, src + i + 0 * A), one);
                sum1 = svdot_u32(sum1, svld1_u8(body, src + i + 1 * A), one);
                sum2 = svdot_u32(sum2, svld1_u8(body, src + i + 2 * A), one);
                sum3 = svdot_u32(sum3, svld1_u8(body, src + i + 3 * A), one);
            }
            sum0 = svadd_u32_x(svptrue_b32(), svadd_u32_x(svptrue_b32(), sum0, sum1), svadd_u32_x(svptrue_b32(), sum2, sum3));
            for (; i < sizeA; i += A)
                sum0 = svdot_u32(sum0, svld1_u8(body, src + i), one);
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                sum0 = svdot_u32(sum0, svsel_u8(tail, svld1_u8(tail, src + i), zero), one);
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sum0);
        }

        SIMD_INLINE void QuantizedPoolingAverageNhwc(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW,
            const svint32_t& bias, const svfloat32_t& norm, const svint32_t& zero, uint8_t* dst)
        {
            const size_t F = svcntw();
            size_t c = 0;
            for (; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32(c, srcC);
                svuint32_t sum = svdup_n_u32(0);
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum = svadd_u32_x(mask, sum, svld1ub_u32(mask, src + h * srcS + w * srcC + c));
                QuantizeSumLinear(sum, bias, norm, zero, dst + c, mask);
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNhwc(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            const size_t F = svcntw();
            int32_t bias = -srcZero * int32_t(spatial);
            svint32_t _bias = svdup_n_s32(bias), _zero = svdup_n_s32(dstZero);
            svfloat32_t _norm = svdup_n_f32(srcScale[0] / (dstScale[0] * float(spatial)));
            Array32u sum(channels);
            for (size_t b = 0; b < batch; ++b)
            {
                GetColSums(src, channels, channels, spatial, sum.data);
                for (size_t c = 0; c < channels; c += F)
                {
                    svbool_t mask = svwhilelt_b32(c, channels);
                    QuantizeSumLinear(svld1_u32(mask, sum.data + c), _bias, _norm, _zero, dst + c, mask);
                }
                src += spatial * channels;
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNchw(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            const svint32_t _bias = svdup_n_s32(bias), _zero = svdup_n_s32(dstZero);
            const svfloat32_t _norm = svdup_n_f32(srcScale[0] / (dstScale[0] * float(spatial)));
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    QuantizeSumLinear(svdup_n_u32(Sum8u(src, spatial)), _bias, _norm, _zero, dst + c, svwhilelt_b32(0, 1));
                    src += spatial;
                }
                dst += channels;
            }
        }

        SIMD_INLINE svuint32_t Sum2x2(const uint8_t* src0, const uint8_t* src1, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t s0 = svld1_u8(mask8, src0);
            svuint8_t s1 = svld1_u8(mask8, src1);
            svuint16_t sum = svadd_u16_x(mask16, svaddlb_u16(s0, svext_u8(s0, s0, 1)), svaddlb_u16(s1, svext_u8(s1, s1, 1)));
            return svunpklo_u32(sum);
        }

        SIMD_INLINE void QuantizedPoolingAverageNchw2x2(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW,
            uint8_t* dst, size_t dstH, size_t dstW, const svint32_t& bias, const svfloat32_t& norm, const svint32_t& zero)
        {
            const size_t F = svcntw(), DF = 2 * F;
            const svbool_t body8 = svwhilelt_b8((size_t)0, DF);
            const svbool_t body16 = svwhilelt_b16((size_t)0, F);
            for (size_t b = 0; b < srcC; ++b)
            {
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    size_t dx = 0, sx = 0;
                    const uint8_t* src0 = src + dy * 2 * srcW;
                    const uint8_t* src1 = src0 + srcW;
                    for (; dx + F <= dstW; dx += F, sx += DF)
                        QuantizeSumLinear(Sum2x2(src0 + sx, src1 + sx, body8, body16), bias, norm, zero, dst + dx, svptrue_b32());
                    if (dx < dstW)
                    {
                        size_t tail = dstW - dx;
                        QuantizeSumLinear(Sum2x2(src0 + sx, src1 + sx, svwhilelt_b8((size_t)0, 2 * tail), svwhilelt_b16((size_t)0, tail)),
                            bias, norm, zero, dst + dx, svwhilelt_b32(dx, dstW));
                    }
                    dst += dstW;
                }
                src += srcH * srcW;
            }
        }

        void SynetQuantizedPoolingAverage(const uint8_t* src, const float* srcScale, int srcZero, size_t batch, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, SimdBool excludePad,
            uint8_t* dst, const float* dstScale, int dstZero, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc && srcC >= svcntw())
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNhwc(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                size_t srcS = srcW * srcC;
                svint32_t zero = svdup_n_s32(dstZero);
                int32_t bias = -srcZero * int32_t(kernelY * kernelX);
                float norm = srcScale[0] / (dstScale[0] * float(kernelY * kernelX));
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                        hStart = Simd::Max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                            wStart = Simd::Max<ptrdiff_t>(0, wStart);
                            const uint8_t* ps = src + hStart * srcS + wStart * srcC;
                            if (excludePad)
                            {
                                int area = int(hEnd - hStart) * int(wEnd - wStart);
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, hEnd - hStart, wEnd - wStart,
                                    svdup_n_s32(-srcZero * area), svdup_n_f32(srcScale[0] / (dstScale[0] * float(area))), zero, dst);
                            }
                            else
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, hEnd - hStart, wEnd - wStart,
                                    svdup_n_s32(bias), svdup_n_f32(norm), zero, dst);
                            dst += srcC;
                        }
                    }
                    src += srcC * srcH * srcW;
                }
                return;
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNchw(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                if (kernelY == 2 && kernelX == 2 && strideY == 2 && strideX == 2 && padY == 0 && padX == 0 && dstH * 2 <= srcH && dstW * 2 <= srcW)
                {
                    QuantizedPoolingAverageNchw2x2(src, srcC * batch, srcH, srcW, dst, dstH, dstW, svdup_n_s32(-srcZero * 4),
                        svdup_n_f32(srcScale[0] / (dstScale[0] * 4.0f)), svdup_n_s32(dstZero));
                    return;
                }
            }
            Base::SynetQuantizedPoolingAverage(src, srcScale, srcZero, batch, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX,
                padY, padX, excludePad, dst, dstScale, dstZero, dstH, dstW, format);
        }
    }
#endif
}
