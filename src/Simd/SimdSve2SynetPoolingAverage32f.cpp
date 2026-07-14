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
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void PoolingAverageNhwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& norm, float* dst, const svbool_t& mask)
        {
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = svadd_f32_x(mask, sum0, svld1_f32(mask, ps));
                }
                src += srcS;
            }
            svst1_f32(mask, dst, svmul_f32_x(mask, sum0, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& norm, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = svadd_f32_x(mask, sum0, svld1_f32(mask, ps + 0 * F));
                    sum1 = svadd_f32_x(mask, sum1, svld1_f32(mask, ps + 1 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, svmul_f32_x(mask, sum0, norm));
            svst1_f32(mask, dst + 1 * F, svmul_f32_x(mask, sum1, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& norm, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = svadd_f32_x(mask, sum0, svld1_f32(mask, ps + 0 * F));
                    sum1 = svadd_f32_x(mask, sum1, svld1_f32(mask, ps + 1 * F));
                    sum2 = svadd_f32_x(mask, sum2, svld1_f32(mask, ps + 2 * F));
                    sum3 = svadd_f32_x(mask, sum3, svld1_f32(mask, ps + 3 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, svmul_f32_x(mask, sum0, norm));
            svst1_f32(mask, dst + 1 * F, svmul_f32_x(mask, sum1, norm));
            svst1_f32(mask, dst + 2 * F, svmul_f32_x(mask, sum2, norm));
            svst1_f32(mask, dst + 3 * F, svmul_f32_x(mask, sum3, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& norm, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);
            svfloat32_t sum4 = svdup_n_f32(0.0f);
            svfloat32_t sum5 = svdup_n_f32(0.0f);
            svfloat32_t sum6 = svdup_n_f32(0.0f);
            svfloat32_t sum7 = svdup_n_f32(0.0f);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = svadd_f32_x(mask, sum0, svld1_f32(mask, ps + 0 * F));
                    sum1 = svadd_f32_x(mask, sum1, svld1_f32(mask, ps + 1 * F));
                    sum2 = svadd_f32_x(mask, sum2, svld1_f32(mask, ps + 2 * F));
                    sum3 = svadd_f32_x(mask, sum3, svld1_f32(mask, ps + 3 * F));
                    sum4 = svadd_f32_x(mask, sum4, svld1_f32(mask, ps + 4 * F));
                    sum5 = svadd_f32_x(mask, sum5, svld1_f32(mask, ps + 5 * F));
                    sum6 = svadd_f32_x(mask, sum6, svld1_f32(mask, ps + 6 * F));
                    sum7 = svadd_f32_x(mask, sum7, svld1_f32(mask, ps + 7 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, svmul_f32_x(mask, sum0, norm));
            svst1_f32(mask, dst + 1 * F, svmul_f32_x(mask, sum1, norm));
            svst1_f32(mask, dst + 2 * F, svmul_f32_x(mask, sum2, norm));
            svst1_f32(mask, dst + 3 * F, svmul_f32_x(mask, sum3, norm));
            svst1_f32(mask, dst + 4 * F, svmul_f32_x(mask, sum4, norm));
            svst1_f32(mask, dst + 5 * F, svmul_f32_x(mask, sum5, norm));
            svst1_f32(mask, dst + 6 * F, svmul_f32_x(mask, sum6, norm));
            svst1_f32(mask, dst + 7 * F, svmul_f32_x(mask, sum7, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc(const float* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const svfloat32_t& norm, float* dst, const svbool_t& body)
        {
            const size_t F = svcntw();
            size_t c = 0;
            for (; c < srcCF8; c += 8 * F)
                PoolingAverageNhwc8(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, body);
            for (; c < srcCF4; c += 4 * F)
                PoolingAverageNhwc4(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, body);
            for (; c < srcCF2; c += 2 * F)
                PoolingAverageNhwc2(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, body);
            for (; c < srcCF1; c += 1 * F)
                PoolingAverageNhwc1(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, body);
            if (c < srcC)
                PoolingAverageNhwc1(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, svwhilelt_b32(c, srcC));
        }

        void SynetPoolingAverage(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format)
        {
            const size_t F = svcntw();
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    svbool_t body = svptrue_b32();
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        svfloat32_t norm = svdup_n_f32(1.0f / (kernelY * kernelX));
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const float* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kernelY, kernelX, norm, dst, body);
                        }
                    }
                    else if (excludePad)
                    {
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            size_t hStart = ph * strideY - padY;
                            size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                            hStart = Simd::Max<ptrdiff_t>(0, hStart);
                            size_t kH = hEnd - hStart;
                            for (size_t pw = 0; pw < dstW; ++pw)
                            {
                                size_t wStart = pw * strideX - padX;
                                size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                                wStart = Simd::Max<ptrdiff_t>(0, wStart);
                                size_t kW = wEnd - wStart;
                                const float* ps = src + hStart * srcS + wStart * srcC;
                                svfloat32_t norm = svdup_n_f32(1.0f / (kH * kW));
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst, body);
                                dst += srcC;
                            }
                        }
                    }
                    else
                    {
                        svfloat32_t norm = svdup_n_f32(1.0f / (kernelY * kernelX));
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            size_t hStart = ph * strideY - padY;
                            size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                            hStart = Simd::Max<ptrdiff_t>(0, hStart);
                            size_t kH = hEnd - hStart;
                            for (size_t pw = 0; pw < dstW; ++pw)
                            {
                                size_t wStart = pw * strideX - padX;
                                size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                                wStart = Simd::Max<ptrdiff_t>(0, wStart);
                                size_t kW = wEnd - wStart;
                                const float* ps = src + hStart * srcS + wStart * srcC;
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst, body);
                                dst += srcC;
                            }
                        }
                    }
                    return;
                }
            }
            Base::SynetPoolingAverage(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, excludePad, format);
        }
    }
#endif
}
