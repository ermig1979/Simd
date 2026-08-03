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
        SIMD_INLINE void PoolingMax32fNhwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, float* dst, const svbool_t& mask)
        {
            svfloat32_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, svld1_f32(mask, ps));
                }
                src += srcS;
            }
            svst1_f32(mask, dst, max0);
        }

        SIMD_INLINE void PoolingMax32fNhwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, svld1_f32(mask, ps + 0 * F));
                    max1 = svmax_f32_x(mask, max1, svld1_f32(mask, ps + 1 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, max0);
            svst1_f32(mask, dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMax32fNhwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            svfloat32_t max2 = min;
            svfloat32_t max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, svld1_f32(mask, ps + 0 * F));
                    max1 = svmax_f32_x(mask, max1, svld1_f32(mask, ps + 1 * F));
                    max2 = svmax_f32_x(mask, max2, svld1_f32(mask, ps + 2 * F));
                    max3 = svmax_f32_x(mask, max3, svld1_f32(mask, ps + 3 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, max0);
            svst1_f32(mask, dst + 1 * F, max1);
            svst1_f32(mask, dst + 2 * F, max2);
            svst1_f32(mask, dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMax32fNhwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, float* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            svfloat32_t max2 = min;
            svfloat32_t max3 = min;
            svfloat32_t max4 = min;
            svfloat32_t max5 = min;
            svfloat32_t max6 = min;
            svfloat32_t max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, svld1_f32(mask, ps + 0 * F));
                    max1 = svmax_f32_x(mask, max1, svld1_f32(mask, ps + 1 * F));
                    max2 = svmax_f32_x(mask, max2, svld1_f32(mask, ps + 2 * F));
                    max3 = svmax_f32_x(mask, max3, svld1_f32(mask, ps + 3 * F));
                    max4 = svmax_f32_x(mask, max4, svld1_f32(mask, ps + 4 * F));
                    max5 = svmax_f32_x(mask, max5, svld1_f32(mask, ps + 5 * F));
                    max6 = svmax_f32_x(mask, max6, svld1_f32(mask, ps + 6 * F));
                    max7 = svmax_f32_x(mask, max7, svld1_f32(mask, ps + 7 * F));
                }
                src += srcS;
            }
            svst1_f32(mask, dst + 0 * F, max0);
            svst1_f32(mask, dst + 1 * F, max1);
            svst1_f32(mask, dst + 2 * F, max2);
            svst1_f32(mask, dst + 3 * F, max3);
            svst1_f32(mask, dst + 4 * F, max4);
            svst1_f32(mask, dst + 5 * F, max5);
            svst1_f32(mask, dst + 6 * F, max6);
            svst1_f32(mask, dst + 7 * F, max7);
        }

        SIMD_INLINE void PoolingMax32fNhwc(const float* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const svfloat32_t& min, float* dst, const svbool_t& body)
        {
            const size_t F = svcntw();
            size_t c = 0;
            for (; c < srcCF8; c += 8 * F)
                PoolingMax32fNhwc8(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF4; c += 4 * F)
                PoolingMax32fNhwc4(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF2; c += 2 * F)
                PoolingMax32fNhwc2(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF1; c += 1 * F)
                PoolingMax32fNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            if (c < srcC)
                PoolingMax32fNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, svwhilelt_b32(c, srcC));
        }

        void SynetPoolingMax32f2D(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
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
                    svfloat32_t min = svdup_n_f32(-FLT_MAX);
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const float* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingMax32fNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kernelY, kernelX, min, dst, body);
                        }
                    }
                    else
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
                                PoolingMax32fNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, min, dst, body);
                                dst += srcC;
                            }
                        }
                    }
                    return;
                }
            }
            Base::SynetPoolingMax32f(src, srcC, srcH, srcW, 1, kernelY, kernelX, 1, strideY, strideX, 0, padY, padX, dst, srcC, dstH, dstW, format);
        }

        void SynetPoolingMax32f(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (kernelC == 1 && strideC == 1 && padC == 0 && srcC == dstC)
                SynetPoolingMax32f2D(src, srcC, srcH, srcW, kernelY, kernelX,
                    strideY, strideX, padY, padX, dst, dstH, dstW, format);
            else
                Base::SynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX,
                    strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }
    }
#endif
}
