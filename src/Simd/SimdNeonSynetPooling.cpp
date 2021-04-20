/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Neon
    {
        SIMD_INLINE void PoolingAverageNhwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& norm, float* dst)
        {
            float32x4_t sum0 = vdupq_n_f32(0);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = vaddq_f32(sum0, Load<false>(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, vmulq_f32(sum0, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& norm, float* dst)
        {
            float32x4_t sum0 = vdupq_n_f32(0);
            float32x4_t sum1 = vdupq_n_f32(0);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = vaddq_f32(sum0, Load<false>(src + w * srcC + 0 * F));
                    sum1 = vaddq_f32(sum1, Load<false>(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, vmulq_f32(sum0, norm));
            Store<false>(dst + 1 * F, vmulq_f32(sum1, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& norm, float* dst)
        {
            float32x4_t sum0 = vdupq_n_f32(0);
            float32x4_t sum1 = vdupq_n_f32(0);
            float32x4_t sum2 = vdupq_n_f32(0);
            float32x4_t sum3 = vdupq_n_f32(0);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = vaddq_f32(sum0, Load<false>(src + w * srcC + 0 * F));
                    sum1 = vaddq_f32(sum1, Load<false>(src + w * srcC + 1 * F));
                    sum2 = vaddq_f32(sum2, Load<false>(src + w * srcC + 2 * F));
                    sum3 = vaddq_f32(sum3, Load<false>(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, vmulq_f32(sum0, norm));
            Store<false>(dst + 1 * F, vmulq_f32(sum1, norm));
            Store<false>(dst + 2 * F, vmulq_f32(sum2, norm));
            Store<false>(dst + 3 * F, vmulq_f32(sum3, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& norm, float* dst)
        {
            float32x4_t sum0 = vdupq_n_f32(0);
            float32x4_t sum1 = vdupq_n_f32(0);
            float32x4_t sum2 = vdupq_n_f32(0);
            float32x4_t sum3 = vdupq_n_f32(0);
            float32x4_t sum4 = vdupq_n_f32(0);
            float32x4_t sum5 = vdupq_n_f32(0);
            float32x4_t sum6 = vdupq_n_f32(0);
            float32x4_t sum7 = vdupq_n_f32(0);
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = vaddq_f32(sum0, Load<false>(src + w * srcC + 0 * F));
                    sum1 = vaddq_f32(sum1, Load<false>(src + w * srcC + 1 * F));
                    sum2 = vaddq_f32(sum2, Load<false>(src + w * srcC + 2 * F));
                    sum3 = vaddq_f32(sum3, Load<false>(src + w * srcC + 3 * F));
                    sum4 = vaddq_f32(sum4, Load<false>(src + w * srcC + 4 * F));
                    sum5 = vaddq_f32(sum5, Load<false>(src + w * srcC + 5 * F));
                    sum6 = vaddq_f32(sum6, Load<false>(src + w * srcC + 6 * F));
                    sum7 = vaddq_f32(sum7, Load<false>(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, vmulq_f32(sum0, norm));
            Store<false>(dst + 1 * F, vmulq_f32(sum1, norm));
            Store<false>(dst + 2 * F, vmulq_f32(sum2, norm));
            Store<false>(dst + 3 * F, vmulq_f32(sum3, norm));
            Store<false>(dst + 4 * F, vmulq_f32(sum4, norm));
            Store<false>(dst + 5 * F, vmulq_f32(sum5, norm));
            Store<false>(dst + 6 * F, vmulq_f32(sum6, norm));
            Store<false>(dst + 7 * F, vmulq_f32(sum7, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc(const float* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const float32x4_t& norm, float* dst)
        {
            size_t c = 0;
            for (; c < srcCF8; c += 8 * F)
                PoolingAverageNhwc8(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c);
            for (; c < srcCF4; c += 4 * F)
                PoolingAverageNhwc4(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c);
            for (; c < srcCF2; c += 2 * F)
                PoolingAverageNhwc2(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c);
            for (; c < srcCF1; c += 1 * F)
                PoolingAverageNhwc1(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c);
            if (c < srcC)
                PoolingAverageNhwc1(src + srcC - F, srcS, srcC, kernelY, kernelX, norm, dst + srcC - F);
        }

        void SynetPoolingForwardAverage(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        float32x4_t norm = vdupq_n_f32(1.0f / (kernelY * kernelX));
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const float* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kernelY, kernelX, norm, dst);
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
                                float32x4_t norm = vdupq_n_f32(1.0f / (kH * kW));
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst);
                                dst += srcC;
                            }
                        }
                    }
                    else
                    {
                        float32x4_t norm = vdupq_n_f32(1.0f / (kernelY * kernelX));
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
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst);
                                dst += srcC;
                            }
}
                    }
                    return;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
            }
            Base::SynetPoolingForwardAverage(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, excludePad, format);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void PoolingMaxHwc1(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t & min, float * dst)
        {
            float32x4_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = vmaxq_f32(max0, Load<false>(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, max0);
        }

        SIMD_INLINE void PoolingMaxHwc2(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t & min, float * dst)
        {
            float32x4_t max0 = min;
            float32x4_t max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = vmaxq_f32(max0, Load<false>(src + w * srcC + 0 * F));
                    max1 = vmaxq_f32(max1, Load<false>(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, max0);
            Store<false>(dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMaxHwc4(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t & min, float * dst)
        {
            float32x4_t max0 = min;
            float32x4_t max1 = min;
            float32x4_t max2 = min;
            float32x4_t max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = vmaxq_f32(max0, Load<false>(src + w * srcC + 0 * F));
                    max1 = vmaxq_f32(max1, Load<false>(src + w * srcC + 1 * F));
                    max2 = vmaxq_f32(max2, Load<false>(src + w * srcC + 2 * F));
                    max3 = vmaxq_f32(max3, Load<false>(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, max0);
            Store<false>(dst + 1 * F, max1);
            Store<false>(dst + 2 * F, max2);
            Store<false>(dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMaxHwc8(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t & min, float * dst)
        {
            float32x4_t max0 = min;
            float32x4_t max1 = min;
            float32x4_t max2 = min;
            float32x4_t max3 = min;
            float32x4_t max4 = min;
            float32x4_t max5 = min;
            float32x4_t max6 = min;
            float32x4_t max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = vmaxq_f32(max0, Load<false>(src + w * srcC + 0 * F));
                    max1 = vmaxq_f32(max1, Load<false>(src + w * srcC + 1 * F));
                    max2 = vmaxq_f32(max2, Load<false>(src + w * srcC + 2 * F));
                    max3 = vmaxq_f32(max3, Load<false>(src + w * srcC + 3 * F));
                    max4 = vmaxq_f32(max4, Load<false>(src + w * srcC + 4 * F));
                    max5 = vmaxq_f32(max5, Load<false>(src + w * srcC + 5 * F));
                    max6 = vmaxq_f32(max6, Load<false>(src + w * srcC + 6 * F));
                    max7 = vmaxq_f32(max7, Load<false>(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * F, max0);
            Store<false>(dst + 1 * F, max1);
            Store<false>(dst + 2 * F, max2);
            Store<false>(dst + 3 * F, max3);
            Store<false>(dst + 4 * F, max4);
            Store<false>(dst + 5 * F, max5);
            Store<false>(dst + 6 * F, max6);
            Store<false>(dst + 7 * F, max7);
        }

        void SynetPoolingForwardMax32f(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    float32x4_t min = vdupq_n_f32(-FLT_MAX);
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
                            const float* ps = src + hStart * srcS + wStart * srcC;
                            size_t c = 0;
                            for (; c < srcCF8; c += 8 * F)
                                PoolingMaxHwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF4; c += 4 * F)
                                PoolingMaxHwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF2; c += 2 * F)
                                PoolingMaxHwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF1; c += 1 * F)
                                PoolingMaxHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            if (c < srcC)
                                PoolingMaxHwc1(ps + srcC - F, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + srcC - F);
                            dst += srcC;
                        }
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && srcH == dstH && srcW == dstW && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Neon::NeuralPooling1x1Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0 && dstW >= F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Neon::NeuralPooling2x2Max2x2(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0 && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Neon::NeuralPooling2x2Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                Base::SynetPoolingForwardMax32f(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void PoolingMaxNhwc1(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const uint8x16_t& min, uint8_t* dst)
        {
            uint8x16_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = vmaxq_u8(max0, Load<false>(ps + 0 * A));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * A, max0);
        }

        SIMD_INLINE void PoolingMaxNhwc2(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const uint8x16_t& min, uint8_t* dst)
        {
            uint8x16_t max0 = min;
            uint8x16_t max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = vmaxq_u8(max0, Load<false>(ps + 0 * A));
                    max1 = vmaxq_u8(max1, Load<false>(ps + 1 * A));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * A, max0);
            Store<false>(dst + 1 * A, max1);
        }

        SIMD_INLINE void PoolingMaxNhwc4(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const uint8x16_t& min, uint8_t* dst)
        {
            uint8x16_t max0 = min;
            uint8x16_t max1 = min;
            uint8x16_t max2 = min;
            uint8x16_t max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = vmaxq_u8(max0, Load<false>(ps + 0 * A));
                    max1 = vmaxq_u8(max1, Load<false>(ps + 1 * A));
                    max2 = vmaxq_u8(max2, Load<false>(ps + 2 * A));
                    max3 = vmaxq_u8(max3, Load<false>(ps + 3 * A));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * A, max0);
            Store<false>(dst + 1 * A, max1);
            Store<false>(dst + 2 * A, max2);
            Store<false>(dst + 3 * A, max3);
        }

        SIMD_INLINE void PoolingMaxNhwc8(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const uint8x16_t& min, uint8_t* dst)
        {
            uint8x16_t max0 = min;
            uint8x16_t max1 = min;
            uint8x16_t max2 = min;
            uint8x16_t max3 = min;
            uint8x16_t max4 = min;
            uint8x16_t max5 = min;
            uint8x16_t max6 = min;
            uint8x16_t max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = vmaxq_u8(max0, Load<false>(ps + 0 * A));
                    max1 = vmaxq_u8(max1, Load<false>(ps + 1 * A));
                    max2 = vmaxq_u8(max2, Load<false>(ps + 2 * A));
                    max3 = vmaxq_u8(max3, Load<false>(ps + 3 * A));
                    max4 = vmaxq_u8(max4, Load<false>(ps + 4 * A));
                    max5 = vmaxq_u8(max5, Load<false>(ps + 5 * A));
                    max6 = vmaxq_u8(max6, Load<false>(ps + 6 * A));
                    max7 = vmaxq_u8(max7, Load<false>(ps + 7 * A));
                }
                src += srcS;
            }
            Store<false>(dst + 0 * A, max0);
            Store<false>(dst + 1 * A, max1);
            Store<false>(dst + 2 * A, max2);
            Store<false>(dst + 3 * A, max3);
            Store<false>(dst + 4 * A, max4);
            Store<false>(dst + 5 * A, max5);
            Store<false>(dst + 6 * A, max6);
            Store<false>(dst + 7 * A, max7);
        }

        void SynetPoolingForwardMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= A)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCA1 = AlignLo(srcC, 1 * A);
                    size_t srcCA2 = AlignLo(srcC, 2 * A);
                    size_t srcCA4 = AlignLo(srcC, 4 * A);
                    size_t srcCA8 = AlignLo(srcC, 8 * A);
                    uint8x16_t min = vdupq_n_u8(0);
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
                            size_t c = 0;
                            for (; c < srcCA8; c += 8 * A)
                                PoolingMaxNhwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCA4; c += 4 * A)
                                PoolingMaxNhwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCA2; c += 2 * A)
                                PoolingMaxNhwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCA1; c += 1 * A)
                                PoolingMaxNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            if (c < srcC)
                                PoolingMaxNhwc1(ps + srcC - A, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + srcC - A);
                            dst += srcC;
                        }
                    }
                }
                else
                    Base::SynetPoolingForwardMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingForwardMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif// SIMD_NEON_ENABLE
}
