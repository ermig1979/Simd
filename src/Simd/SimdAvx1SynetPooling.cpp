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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx
    {
        SIMD_INLINE void PoolingAverageNhwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256& norm, float* dst)
        {
            __m256 sum0 = _mm256_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, _mm256_mul_ps(sum0, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256& norm, float* dst)
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, _mm256_mul_ps(sum0, norm));
            _mm256_storeu_ps(dst + 1 * F, _mm256_mul_ps(sum1, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256& norm, float* dst)
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                    sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(src + w * srcC + 2 * F));
                    sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, _mm256_mul_ps(sum0, norm));
            _mm256_storeu_ps(dst + 1 * F, _mm256_mul_ps(sum1, norm));
            _mm256_storeu_ps(dst + 2 * F, _mm256_mul_ps(sum2, norm));
            _mm256_storeu_ps(dst + 3 * F, _mm256_mul_ps(sum3, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256& norm, float* dst)
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            __m256 sum4 = _mm256_setzero_ps();
            __m256 sum5 = _mm256_setzero_ps();
            __m256 sum6 = _mm256_setzero_ps();
            __m256 sum7 = _mm256_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                    sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(src + w * srcC + 2 * F));
                    sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(src + w * srcC + 3 * F));
                    sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(src + w * srcC + 4 * F));
                    sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(src + w * srcC + 5 * F));
                    sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(src + w * srcC + 6 * F));
                    sum7 = _mm256_add_ps(sum7, _mm256_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, _mm256_mul_ps(sum0, norm));
            _mm256_storeu_ps(dst + 1 * F, _mm256_mul_ps(sum1, norm));
            _mm256_storeu_ps(dst + 2 * F, _mm256_mul_ps(sum2, norm));
            _mm256_storeu_ps(dst + 3 * F, _mm256_mul_ps(sum3, norm));
            _mm256_storeu_ps(dst + 4 * F, _mm256_mul_ps(sum4, norm));
            _mm256_storeu_ps(dst + 5 * F, _mm256_mul_ps(sum5, norm));
            _mm256_storeu_ps(dst + 6 * F, _mm256_mul_ps(sum6, norm));
            _mm256_storeu_ps(dst + 7 * F, _mm256_mul_ps(sum7, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc(const float* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const __m256& norm, float* dst)
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
                        __m256 norm = _mm256_set1_ps(1.0f / (kernelY * kernelX));
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
                                __m256 norm = _mm256_set1_ps(1.0f / (kH * kW));
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst);
                                dst += srcC;
                            }
                        }
                    }
                    else
                    {
                        __m256 norm = _mm256_set1_ps(1.0f / (kernelY * kernelX));
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
            Sse2::SynetPoolingForwardAverage(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, excludePad, format);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void PoolingMaxHwc1(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float * ps = src + w * srcC;
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(ps + 0 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
        }

        SIMD_INLINE void PoolingMaxHwc2(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(ps + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(ps + 1 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMaxHwc4(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            __m256 max2 = min;
            __m256 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(ps + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(ps + 1 * F));
                    max2 = _mm256_max_ps(max2, _mm256_loadu_ps(ps + 2 * F));
                    max3 = _mm256_max_ps(max3, _mm256_loadu_ps(ps + 3 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
            _mm256_storeu_ps(dst + 2 * F, max2);
            _mm256_storeu_ps(dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMaxHwc8(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            __m256 max2 = min;
            __m256 max3 = min;
            __m256 max4 = min;
            __m256 max5 = min;
            __m256 max6 = min;
            __m256 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(ps + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(ps + 1 * F));
                    max2 = _mm256_max_ps(max2, _mm256_loadu_ps(ps + 2 * F));
                    max3 = _mm256_max_ps(max3, _mm256_loadu_ps(ps + 3 * F));
                    max4 = _mm256_max_ps(max4, _mm256_loadu_ps(ps + 4 * F));
                    max5 = _mm256_max_ps(max5, _mm256_loadu_ps(ps + 5 * F));
                    max6 = _mm256_max_ps(max6, _mm256_loadu_ps(ps + 6 * F));
                    max7 = _mm256_max_ps(max7, _mm256_loadu_ps(ps + 7 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
            _mm256_storeu_ps(dst + 2 * F, max2);
            _mm256_storeu_ps(dst + 3 * F, max3);
            _mm256_storeu_ps(dst + 4 * F, max4);
            _mm256_storeu_ps(dst + 5 * F, max5);
            _mm256_storeu_ps(dst + 6 * F, max6);
            _mm256_storeu_ps(dst + 7 * F, max7);
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
                    __m256 min = _mm256_set1_ps(-FLT_MAX);
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
                if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0 && dstW >= F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Avx::NeuralPooling2x2Max2x2(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                Sse2::SynetPoolingForwardMax32f(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif// SIMD_AVX_ENABLE
}
