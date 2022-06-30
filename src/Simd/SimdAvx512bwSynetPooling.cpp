/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE void PoolingAverageNhwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& norm, float* dst, __mmask16 tail = -1)
        {
            __m512 sum0 = _mm512_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = _mm512_add_ps(sum0, _mm512_maskz_loadu_ps(tail, ps + 0 * F));
                }
                src += srcS;
            }
            _mm512_mask_storeu_ps(dst + 0 * F, tail, _mm512_mul_ps(sum0, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& norm, float* dst)
        {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(ps + 0 * F));
                    sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(ps + 1 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, _mm512_mul_ps(sum0, norm));
            _mm512_storeu_ps(dst + 1 * F, _mm512_mul_ps(sum1, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& norm, float* dst)
        {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(ps + 0 * F));
                    sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(ps + 1 * F));
                    sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(ps + 2 * F));
                    sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(ps + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, _mm512_mul_ps(sum0, norm));
            _mm512_storeu_ps(dst + 1 * F, _mm512_mul_ps(sum1, norm));
            _mm512_storeu_ps(dst + 2 * F, _mm512_mul_ps(sum2, norm));
            _mm512_storeu_ps(dst + 3 * F, _mm512_mul_ps(sum3, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& norm, float* dst)
        {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
            __m512 sum4 = _mm512_setzero_ps();
            __m512 sum5 = _mm512_setzero_ps();
            __m512 sum6 = _mm512_setzero_ps();
            __m512 sum7 = _mm512_setzero_ps();
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const float* ps = src + w * srcC;
                    sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(ps + 0 * F));
                    sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(ps + 1 * F));
                    sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(ps + 2 * F));
                    sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(ps + 3 * F));
                    sum4 = _mm512_add_ps(sum4, _mm512_loadu_ps(ps + 4 * F));
                    sum5 = _mm512_add_ps(sum5, _mm512_loadu_ps(ps + 5 * F));
                    sum6 = _mm512_add_ps(sum6, _mm512_loadu_ps(ps + 6 * F));
                    sum7 = _mm512_add_ps(sum7, _mm512_loadu_ps(ps + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, _mm512_mul_ps(sum0, norm));
            _mm512_storeu_ps(dst + 1 * F, _mm512_mul_ps(sum1, norm));
            _mm512_storeu_ps(dst + 2 * F, _mm512_mul_ps(sum2, norm));
            _mm512_storeu_ps(dst + 3 * F, _mm512_mul_ps(sum3, norm));
            _mm512_storeu_ps(dst + 4 * F, _mm512_mul_ps(sum4, norm));
            _mm512_storeu_ps(dst + 5 * F, _mm512_mul_ps(sum5, norm));
            _mm512_storeu_ps(dst + 6 * F, _mm512_mul_ps(sum6, norm));
            _mm512_storeu_ps(dst + 7 * F, _mm512_mul_ps(sum7, norm));
        }

        SIMD_INLINE void PoolingAverageNhwc(const float* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const __m512& norm, float* dst, __mmask16 tail)
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
                PoolingAverageNhwc1(src + c, srcS, srcC, kernelY, kernelX, norm, dst + c, tail);
        }

        void SynetPoolingAverage(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC > Avx::F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    __mmask16 tail = TailMask16(srcC - srcCF1);
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        __m512 norm = _mm512_set1_ps(1.0f / (kernelY * kernelX));
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const float* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kernelY, kernelX, norm, dst, tail);
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
                                __m512 norm = _mm512_set1_ps(1.0f / (kH * kW));
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst, tail);
                                dst += srcC;
                            }
                        }
                    }
                    else
                    {
                        __m512 norm = _mm512_set1_ps(1.0f / (kernelY * kernelX));
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
                                PoolingAverageNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, norm, dst, tail);
                                dst += srcC;
                            }
                        }
                    }
                    return;
                }
            }
            Avx::SynetPoolingAverage(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, excludePad, format);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void PoolingMax32f2DHwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst, __mmask16 tail = -1)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_maskz_loadu_ps(tail, src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm512_mask_storeu_ps(dst + 0 * F, tail, max0);
        }

        SIMD_INLINE void PoolingMax32f2DHwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMax32f2DHwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
            _mm512_storeu_ps(dst + 2 * F, max2);
            _mm512_storeu_ps(dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMax32f2DHwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
            _mm512_storeu_ps(dst + 2 * F, max2);
            _mm512_storeu_ps(dst + 3 * F, max3);
            _mm512_storeu_ps(dst + 4 * F, max4);
            _mm512_storeu_ps(dst + 5 * F, max5);
            _mm512_storeu_ps(dst + 6 * F, max6);
            _mm512_storeu_ps(dst + 7 * F, max7);
        }

        void SynetPoolingMax32f2D(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                size_t srcS = srcW * srcC;
                size_t srcCF1 = AlignLo(srcC, 1 * F);
                size_t srcCF2 = AlignLo(srcC, 2 * F);
                size_t srcCF4 = AlignLo(srcC, 4 * F);
                size_t srcCF8 = AlignLo(srcC, 8 * F);
                __m512 min = _mm512_set1_ps(-FLT_MAX);
                __mmask16 tail = TailMask16(srcC - srcCF1);
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
                            PoolingMax32f2DHwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF4; c += 4 * F)
                            PoolingMax32f2DHwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF2; c += 2 * F)
                            PoolingMax32f2DHwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF1; c += 1 * F)
                            PoolingMax32f2DHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        if (c < srcC)
                            PoolingMax32f2DHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c, tail);
                        dst += srcC;
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && srcH == dstH && srcW == dstW && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        NeuralPooling1x1Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0 && dstW >= F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        NeuralPooling2x2Max2x2(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0 && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        NeuralPooling2x2Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                Avx2::SynetPoolingMax32f(src, srcC, srcH, srcW, 1, kernelY, kernelX, 1, strideY, strideX, 0, padY, padX, dst, srcC, dstH, dstW, format);
            }
            else
                assert(0);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512 MaxNhwcCr2(__m512 lo, __m512 hi)
        {
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst, _mm512_castps512_ps256(MaxNhwcCr2(max0, _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst, MaxNhwcCr2(max0, max1));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(max0, max1));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(max2, max3));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(max0, max1));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(max2, max3));
            _mm512_storeu_ps(dst + 2 * F, MaxNhwcCr2(max4, max5));
            _mm512_storeu_ps(dst + 3 * F, MaxNhwcCr2(max6, max7));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm_storeu_ps(dst, _mm512_castps512_ps128(MaxNhwcCr2(MaxNhwcCr2(max0, _mm512_setzero_ps()), _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst, _mm512_castps512_ps256(MaxNhwcCr2(MaxNhwcCr2(max0, max1), _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(MaxNhwcCr2(max0, max1), MaxNhwcCr2(max2, max3)));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(MaxNhwcCr2(max0, max1), MaxNhwcCr2(max2, max3)));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(MaxNhwcCr2(max4, max5), MaxNhwcCr2(max6, max7)));
        }

        void SynetPoolingMax32f3D(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc && srcC >= F)
            {
                if (kernelC == 2 && strideC == 2 && padC == 0 && srcC == dstC * 2)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcC16 = AlignLo(srcC, 16);
                    size_t srcC32 = AlignLo(srcC, 32);
                    size_t srcC64 = AlignLo(srcC, 64);
                    size_t srcC128 = AlignLo(srcC, 128);
                    __m512 min = _mm512_set1_ps(-FLT_MAX);
                    for (size_t dh = 0; dh < dstH; ++dh)
                    {
                        size_t hBeg = dh * strideY - padY;
                        size_t hEnd = Simd::Min(hBeg + kernelY, srcH);
                        hBeg = Simd::Max<ptrdiff_t>(0, hBeg);
                        for (size_t dw = 0; dw < dstW; ++dw)
                        {
                            size_t wBeg = dw * strideX - padX;
                            size_t wEnd = Simd::Min(wBeg + kernelX, srcW);
                            wBeg = Simd::Max<ptrdiff_t>(0, wBeg);
                            const float* ps = src + hBeg * srcS + wBeg * srcC;
                            size_t c = 0, d = 0;
                            for (; c < srcC128; c += 128, d += 64)
                                PoolingMax32f3DNhwcCr2_8(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC64; c += 64, d += 32)
                                PoolingMax32f3DNhwcCr2_4(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC32; c += 32, d += 16)
                                PoolingMax32f3DNhwcCr2_2(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC16; c += 16, d += 8)
                                PoolingMax32f3DNhwcCr2_1(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            if (c < srcC)
                                PoolingMax32f3DNhwcCr2_1(ps + srcC - 16, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + dstC - 8);
                            dst += dstC;
                        }
                    }
                    return;
                }
                if (kernelC == 4 && strideC == 4 && padC == 0 && srcC == dstC * 4)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcC16 = AlignLo(srcC, 16);
                    size_t srcC32 = AlignLo(srcC, 32);
                    size_t srcC64 = AlignLo(srcC, 64);
                    size_t srcC128 = AlignLo(srcC, 128);
                    __m512 min = _mm512_set1_ps(-FLT_MAX);
                    for (size_t dh = 0; dh < dstH; ++dh)
                    {
                        size_t hBeg = dh * strideY - padY;
                        size_t hEnd = Simd::Min(hBeg + kernelY, srcH);
                        hBeg = Simd::Max<ptrdiff_t>(0, hBeg);
                        for (size_t dw = 0; dw < dstW; ++dw)
                        {
                            size_t wBeg = dw * strideX - padX;
                            size_t wEnd = Simd::Min(wBeg + kernelX, srcW);
                            wBeg = Simd::Max<ptrdiff_t>(0, wBeg);
                            const float* ps = src + hBeg * srcS + wBeg * srcC;
                            size_t c = 0, d = 0;
                            for (; c < srcC128; c += 128, d += 32)
                                PoolingMax32f3DNhwcCr4_8(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC64; c += 64, d += 16)
                                PoolingMax32f3DNhwcCr4_4(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC32; c += 32, d += 8)
                                PoolingMax32f3DNhwcCr4_2(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC16; c += 16, d += 4)
                                PoolingMax32f3DNhwcCr4_1(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            if (c < srcC)
                                PoolingMax32f3DNhwcCr4_1(ps + srcC - 16, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + dstC - 4);
                            dst += dstC;
                        }
                    }
                    return;
                }
            }
            Avx2::SynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX,
                strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }

        //-----------------------------------------------------------------------------------------

        void SynetPoolingMax32f(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (kernelC == 1 && strideC == 1 && padC == 0 && srcC == dstC)
                SynetPoolingMax32f2D(src, srcC, srcH, srcW, kernelY, kernelX,
                    strideY, strideX, padY, padX, dst, dstH, dstW, format);
            else
                SynetPoolingMax32f3D(src, srcC, srcH, srcW, kernelC, kernelY, kernelX,
                    strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void PoolingMax8uNhwc1(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst, __mmask64 tail = -1)
        {
            __m512i max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_epu8(max0, _mm512_maskz_loadu_epi8(tail, src + w * srcC));
                }
                src += srcS;
            }
            _mm512_mask_storeu_epi8(dst, tail, max0);
        }

        SIMD_INLINE void PoolingMax8uNhwc2(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
        {
            __m512i max0 = min;
            __m512i max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m512i* ps = (__m512i*)(src + w * srcC);
                    max0 = _mm512_max_epu8(max0, _mm512_loadu_si512(ps + 0));
                    max1 = _mm512_max_epu8(max1, _mm512_loadu_si512(ps + 1));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)dst + 0, max0);
            _mm512_storeu_si512((__m512i*)dst + 1, max1);
        }

        SIMD_INLINE void PoolingMax8uNhwc4(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
        {
            __m512i max0 = min;
            __m512i max1 = min;
            __m512i max2 = min;
            __m512i max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m512i* ps = (__m512i*)(src + w * srcC);
                    max0 = _mm512_max_epu8(max0, _mm512_loadu_si512(ps + 0));
                    max1 = _mm512_max_epu8(max1, _mm512_loadu_si512(ps + 1));
                    max2 = _mm512_max_epu8(max2, _mm512_loadu_si512(ps + 2));
                    max3 = _mm512_max_epu8(max3, _mm512_loadu_si512(ps + 3));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)dst + 0, max0);
            _mm512_storeu_si512((__m512i*)dst + 1, max1);
            _mm512_storeu_si512((__m512i*)dst + 2, max2);
            _mm512_storeu_si512((__m512i*)dst + 3, max3);
        }

        SIMD_INLINE void PoolingMax8uNhwc8(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
        {
            __m512i max0 = min;
            __m512i max1 = min;
            __m512i max2 = min;
            __m512i max3 = min;
            __m512i max4 = min;
            __m512i max5 = min;
            __m512i max6 = min;
            __m512i max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m512i* ps = (__m512i*)(src + w * srcC);
                    max0 = _mm512_max_epu8(max0, _mm512_loadu_si512(ps + 0));
                    max1 = _mm512_max_epu8(max1, _mm512_loadu_si512(ps + 1));
                    max2 = _mm512_max_epu8(max2, _mm512_loadu_si512(ps + 2));
                    max3 = _mm512_max_epu8(max3, _mm512_loadu_si512(ps + 3));
                    max4 = _mm512_max_epu8(max4, _mm512_loadu_si512(ps + 4));
                    max5 = _mm512_max_epu8(max5, _mm512_loadu_si512(ps + 5));
                    max6 = _mm512_max_epu8(max6, _mm512_loadu_si512(ps + 6));
                    max7 = _mm512_max_epu8(max7, _mm512_loadu_si512(ps + 7));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)dst + 0, max0);
            _mm512_storeu_si512((__m512i*)dst + 1, max1);
            _mm512_storeu_si512((__m512i*)dst + 2, max2);
            _mm512_storeu_si512((__m512i*)dst + 3, max3);
            _mm512_storeu_si512((__m512i*)dst + 4, max4);
            _mm512_storeu_si512((__m512i*)dst + 5, max5);
            _mm512_storeu_si512((__m512i*)dst + 6, max6);
            _mm512_storeu_si512((__m512i*)dst + 7, max7);
        }

        void SynetPoolingMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                size_t srcS = srcW * srcC;
                size_t srcCA1 = AlignLo(srcC, 1 * A);
                size_t srcCA2 = AlignLo(srcC, 2 * A);
                size_t srcCA4 = AlignLo(srcC, 4 * A);
                size_t srcCA8 = AlignLo(srcC, 8 * A);
                __mmask64 tail = TailMask64(srcC - srcCA1);
                __m512i min = _mm512_set1_epi8(0);
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
                            PoolingMax8uNhwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA4; c += 4 * A)
                            PoolingMax8uNhwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA2; c += 2 * A)
                            PoolingMax8uNhwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA1; c += 1 * A)
                            PoolingMax8uNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        if (c < srcC)
                            PoolingMax8uNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c, tail);
                        dst += srcC;
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
