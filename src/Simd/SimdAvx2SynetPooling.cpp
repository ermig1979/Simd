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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx2
    {
        void SynetPoolingForwardMax32f(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
            {
                if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && srcH == dstH && srcW == dstW && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Avx2::NeuralPooling1x1Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0 && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Avx2::NeuralPooling2x2Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
            }
            Avx::SynetPoolingForwardMax32f(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void PoolingMaxNhwc1(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256i& min, uint8_t* dst)
        {
            __m256i max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m256i* ps = (__m256i*)(src + w * srcC);
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(ps + 0));
                }
                src += srcS;
            }
            _mm256_storeu_si256((__m256i*)dst + 0, max0);
        }

        SIMD_INLINE void PoolingMaxNhwc2(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256i& min, uint8_t* dst)
        {
            __m256i max0 = min;
            __m256i max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m256i* ps = (__m256i*)(src + w * srcC);
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(ps + 0));
                    max1 = _mm256_max_epu8(max1, _mm256_loadu_si256(ps + 1));
                }
                src += srcS;
            }
            _mm256_storeu_si256((__m256i*)dst + 0, max0);
            _mm256_storeu_si256((__m256i*)dst + 1, max1);
        }

        SIMD_INLINE void PoolingMaxNhwc4(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256i& min, uint8_t* dst)
        {
            __m256i max0 = min;
            __m256i max1 = min;
            __m256i max2 = min;
            __m256i max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m256i* ps = (__m256i*)(src + w * srcC);
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(ps + 0));
                    max1 = _mm256_max_epu8(max1, _mm256_loadu_si256(ps + 1));
                    max2 = _mm256_max_epu8(max2, _mm256_loadu_si256(ps + 2));
                    max3 = _mm256_max_epu8(max3, _mm256_loadu_si256(ps + 3));
                }
                src += srcS;
            }
            _mm256_storeu_si256((__m256i*)dst + 0, max0);
            _mm256_storeu_si256((__m256i*)dst + 1, max1);
            _mm256_storeu_si256((__m256i*)dst + 2, max2);
            _mm256_storeu_si256((__m256i*)dst + 3, max3);
        }

        SIMD_INLINE void PoolingMaxNhwc8(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256i& min, uint8_t* dst)
        {
            __m256i max0 = min;
            __m256i max1 = min;
            __m256i max2 = min;
            __m256i max3 = min;
            __m256i max4 = min;
            __m256i max5 = min;
            __m256i max6 = min;
            __m256i max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const __m256i* ps = (__m256i*)(src + w * srcC);
                    max0 = _mm256_max_epu8(max0, _mm256_loadu_si256(ps + 0));
                    max1 = _mm256_max_epu8(max1, _mm256_loadu_si256(ps + 1));
                    max2 = _mm256_max_epu8(max2, _mm256_loadu_si256(ps + 2));
                    max3 = _mm256_max_epu8(max3, _mm256_loadu_si256(ps + 3));
                    max4 = _mm256_max_epu8(max4, _mm256_loadu_si256(ps + 4));
                    max5 = _mm256_max_epu8(max5, _mm256_loadu_si256(ps + 5));
                    max6 = _mm256_max_epu8(max6, _mm256_loadu_si256(ps + 6));
                    max7 = _mm256_max_epu8(max7, _mm256_loadu_si256(ps + 7));
                }
                src += srcS;
            }
            _mm256_storeu_si256((__m256i*)dst + 0, max0);
            _mm256_storeu_si256((__m256i*)dst + 1, max1);
            _mm256_storeu_si256((__m256i*)dst + 2, max2);
            _mm256_storeu_si256((__m256i*)dst + 3, max3);
            _mm256_storeu_si256((__m256i*)dst + 4, max4);
            _mm256_storeu_si256((__m256i*)dst + 5, max5);
            _mm256_storeu_si256((__m256i*)dst + 6, max6);
            _mm256_storeu_si256((__m256i*)dst + 7, max7);
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
                    __m256i min = _mm256_set1_epi8(0);
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
                    Sse41::SynetPoolingForwardMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingForwardMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
