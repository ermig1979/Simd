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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE void PoolingMaxNhwc1(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst, __mmask64 tail = -1)
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

        SIMD_INLINE void PoolingMaxNhwc2(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
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

        SIMD_INLINE void PoolingMaxNhwc4(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
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

        SIMD_INLINE void PoolingMaxNhwc8(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512i& min, uint8_t* dst)
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

        void SynetPoolingForwardMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
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
                            PoolingMaxNhwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA4; c += 4 * A)
                            PoolingMaxNhwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA2; c += 2 * A)
                            PoolingMaxNhwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCA1; c += 1 * A)
                            PoolingMaxNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        if (c < srcC)
                            PoolingMaxNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c, tail);
                        dst += srcC;
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingForwardMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
