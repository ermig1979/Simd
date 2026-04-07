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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE void PoolingMax16bNhwc1(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, uint16_t* dst, __mmask16 tail = -1)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    __m256i src0 = _mm256_maskz_loadu_epi16(tail, src + w * srcC);
                    max0 = _mm512_max_ps(max0, BFloat16ToFloat32(src0));
                }
                src += srcS;
            }
            _mm256_mask_storeu_epi16(dst + 0 * DF, tail, PackFloat32ToBFloat16(max0));
        }

        SIMD_INLINE void PoolingMax16bNhwc2(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, uint16_t* dst)
        {
            __m512 max0 = min, max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    __m512i src01 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 0 * DF));
                    max0 = _mm512_max_ps(max0, BFloat16ToFloat32Even(src01));
                    max1 = _mm512_max_ps(max1, BFloat16ToFloat32Odd(src01));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)(dst + 0 * DF), Float32ToBFloat16Interlived(max0, max1));
        }

        SIMD_INLINE void PoolingMax16bNhwc4(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, uint16_t* dst)
        {
            __m512 max0 = min, max1 = min;
            __m512 max2 = min, max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    __m512i src01 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 0 * DF));
                    max0 = _mm512_max_ps(max0, BFloat16ToFloat32Even(src01));
                    max1 = _mm512_max_ps(max1, BFloat16ToFloat32Odd(src01));
                    __m512i src23 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 1 * DF));
                    max2 = _mm512_max_ps(max2, BFloat16ToFloat32Even(src23));
                    max3 = _mm512_max_ps(max3, BFloat16ToFloat32Odd(src23));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)(dst + 0 * DF), Float32ToBFloat16Interlived(max0, max1));
            _mm512_storeu_si512((__m512i*)(dst + 1 * DF), Float32ToBFloat16Interlived(max2, max3));
        }

        SIMD_INLINE void PoolingMax16bNhwc8(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, uint16_t* dst)
        {
            __m512 max0 = min, max1 = min;
            __m512 max2 = min, max3 = min;
            __m512 max4 = min, max5 = min;
            __m512 max6 = min, max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    __m512i src01 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 0 * DF));
                    max0 = _mm512_max_ps(max0, BFloat16ToFloat32Even(src01));
                    max1 = _mm512_max_ps(max1, BFloat16ToFloat32Odd(src01));
                    __m512i src23 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 1 * DF));
                    max2 = _mm512_max_ps(max2, BFloat16ToFloat32Even(src23));
                    max3 = _mm512_max_ps(max3, BFloat16ToFloat32Odd(src23));
                    __m512i src45 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 2 * DF));
                    max4 = _mm512_max_ps(max4, BFloat16ToFloat32Even(src45));
                    max5 = _mm512_max_ps(max5, BFloat16ToFloat32Odd(src45));
                    __m512i src67 = _mm512_loadu_si512((__m512i*)(src + w * srcC + 3 * DF));
                    max6 = _mm512_max_ps(max6, BFloat16ToFloat32Even(src67));
                    max7 = _mm512_max_ps(max7, BFloat16ToFloat32Odd(src67));
                }
                src += srcS;
            }
            _mm512_storeu_si512((__m512i*)(dst + 0 * DF), Float32ToBFloat16Interlived(max0, max1));
            _mm512_storeu_si512((__m512i*)(dst + 1 * DF), Float32ToBFloat16Interlived(max2, max3));
            _mm512_storeu_si512((__m512i*)(dst + 2 * DF), Float32ToBFloat16Interlived(max4, max5));
            _mm512_storeu_si512((__m512i*)(dst + 3 * DF), Float32ToBFloat16Interlived(max6, max7));
        }

        void SynetPoolingMax16b(const uint16_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint16_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
#if (defined(_MSC_VER) && !defined(NDEBUG) && defined(SIMD_X86_ENABLE)) || defined(__clang__)
                Avx2::SynetPoolingMax16b(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
#else
                Array32f max(srcC);
                size_t srcS = srcW * srcC;
                size_t srcCF1 = AlignLo(srcC, 1 * F);
                size_t srcCF2 = AlignLo(srcC, 2 * F);
                size_t srcCF4 = AlignLo(srcC, 4 * F);
                size_t srcCF8 = AlignLo(srcC, 8 * F);
                __mmask16 tail = TailMask16(srcC - srcCF1);
                __m512 min = _mm512_set1_ps(-FLT_MAX);
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
                        const uint16_t* ps = src + hStart * srcS + wStart * srcC;
                        size_t c = 0;
                        for (; c < srcCF8; c += 8 * F)
                            PoolingMax16bNhwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF4; c += 4 * F)
                            PoolingMax16bNhwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF2; c += 2 * F)
                            PoolingMax16bNhwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        for (; c < srcCF1; c += 1 * F)
                            PoolingMax16bNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                        if (c < srcC)
                            PoolingMax16bNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c, tail);
                        dst += srcC;
                    }
                }
#endif
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingMax16b(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif
}
