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
        SIMD_INLINE void PoolingMax8uNhwc1(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svuint8_t& min, uint8_t* dst, const svbool_t& mask)
        {
            svuint8_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = svmax_u8_x(mask, max0, svld1_u8(mask, ps));
                }
                src += srcS;
            }
            svst1_u8(mask, dst, max0);
        }

        SIMD_INLINE void PoolingMax8uNhwc2(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svuint8_t& min, uint8_t* dst, const svbool_t& mask)
        {
            const size_t A = svcntb();
            svuint8_t max0 = min;
            svuint8_t max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = svmax_u8_x(mask, max0, svld1_u8(mask, ps + 0 * A));
                    max1 = svmax_u8_x(mask, max1, svld1_u8(mask, ps + 1 * A));
                }
                src += srcS;
            }
            svst1_u8(mask, dst + 0 * A, max0);
            svst1_u8(mask, dst + 1 * A, max1);
        }

        SIMD_INLINE void PoolingMax8uNhwc4(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svuint8_t& min, uint8_t* dst, const svbool_t& mask)
        {
            const size_t A = svcntb();
            svuint8_t max0 = min;
            svuint8_t max1 = min;
            svuint8_t max2 = min;
            svuint8_t max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = svmax_u8_x(mask, max0, svld1_u8(mask, ps + 0 * A));
                    max1 = svmax_u8_x(mask, max1, svld1_u8(mask, ps + 1 * A));
                    max2 = svmax_u8_x(mask, max2, svld1_u8(mask, ps + 2 * A));
                    max3 = svmax_u8_x(mask, max3, svld1_u8(mask, ps + 3 * A));
                }
                src += srcS;
            }
            svst1_u8(mask, dst + 0 * A, max0);
            svst1_u8(mask, dst + 1 * A, max1);
            svst1_u8(mask, dst + 2 * A, max2);
            svst1_u8(mask, dst + 3 * A, max3);
        }

        SIMD_INLINE void PoolingMax8uNhwc8(const uint8_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svuint8_t& min, uint8_t* dst, const svbool_t& mask)
        {
            const size_t A = svcntb();
            svuint8_t max0 = min;
            svuint8_t max1 = min;
            svuint8_t max2 = min;
            svuint8_t max3 = min;
            svuint8_t max4 = min;
            svuint8_t max5 = min;
            svuint8_t max6 = min;
            svuint8_t max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint8_t* ps = src + w * srcC;
                    max0 = svmax_u8_x(mask, max0, svld1_u8(mask, ps + 0 * A));
                    max1 = svmax_u8_x(mask, max1, svld1_u8(mask, ps + 1 * A));
                    max2 = svmax_u8_x(mask, max2, svld1_u8(mask, ps + 2 * A));
                    max3 = svmax_u8_x(mask, max3, svld1_u8(mask, ps + 3 * A));
                    max4 = svmax_u8_x(mask, max4, svld1_u8(mask, ps + 4 * A));
                    max5 = svmax_u8_x(mask, max5, svld1_u8(mask, ps + 5 * A));
                    max6 = svmax_u8_x(mask, max6, svld1_u8(mask, ps + 6 * A));
                    max7 = svmax_u8_x(mask, max7, svld1_u8(mask, ps + 7 * A));
                }
                src += srcS;
            }
            svst1_u8(mask, dst + 0 * A, max0);
            svst1_u8(mask, dst + 1 * A, max1);
            svst1_u8(mask, dst + 2 * A, max2);
            svst1_u8(mask, dst + 3 * A, max3);
            svst1_u8(mask, dst + 4 * A, max4);
            svst1_u8(mask, dst + 5 * A, max5);
            svst1_u8(mask, dst + 6 * A, max6);
            svst1_u8(mask, dst + 7 * A, max7);
        }

        SIMD_INLINE void PoolingMax8uNhwc(const uint8_t* src, size_t srcS, size_t srcC, size_t srcCA1,
            size_t srcCA2, size_t srcCA4, size_t srcCA8, size_t kernelY, size_t kernelX, const svuint8_t& min, uint8_t* dst, const svbool_t& body)
        {
            const size_t A = svcntb();
            size_t c = 0;
            for (; c < srcCA8; c += 8 * A)
                PoolingMax8uNhwc8(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCA4; c += 4 * A)
                PoolingMax8uNhwc4(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCA2; c += 2 * A)
                PoolingMax8uNhwc2(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCA1; c += 1 * A)
                PoolingMax8uNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            if (c < srcC)
                PoolingMax8uNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, svwhilelt_b8(c, srcC));
        }

        void SynetPoolingMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            const size_t A = svcntb();
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= A)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCA1 = AlignLo(srcC, 1 * A);
                    size_t srcCA2 = AlignLo(srcC, 2 * A);
                    size_t srcCA4 = AlignLo(srcC, 4 * A);
                    size_t srcCA8 = AlignLo(srcC, 8 * A);
                    svbool_t body = svptrue_b8();
                    svuint8_t min = svdup_n_u8(0);
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const uint8_t* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingMax8uNhwc(ps, srcS, srcC, srcCA1, srcCA2, srcCA4, srcCA8, kernelY, kernelX, min, dst, body);
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
                                const uint8_t* ps = src + hStart * srcS + wStart * srcC;
                                PoolingMax8uNhwc(ps, srcS, srcC, srcCA1, srcCA2, srcCA4, srcCA8, kH, kW, min, dst, body);
                                dst += srcC;
                            }
                        }
                    }
                    return;
                }
            }
            Base::SynetPoolingMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
        }
    }
#endif
}
