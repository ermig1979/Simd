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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetPoolingAverage(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
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
                        for (size_t c = 0; c < srcC; ++c)
                            dst[c] = 0.0f;
                        for (size_t h = hStart; h < hEnd; ++h)
                        {
                            for (size_t w = wStart; w < wEnd; ++w)
                            {
                                const float* pc = src + (h * srcW + w) * srcC;
                                for (size_t c = 0; c < srcC; ++c)
                                    dst[c] += pc[c];
                            }
                        }
                        if (excludePad)
                            for (size_t c = 0; c < srcC; ++c)
                                dst[c] = dst[c] / float((hEnd - hStart) * (wEnd - wStart));
                        else
                            for (size_t c = 0; c < srcC; ++c)
                                dst[c] = dst[c] / float(kernelY * kernelX);
                        dst += srcC;
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < srcC; ++c)
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
                            float sum = 0.0f;
                            for (size_t h = hStart; h < hEnd; ++h)
                                for (size_t w = wStart; w < wEnd; ++w)
                                    sum += src[h * srcW + w];
                            if (excludePad)
                                dst[ph * dstW + pw] = sum / float((hEnd - hStart) * (wEnd - wStart));
                            else
                                dst[ph * dstW + pw] = sum / float(kernelY * kernelX);
                        }
                    }
                    src += srcW * srcH;
                    dst += dstW * dstH;
                }
            }
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        template<class T> void SynetPoolingMax2D(const T* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, T* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
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
                        for (size_t c = 0; c < srcC; ++c)
                            dst[c] = std::numeric_limits<T>::lowest();
                        for (size_t sh = hBeg; sh < hEnd; ++sh)
                        {
                            for (size_t sw = wBeg; sw < wEnd; ++sw)
                            {
                                const T * ps = src + (sh * srcW + sw) * srcC;
                                for (size_t c = 0; c < srcC; ++c)
                                    dst[c] = Simd::Max(dst[c], ps[c]);
                            }
                        }
                        dst += srcC;
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < srcC; ++c)
                {
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
                            T max = std::numeric_limits<T>::lowest();;
                            for (size_t sh = hBeg; sh < hEnd; ++sh)
                                for (size_t sw = wBeg; sw < wEnd; ++sw)
                                    max = Simd::Max(max, src[sh * srcW + sw]);
                            dst[dh * dstW + dw] = max;
                        }
                    }
                    src += srcW * srcH;
                    dst += dstW * dstH;
                }
            }
            else
                assert(0);
        }

        template <class T> void SynetPoolingMax3D(const T* src, size_t srcC, size_t srcH, size_t srcW, 
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX, 
            size_t padC, size_t padY, size_t padX, T* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
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
                        for (size_t dc = 0; dc < dstC; ++dc)
                        {
                            size_t cBeg = dc * strideC - padC;
                            size_t cEnd = Simd::Min(cBeg + kernelC, srcC);
                            cBeg = Simd::Max<ptrdiff_t>(0, cBeg);
                            T max = std::numeric_limits<T>::lowest();
                            for (size_t sh = hBeg; sh < hEnd; ++sh)
                            {
                                for (size_t sw = wBeg; sw < wEnd; ++sw)
                                {
                                    const T* ps = src + (sh * srcW + sw) * srcC;
                                    for (size_t c = cBeg; c < cEnd; ++c)
                                        max = Simd::Max(max, ps[c]);
                                }
                            }
                            dst[(dh * dstW + dw) * dstC + dc] = max;
                        }
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t dc = 0; dc < dstC; ++dc)
                {
                    size_t cBeg = dc * strideC - padC;
                    size_t cEnd = Simd::Min(cBeg + kernelC, srcC);
                    cBeg = Simd::Max<ptrdiff_t>(0, cBeg);
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
                            T max = std::numeric_limits<T>::lowest();
                            for (size_t sc = cBeg; sc < cEnd; ++sc)
                                for (size_t sh = hBeg; sh < hEnd; ++sh)
                                    for (size_t sw = wBeg; sw < wEnd; ++sw)
                                        max = Simd::Max(max, src[(sc * srcH + sh) * srcW + sw]);
                            dst[(dc * dstH + dh) * dstW + dw] = max;
                        }
                    }
                }
            }
            else
                assert(0);
        }

        void SynetPoolingMax32f(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if(kernelC == 1 && strideC == 1 && padC == 0 && srcC == dstC)
                SynetPoolingMax2D(src, srcC, srcH, srcW, kernelY, kernelX, 
                    strideY, strideX, padY, padX, dst, dstH, dstW, format);
            else
                SynetPoolingMax3D(src, srcC, srcH, srcW, kernelC, kernelY, kernelX, 
                    strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }

        void SynetPoolingMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            SynetPoolingMax2D(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
        }
    }
#endif
}
