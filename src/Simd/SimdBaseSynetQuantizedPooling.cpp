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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void QuantizedPoolingAverageGlobal(const uint8_t* src, int srcZero, const float *srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            constexpr int min = std::numeric_limits<uint8_t>::min();
            constexpr int max = std::numeric_limits<uint8_t>::max();
            float norm = srcScale[0] / (dstScale[0] * float(spatial));
            if (format == SimdTensorFormatNhwc)
            {
                Array8u sum(channels);
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t c = 0; c < channels; ++c)
                        sum[c] = 0;
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            sum[c] += src[c];
                    }
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = (uint8_t)QuantizeSumLinear(sum[c], bias, norm, dstZero, min, max);
                    src += spatial * channels;
                    dst += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        int32_t sum = 0;
                        for (size_t s = 0; s < spatial; ++s)
                            sum += src[s];
                        dst[0] = (uint8_t)QuantizeSumLinear(sum, bias, norm, dstZero, min, max);
                        src += spatial;
                        dst += 1;
                    }
                }
            }
            else
                assert(0);
        }

        void SynetQuantizedPoolingAverage(const uint8_t* src, const float* srcScale, int srcZero, size_t batch, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, SimdBool excludePad,
            uint8_t* dst, const float* dstScale, int dstZero, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
            {
                QuantizedPoolingAverageGlobal(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero, format);
                return;
            }
            int32_t bias = -srcZero * int32_t(kernelY * kernelX);
            constexpr int min = std::numeric_limits<uint8_t>::min();
            constexpr int max = std::numeric_limits<uint8_t>::max();
            float norm = srcScale[0] / (dstScale[0] * float(kernelY * kernelX));
            if (format == SimdTensorFormatNhwc)
            {
                Array8u sum(srcC);
                for (size_t b = 0; b < batch; ++b)
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
                                sum[c] = 0;
                            for (size_t h = hStart; h < hEnd; ++h)
                            {
                                for (size_t w = wStart; w < wEnd; ++w)
                                {
                                    const uint8_t* ps = src + (h * srcW + w) * srcC;
                                    for (size_t c = 0; c < srcC; ++c)
                                        sum[c] += ps[c];
                                }
                            }
                            if (excludePad)
                                for (size_t c = 0; c < srcC; ++c)
                                {
                                    int area = int(hEnd - hStart) * int(wEnd - wStart), bias = -srcZero * area;
                                    float norm = srcScale[0] / (dstScale[0] * float(area));
                                    dst[c] = (uint8_t)QuantizeSumLinear(sum[c], bias, norm, dstZero, min, max);
                                }
                            else
                                for (size_t c = 0; c < srcC; ++c)
                                    dst[c] = (uint8_t)QuantizeSumLinear(sum[c], bias, norm, dstZero, min, max);
                            dst += srcC;
                        }
                    }
                    src += srcC * srcH * srcW;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t b = 0; b < batch; ++b)
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
                                int32_t sum = 0;
                                for (size_t h = hStart; h < hEnd; ++h)
                                    for (size_t w = wStart; w < wEnd; ++w)
                                        sum += src[h * srcW + w];
                                if (excludePad)
                                {
                                    int area = int(hEnd - hStart) * int(wEnd - wStart), bias = -srcZero * area;
                                    float norm = srcScale[0] / (dstScale[0] * float(area));
                                    dst[ph * dstW + pw] = (uint8_t)QuantizeSumLinear(sum, bias, norm, dstZero, min, max);
                                }
                                else
                                    dst[ph * dstW + pw] = (uint8_t)QuantizeSumLinear(sum, bias, norm, dstZero, min, max);
                            }
                        }
                        src += srcW * srcH;
                        dst += dstW * dstH;
                    }
                }
            }
            else
                assert(0);
        }
    }
#endif
}
