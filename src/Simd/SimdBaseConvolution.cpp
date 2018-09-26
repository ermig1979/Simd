/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdConvolution.h"

namespace Simd
{
    namespace Base
    {
        void ConvolutionGemm::ImgToCol(const float * src, const ConvParam & p, float * dst)
        {
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy*p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX*p.dilationY*p.strideX*p.strideY != 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx * p.dilationX - p.padX;
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = 0;
                                        sx += p.strideX;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = 0;
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                const ptrdiff_t bodySize = p.dstW - p.padX - p.padW;
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {
                            size_t sy = ky - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy, ++sy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx - p.padX, dx = 0;
                                    const float * psrc = src + sy * p.srcW;
                                    for (; dx < p.padX; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = 0;
                                    }
                                    if (bodySize > 0)
                                    {
                                        memcpy(dst, psrc + sx, bodySize * sizeof(float));
                                        dst += bodySize;
                                        dx += bodySize;
                                        sx += bodySize;
                                    }
                                    for (; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = 0;
                                    }
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                    dst += p.dstW;
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
        }

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            return new ConvolutionGemm(param);
        }
    }
}
