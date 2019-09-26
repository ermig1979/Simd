/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __TestSynetConvolutionParam_h__
#define __TestSynetConvolutionParam_h__

#include "Test/TestConfig.h"

namespace Test
{
    template<bool back> struct SynetConvolutionParam
    {
        SimdBool trans;
        size_t batch;
        SimdConvolutionParameters conv;

        SynetConvolutionParam(SimdBool t, size_t n, size_t sC, size_t sH, size_t sW, size_t dC, size_t kY, size_t kX, size_t dY, size_t dX,
            size_t sY, size_t sX, size_t pY, size_t pX, size_t pH, size_t pW, size_t g, ::SimdConvolutionActivationType a)
        {
            trans = t;
            batch = n;
            conv.srcC = sC;
            conv.srcH = sH;
            conv.srcW = sW;
            conv.srcT = SimdTensorData32f;
            conv.srcF = trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
            conv.dstC = dC;
            conv.dstT = SimdTensorData32f;
            conv.dstF = trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
            conv.kernelY = kY;
            conv.kernelX = kX;
            conv.dilationY = dY;
            conv.dilationX = dX;
            conv.strideY = sY;
            conv.strideX = sX;
            conv.padY = pY;
            conv.padX = pX;
            conv.padH = pH;
            conv.padW = pW;
            conv.group = g;
            conv.activation = a;
            if (back)
            {
                conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
                conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
            }
            else
            {
                conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
                conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
            }
        }

        SynetConvolutionParam(size_t n, size_t sC, size_t sH, size_t sW, size_t dC, Size k, Size d, Size s, Size b, Size e, size_t g, ::SimdConvolutionActivationType a, ::SimdBool t)
        {
            trans = t;
            batch = n;
            conv.srcC = sC;
            conv.srcH = sH;
            conv.srcW = sW;
            conv.srcT = SimdTensorData32f;
            conv.srcF = trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
            conv.dstC = dC;
            conv.dstT = SimdTensorData32f;
            conv.dstF = trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
            conv.kernelY = k.y;
            conv.kernelX = k.x;
            conv.dilationY = d.y;
            conv.dilationX = d.x;
            conv.strideY = s.y;
            conv.strideX = s.y;
            conv.padY = b.y;
            conv.padX = b.x;
            conv.padH = e.y;
            conv.padW = e.x;
            conv.group = g;
            conv.activation = a;
            if (back)
            {
                conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
                conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
            }
            else
            {
                conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
                conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
            }
        }

        String Decription() const
        {
            std::stringstream ss;
            ss << "[" << this->batch << "x" << conv.srcC << "x" << conv.srcH << "x" << conv.srcW;
            ss << "-" << conv.dstC << "x" << conv.kernelY << "x" << conv.kernelX;
            ss << "-" << conv.strideX << "-" << Simd::Max(conv.padX, conv.padW) << "-" << conv.group << "-" << this->trans;
            ss << "]";
            return ss.str();
        }
    };
}

#endif//__TestSynetConvolutionParam_h__
