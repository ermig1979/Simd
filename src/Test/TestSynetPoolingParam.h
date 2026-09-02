/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __TestSynetPoolingParam_h__
#define __TestSynetPoolingParam_h__

#include "Test/TestConfig.h"

namespace Test
{
    struct ParamPooling
    {
        size_t batch, srcC, srcH, srcW, kernelC, kernelY, kernelX, strideC, strideY, strideX, padC, padY, padX, dstC, dstH, dstW;
        SimdTensorFormatType format;
        SimdBool ceil, excludePad;

        ParamPooling(size_t b, size_t sC, size_t sH, size_t sW, Size k, Size s, Size pb, Size pe, ::SimdTensorFormatType f, ::SimdBool c, SimdBool ep)
            : batch(b), srcC(sC), srcH(sH), srcW(sW), kernelC(1), kernelY(k.y), kernelX(k.x), strideC(1), strideY(s.y), strideX(s.x)
            , padC(0), padY(pb.y), padX(pb.x), format(f), ceil(c), excludePad(ep)
        {
            SetDst(0, pe.y, pe.x);
        }

        ParamPooling(size_t b, size_t sC, size_t sH, size_t sW, const Shape& k, const Shape& s, const Shape& pb, const Shape& pe, ::SimdTensorFormatType f, ::SimdBool c, SimdBool ep)
            : batch(b), srcC(sC), srcH(sH), srcW(sW), kernelC(k[0]), kernelY(k[1]), kernelX(k[2]), strideC(s[0]), strideY(s[1]), strideX(s[2])
            , padC(pb[0]), padY(pb[1]), padX(pb[2]), format(f), ceil(c), excludePad(ep)
        {
            SetDst(pe[0], pe[1], pe[2]);
        }

        ParamPooling(size_t b, size_t sC, size_t sH, size_t sW, ::SimdTensorFormatType f)
            : batch(b), srcC(sC), srcH(sH), srcW(sW), kernelC(1), kernelY(sH), kernelX(sW), strideC(1), strideY(1), strideX(1)
            , padC(0), padY(0), padX(0), format(f), ceil(SimdFalse), excludePad(SimdFalse)
        {
            SetDst(0, 0, 0);
        }

    protected:
        SIMD_INLINE void SetDst(size_t padD, size_t padH, size_t padW)
        {
            if (ceil)
            {
                dstC = (size_t)(::ceil((float)(srcC + padC + padD - kernelC) / strideC)) + 1;
                dstH = (size_t)(::ceil((float)(srcH + padY + padH - kernelY) / strideY)) + 1;
                dstW = (size_t)(::ceil((float)(srcW + padX + padW - kernelX) / strideX)) + 1;
            }
            else
            {
                dstC = (size_t)(::floor((float)(srcC + padC + padD - kernelC) / strideC)) + 1;
                dstH = (size_t)(::floor((float)(srcH + padY + padH - kernelY) / strideY)) + 1;
                dstW = (size_t)(::floor((float)(srcW + padX + padW - kernelX) / strideX)) + 1;
            }
        }
    };
}

#endif
