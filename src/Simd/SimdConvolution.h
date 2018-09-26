/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifndef __SimdConvolution_h__
#define __SimdConvolution_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdBase.h"

#ifdef _N
#define _N_OLD _N
#undef _N
#endif

namespace Simd
{
    struct ConvParam
    {
        size_t srcC, srcH, srcW, dstC, dstH, dstW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group;

        ConvParam(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            this->srcC = srcC;
            this->srcH = srcH;
            this->srcW = srcW;
            this->dstC = dstC;
            this->dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            this->dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            this->kernelY = kernelY;
            this->kernelX = kernelX;
            this->dilationY = dilationY;
            this->dilationX = dilationX;
            this->strideY = strideY;
            this->strideX = strideX;
            this->padY = padY;
            this->padX = padX;
            this->padH = padH;
            this->padW = padW;
            this->group = group;
        }

        bool Is1x1() const
        {
            return kernelY == 1 && kernelX == 1 && dilationY == 1 && dilationX == 1 && strideY == 1 && strideX == 1 && padY == 0 && padX == 0 && padH == 0 && padW == 0;
        }
    };

    class Convolution : Deletable
    {
    public:
        Convolution(const ConvParam & p)
            : _param(p)
        {
        }

        const ConvParam & Param() const
        {
            return _param;
        }

        virtual size_t BufferSize() const = 0;

        virtual void SetWeight(const float * weight, const float * bias) = 0;

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

    private:
        ConvParam _param;
    };

    namespace Base
    {
        class ConvolutionGemm : public Convolution
        {
        public:
            ConvolutionGemm(const ConvParam & p)
                : Convolution(p)
                , _is1x1(p.Is1x1())
                , _0(0.0f)
                , _1(1.0f)
            {
                _M = p.dstC / p.group;
                _N = p.dstH  * p.dstW;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _weightStep = p.dstC * _K / p.group;
                _srcStep = _K * _N;
                _dstStep = p.dstC * _N / p.group;
            }

            virtual size_t BufferSize() const
            {
                const ConvParam & p = Param();
                return p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
            };

            virtual void SetWeight(const float * weight, const float * bias)
            {
                _weight = weight;
                _bias = bias;
            }

            virtual void Forward(const float * src, float * buf, float * dst)
            {
                const ConvParam & p = Param();
                if (!_is1x1)
                {
                    ImgToCol(src, p, buf);
                    src = buf;
                }
                for (size_t g = 0; g < p.group; ++g)
                    Base::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);

                if (_bias)
                    Base::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
            }

            static void ImgToCol(const float * src, const ConvParam & p, float * dst);

        protected:
            bool _is1x1;
            float _0, _1;
            const float * _weight;
            const float * _bias;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);

    }
}

#ifdef _N_OLD
#define _N _N_OLD
#undef _N_OLD
#endif

#endif//__SimConvolution_h__
