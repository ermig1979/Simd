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
        Convolution(const ConvParam & p) : _param(p) {}
        virtual size_t BufferSize() const = 0;
        virtual void SetWeight(const float * weight, const float * bias) = 0;
        virtual void Forward(const float * src, float * buf, float * dst) = 0;

    protected:
        ConvParam _param;
    };

    namespace Base
    {
        class ConvolutionGemm : public Convolution
        {
        public:
            ConvolutionGemm(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetWeight(const float * weight, const float * bias);
            virtual void Forward(const float * src, float * buf, float * dst);
            static void ImgToCol(const float * src, const ConvParam & p, float * dst);

        protected:
            bool _is1x1;
            float _0, _1;
            const float * _weight, * _bias;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        class ConvolutionGemm : public Base::ConvolutionGemm
        {
        public:
            ConvolutionGemm(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class ConvolutionGemm : public Sse::ConvolutionGemm
        {
        public:
            ConvolutionGemm(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
}
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ConvolutionGemm : public Avx::ConvolutionGemm
        {
        public:
            ConvolutionGemm(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
}
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ConvolutionGemm : public Avx2::ConvolutionGemm
        {
        public:
            ConvolutionGemm(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }
#endif//SIMD_AVX512F_ENABLE
}

#ifdef _N_OLD
#define _N _N_OLD
#undef _N_OLD
#endif

#endif//__SimConvolution_h__
