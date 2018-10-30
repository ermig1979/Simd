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

        SIMD_INLINE bool IsKernel(size_t value) const
        {
            return kernelY == value && kernelX == value;
        }

        SIMD_INLINE bool IsDilation(size_t value) const
        {
            return dilationY == value && dilationX == value;
        }

        SIMD_INLINE bool IsStride(size_t value) const
        {
            return strideY == value && strideX == value;
        }

        SIMD_INLINE bool IsPad(size_t value) const
        {
            return padY == value && padX == value && padH == value && padW == value;
        }
    };

    class Convolution : public Deletable
    {
    public:
        Convolution(const ConvParam & p) 
            : _param(p)
            , _0(0.0f)
            , _1(1.0f)
            , _activationType(::SimdConvolutionActivationIdentity) 
        {
        }

        virtual size_t BufferSize() const = 0;
        virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal) = 0;
        virtual void Forward(const float * src, float * buf, float * dst) = 0;

        void SetActivation(::SimdConvolutionActivationType type, const float * params)
        {
            _activationType = type;
            if (_activationType == ::SimdConvolutionActivationLeakyRelu)
                _activationParams[0] = params[0];
            if (_activationType == ::SimdConvolutionActivationRestrictRange)
            {
                _activationParams[0] = params[0];
                _activationParams[1] = params[1];
            }
        }

        float * Buffer(float * buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(BufferSize());
                return _buffer.data;
            }
        }

    protected:
        ConvParam _param;
        Array32f _buffer;
        float _0, _1;
        ::SimdConvolutionActivationType _activationType;
        float _activationParams[2];
    };

    namespace Base
    {
        class ConvolutionImgToCol : public Convolution
        {
        public:
            ConvolutionImgToCol(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            virtual void GemmAndBias(const float * src, float * dst);

            static void ImgToCol(const float * src, const ConvParam & p, float * dst);

            bool _is1x1;
            const float * _weight, * _bias;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K;
        };

        class ConvolutionImgToRow : public Convolution
        {
        public:
            ConvolutionImgToRow(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void GemmAndBias(const float * src, float * dst);

            static void ImgToRow(const float * src, const ConvParam & p, float * dst);

            const float * _weight, *_bias;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K;
        };

        class ConvolutionWinograd2x3p : public Convolution
        {
        public:
            ConvolutionWinograd2x3p(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            size_t _count, _block, _tileH, _tileW, _strideW, _strideS, _strideD, _M, _N, _K;
            int _pad;
            Array32f _weight;
            const float * _bias;
        };

        class ConvolutionDirect : public Convolution
        {
        public:
            ConvolutionDirect(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            void Pad(const float * src, float * dst) const;
            virtual void ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const;

            size_t _weightStep, _srcStep, _dstStep, _srcC, _srcH, _srcW, _dstC;
            int _pad;
            const float * _weight, * _bias;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, float * dst);

        class ConvolutionImgToCol : public Base::ConvolutionImgToCol
        {
        public:
            ConvolutionImgToCol(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd2x3p : public Base::ConvolutionWinograd2x3p
        {
        public:
            ConvolutionWinograd2x3p(const ConvParam & p);
            virtual void SetWeight(const float * weight, const float * bias, SimdBool * internal);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionDirect : public Base::ConvolutionDirect
        {
        public:
            ConvolutionDirect(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_SSE3_ENABLE    
    namespace Sse3
    {
        class ConvolutionImgToRow : public Base::ConvolutionImgToRow
        {
        public:
            ConvolutionImgToRow(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }
#endif//SIMD_SSE3_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, float * dst);

        class ConvolutionImgToCol : public Sse::ConvolutionImgToCol
        {
        public:
            ConvolutionImgToCol(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionImgToRow : public Sse3::ConvolutionImgToRow
        {
        public:
            ConvolutionImgToRow(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd2x3p : public Sse::ConvolutionWinograd2x3p
        {
        public:
            ConvolutionWinograd2x3p(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionDirect : public Sse::ConvolutionDirect
        {
        public:
            ConvolutionDirect(const ConvParam & p);

        protected:
            virtual void ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
}
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ConvolutionImgToCol : public Avx::ConvolutionImgToCol
        {
        public:
            ConvolutionImgToCol(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionImgToRow : public Avx::ConvolutionImgToRow
        {
        public:
            ConvolutionImgToRow(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd2x3p : public Avx::ConvolutionWinograd2x3p
        {
        public:
            ConvolutionWinograd2x3p(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionDirect : public Avx::ConvolutionDirect
        {
        public:
            ConvolutionDirect(const ConvParam & p);

        protected:
            virtual void ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
}
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ConvolutionImgToCol : public Avx2::ConvolutionImgToCol
        {
        public:
            ConvolutionImgToCol(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionImgToRow : public Avx2::ConvolutionImgToRow
        {
        public:
            ConvolutionImgToRow(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd2x3p : public Avx2::ConvolutionWinograd2x3p
        {
        public:
            ConvolutionWinograd2x3p(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionDirect : public Avx2::ConvolutionDirect
        {
        public:
            ConvolutionDirect(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const;
        };

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group);
    }
#endif//SIMD_AVX512F_ENABLE
}

#endif//__SimConvolution_h__
