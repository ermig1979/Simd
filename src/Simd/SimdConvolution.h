/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdConvolution_h__
#define __SimdConvolution_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct ConvParam
    {
        size_t batch, srcC, srcH, srcW, dstC, dstH, dstW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group;
        SimdBool srcT, dstT;
        SimdConvolutionActivationType activation;
        SimdGemm32fNNPtr gemm;

        ConvParam(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, 
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm)
        {
            this->batch = batch;
            this->srcC = srcC;
            this->srcH = srcH;
            this->srcW = srcW;
            this->srcT = srcT;
            this->dstC = dstC;
            this->dstT = dstT;
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
            this->activation = activation;
            this->gemm = gemm;
        }

        bool Valid()
        {
            return srcT == dstT;
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

        SIMD_INLINE bool IsDepthwise() const
        {
            return srcC == group && dstC == group;
        }

        SIMD_INLINE bool IsChw() const
        {
            return srcT == 0 && dstT == 0;
        }

        SIMD_INLINE bool IsHwc() const
        {
            return srcT == 1 && dstT == 1;
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << strideX << "-" << Simd::Max(padX, padW) << "-" << group << "-" << srcT;
            return ss.str();
        }
#endif
    };

    class Convolution : public Deletable
    {
    public:
        Convolution(const ConvParam & p) 
            : _param(p)
            , _0(0.0f)
            , _1(1.0f)
            , _hwcRun(0)
            , _hwcReorderB(0)
            , _biasAndActivation(0)
        {
        }

        virtual size_t BufferSize() const = 0;
        virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params) = 0;
        virtual void Forward(const float * src, float * buf, float * dst) = 0;

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
        typedef void(*HwcReorderB)(size_t M, size_t N, size_t K, const float * B, float * pB);
        typedef void(*HwcRun)(size_t M, size_t N, size_t K, const float * A, const float * B, float * C);
        typedef void(*BiasAndActivation)(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        ConvParam _param;
        Array32f _buffer;
        float _0, _1;
        const float * _params;
        RuntimeGemm _gemm;
        Array32f _hwcWeight;
        HwcRun _hwcRun;
        HwcReorderB _hwcReorderB;
        BiasAndActivation _biasAndActivation;
    };

    namespace Base
    {
        class ConvolutionGemmNN : public Convolution
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            virtual void ImgToCol(const float * src, float * dst);
            virtual void ImgToRow(const float * src, float * dst);

            bool _is1x1, _merge;
            const float * _weight, * _bias;
            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD;
        };

        class ConvolutionGemmNT : public Convolution
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void GemmAndBias(const float * src, float * dst);

            static void ImgToRow(const float * src, const ConvParam & p, float * dst);

            bool _is1x1;
            const float * _weight, *_bias;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K, _batch, _sizeS, _sizeB, _sizeD;
        };

        class ConvolutionWinograd : public Convolution
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            typedef void(*SetFilter)(const float * src, size_t size, float * dst, SimdBool trans);
            typedef void(*SetInput)(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, SimdBool pad, SimdBool trans);
            typedef void(*SetOutput)(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

            void SetBlock(size_t block);

            bool _merge;
            size_t _count, _block, _tileH, _tileW, _strideW, _strideS, _strideD, _M, _N, _K, _batch, _sizeS, _sizeD, _hwcStrideW;
            SimdBool _pad;
            Array32f _weight;
            const float * _bias;
            SetFilter _setFilter;
            SetInput _setInput;
            SetOutput _setOutput;
        };

        class ConvolutionDirectChw : public Convolution
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionBiasActivationPtr)(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW);

        protected:
            void Pad(const float * src, float * dst) const;
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();

            size_t _grW, _grS, _grD, _srcC, _srcH, _srcW, _dstC;
            int _pad;
            const float * _weight, * _bias;
            ConvolutionBiasActivationPtr _convolutionBiasActivation;
        };

        class ConvolutionDirectHwc : public Convolution
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionBiasActivationPtr)(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation(); 

            const float * _weight, *_bias;
            size_t _batch, _sizeS, _sizeD;
            ConvolutionBiasActivationPtr _convolutionBiasActivation;
        };

        class ConvolutionDepthwiseDotProduct : public Convolution
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual size_t BufferSize() const;
            virtual void SetParams(const float * weight, SimdBool trans, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            size_t _count, _size, _batch, _sizeS, _sizeD;
            const float * _weight, * _bias;
        }; 

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, 
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Base::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
        };

        class ConvolutionWinograd : public Base::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
        };

        class ConvolutionDirectChw : public Base::ConvolutionDirectChw
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectHwc : public Base::ConvolutionDirectHwc
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);

            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Base::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_SSE3_ENABLE    
    namespace Sse3
    {
        class ConvolutionGemmNT : public Base::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);

            static bool Preferable(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_SSE3_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, ::SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Sse::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
        protected:
            virtual void ImgToRow(const float * src, float * dst);
        };

        class ConvolutionGemmNT : public Sse3::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Sse::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
        };

        class ConvolutionDirectChw : public Sse::ConvolutionDirectChw
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectHwc : public Sse::ConvolutionDirectHwc
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);
        
            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Sse::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ConvolutionGemmNN : public Avx::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
        protected:
            virtual void ImgToCol(const float * src, float * dst);
        private:
            Array32i _index, _nose, _tail, _start;
        };

        class ConvolutionGemmNT : public Avx::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Avx::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
        };

        class ConvolutionDirectChw : public Avx::ConvolutionDirectChw
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectHwc : public Avx::ConvolutionDirectHwc
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ConvolutionGemmNN : public Avx2::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
        protected:
            virtual void ImgToCol(const float * src, float * dst);
        private:
            Array32i _index;
            Array16u _nose, _tail;
        };

        class ConvolutionGemmNT : public Avx2::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Avx2::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
        };

        class ConvolutionDirectChw : public Avx2::ConvolutionDirectChw
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectHwc : public Avx2::ConvolutionDirectHwc
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Base::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
        };

        class ConvolutionGemmNT : public Base::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);

            static bool Preferable(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Base::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);

            static bool Preferable(const ConvParam & p);
        };

        class ConvolutionDirectChw : public Base::ConvolutionDirectChw
        {
        public:
            ConvolutionDirectChw(const ConvParam & p);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectHwc : public Base::ConvolutionDirectHwc
        {
        public:
            ConvolutionDirectHwc(const ConvParam & p);

            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Base::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        void * ConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimConvolution_h__
