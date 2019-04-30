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
#ifndef __SimdMergedConvolution_h__
#define __SimdMergedConvolution_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct MergConvParam
    {
        size_t batch, srcC, srcH, srcW, dstC, dstH, dstW, kernelY, kernelX, strideY, strideX, padY, padX, padH, padW;
        SimdConvolutionActivationType activation0, activation1;
        SimdGemm32fNNPtr gemm;

        MergConvParam(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, 
            size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, 
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm)
        {
            this->batch = batch;
            this->srcC = srcC;
            this->srcH = srcH;
            this->srcW = srcW;
            this->dstC = dstC;
            this->dstH = (srcH + padY + padH - kernelY) / strideY + 1;
            this->dstW = (srcW + padX + padW - kernelX) / strideX + 1;
            this->kernelY = kernelY;
            this->kernelX = kernelX;
            this->strideY = strideY;
            this->strideX = strideX;
            this->padY = padY;
            this->padX = padX;
            this->padH = padH;
            this->padW = padW;
            this->activation0 = activation0;
            this->activation1 = activation1;
            this->gemm = gemm;
        }

        bool Valid()
        {
            return dstH > 0 && dstW > 0;
        }

        SIMD_INLINE bool IsKernel(size_t value) const
        {
            return kernelY == value && kernelX == value;
        }

        SIMD_INLINE bool IsStride(size_t value) const
        {
            return strideY == value && strideX == value;
        }

        SIMD_INLINE bool IsPad(size_t value) const
        {
            return padY == value && padX == value && padH == value && padW == value;
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << strideX << "-" << Simd::Max(padX, padW);
            return ss.str();
        }
#endif
    };

    class MergedConvolution : public Deletable
    {
    public:
        MergedConvolution(const MergConvParam & p) 
            : _param(p)
            , _0(0.0f)
            , _1(1.0f)
        {
        }

        virtual size_t ExternalBufferSize() const
        {
            return 1;
        }

        virtual size_t InternalBufferSize() const
        {
            return 1;
        }

        virtual void SetParams(const float * weight0, const float * weight1, SimdBool * internal, 
            const float * bias0, const float * bias1, const float * params0, const float * params1)
        {
            _weight0 = weight0;
            _weight1 = weight1;
            if (internal)
                *internal = SimdFalse;
            _bias0 = bias0;
            _bias1 = bias1;
            _params0 = params0;
            _params1 = params1;
        }

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

        float * Buffer(float * buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

    protected:
        MergConvParam _param;
        Array32f _buffer;
        float _0, _1;
        const float * _weight0, * _weight1, * _bias0, * _bias1, * _params0, * _params1;
        RuntimeGemm _gemm;
    };

    namespace Base
    {
        class MergedConvolution : public Simd::MergedConvolution
        {
        public:
            MergedConvolution(const MergConvParam & p);

            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight0, const float * weight1, SimdBool * internal, const float * bias0, const float * bias1, const float * params0, const float * params1);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            typedef void(*Depthwise)(const float * src, const MergConvParam & p, size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst);
            typedef void(*NhwcReorderB)(size_t M, size_t N, size_t K, const float * B, float * pB);
            typedef void(*NhwcRun)(size_t m, size_t M, size_t N, size_t K, const float * A, const float * B, float * C);
            typedef void(*BiasAndActivation)(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

            void SetSize(size_t L2);

            bool _merge;
            size_t _batch, _block, _M, _N, _K, _sizeS, _sizeB, _sizeD;
            Depthwise _depthwise;
            Array32f _nhwcWeight;
            NhwcRun _nhwcRun;
            NhwcReorderB _nhwcReorderB;
            BiasAndActivation _biasAndActivation;
        };

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        class MergedConvolution : public Base::MergedConvolution
        {
        public:
            MergedConvolution(const MergConvParam & p);
        };

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class MergedConvolution : public Sse::MergedConvolution
        {
        public:
            MergedConvolution(const MergConvParam & p);
        };

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class MergedConvolution : public Avx::MergedConvolution
        {
        public:
            MergedConvolution(const MergConvParam & p);
        };

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class MergedConvolution : public Avx2::MergedConvolution
        {
        public:
            MergedConvolution(const MergConvParam & p);
        };

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX512F_ENABLE
}

#endif//__SimMergedConvolution_h__
