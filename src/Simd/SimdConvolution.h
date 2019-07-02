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
#include "Simd/SimdGemm.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct ConvParam : public SimdConvolutionParameters
    {
        SimdBool trans;
        size_t batch;
        SimdGemm32fNNPtr gemm;

        ConvParam(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            *((SimdConvolutionParameters*)this) = *conv;
            this->trans = trans;
            this->batch = batch;
            this->gemm = gemm;
        }

        bool Valid()
        {
            return 
                dstH == (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1 && dstH > 0 &&
                dstW == (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1 && dstW > 0;
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
        SIMD_INLINE bool Is1x1() const
        {
            return IsKernel(1) && IsDilation(1) && IsStride(1) && IsPad(0);
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << strideX << "-" << Simd::Max(padX, padW) << "-" << group << "-" << trans;
            return ss.str();
        }

        long long Flop() const
        {
            return batch * kernelY * kernelX * srcC * dstH * dstW * dstC / group * 2;
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
            , _nhwcRun(0)
            , _nhwcReorderB(0)
            , _biasAndActivation(0)
        {
        }

        const ConvParam & Param() const 
        {
            return _param;
        }

        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const
        {
            return 1;
        }

        virtual size_t InternalBufferSize() const
        {
            return _buffer.size + _nhwcWeight.size;
        }

        virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            _weight = weight;
            if (internal)
                *internal = SimdFalse;
            _bias = bias;
            _params = params;
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
        typedef void(*NhwcReorderB)(size_t M, size_t N, size_t K, const float * B, float * pB);
        typedef void(*NhwcRun)(size_t M, size_t N, size_t K, const float * A, const float * B, float * C);
        typedef void(*BiasAndActivation)(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        ConvParam _param;
        Array32f _buffer;
        float _0, _1;
        const float * _weight, * _bias, * _params;
        RuntimeGemm _gemm;
        Array32f _nhwcWeight;
        NhwcRun _nhwcRun;
        NhwcReorderB _nhwcReorderB;
        BiasAndActivation _biasAndActivation;
    };

    namespace Base
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Convolution
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
            virtual String Desc() const { return "Base::GemmNN"; }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            virtual void ImgToCol(const float * src, float * dst);
            virtual void ImgToRow(const float * src, float * dst);

            bool _is1x1, _merge;
            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD;
        };

        class ConvolutionGemmNT : public Convolution
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual String Desc() const { return "Base::GemmNT"; }
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            virtual void GemmAndBias(const float * src, float * dst);

            static void ImgToRow(const float * src, const ConvParam & p, float * dst);

            bool _is1x1;
            size_t _weightStep, _srcStep, _dstStep, _M, _N, _K, _batch, _sizeS, _sizeB, _sizeD;
        };

        class ConvolutionWinograd : public Convolution
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Base::Winograd" + ToStr(_block) + "x3"; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            typedef void(*SetFilter)(const float * src, size_t size, float * dst, SimdBool trans);
            typedef void(*SetInput)(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, SimdBool pad, SimdBool trans);
            typedef void(*SetOutput)(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

            void SetBlock(size_t block);

            bool _merge;
            size_t _count, _block, _tileH, _tileW, _strideW, _strideS, _strideD, _M, _N, _K, _batch, _sizeS, _sizeD, _nhwcStrideW;
            SimdBool _pad;
            Array32f _winogradWeight;
            SetFilter _setFilter;
            SetInput _setInput;
            SetOutput _setOutput;
        };

        class ConvolutionDirectNchw : public Convolution
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Base::DirectNchw"; }
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionBiasActivationPtr)(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW);
        protected:
            void Pad(const float * src, float * dst) const;
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();

            size_t _grW, _grS, _grD, _srcC, _srcH, _srcW, _dstC;
            int _pad;
            ConvolutionBiasActivationPtr _convolutionBiasActivation;
        };

        class ConvolutionDirectNhwc : public Convolution
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Base::DirectNhwc"; }
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            typedef void(*ConvolutionBiasActivationPtr)(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation(); 

            size_t _batch, _sizeS, _sizeD;
            ConvolutionBiasActivationPtr _convolutionBiasActivation;
        };

        class ConvolutionDepthwiseDotProduct : public Convolution
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual String Desc() const { return "Base::DepthwiseDotProduct"; }
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

        protected:
            size_t _count, _size, _batch, _sizeS, _sizeD;
        }; 

        class ConvolutionNhwcDirect : public Convolution
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Base::NhwcDirect"; }
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const ConvParam & p);

            struct AlgParam
            {
                size_t microD, macroH, macroC, macroD;
            };
            typedef void(*ConvolutionPtr)(const float * src, const ConvParam & p, const AlgParam & a, const float * weight, const float * bias, const float * params, float * dst);

        protected:
            void SetAlgParam(size_t microD, size_t L1, size_t L2, size_t L3);
            void ReorderWeight(const float * src, float * dst);

            size_t _sizeS, _sizeD;
            AlgParam _alg;
            Array32f _rWeight, _rBias, _rParams;
            ConvolutionPtr _convolution;
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Base::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
            virtual String Desc() const { return "Sse::GemmNN"; }
        };

        class ConvolutionWinograd : public Base::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Sse::Winograd" + ToStr(_block) + "x3"; }
        };

        class ConvolutionDirectNchw : public Base::ConvolutionDirectNchw
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Sse::DirectNchw"; }

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectNhwc : public Base::ConvolutionDirectNhwc
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Sse::DirectNhwc"; }

            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Base::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual String Desc() const { return "Sse::DepthwiseDotProduct"; }
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionNhwcDirect : public Base::ConvolutionNhwcDirect
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Sse::NhwcDirect"; }

            static bool Preferable(const ConvParam & p);
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_SSE3_ENABLE    
    namespace Sse3
    {
        class ConvolutionGemmNT : public Base::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual String Desc() const { return "Sse3::GemmNT"; }

            static bool Preferable(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
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
            virtual String Desc() const { return "Avx::GemmNN"; }
        protected:
            virtual void ImgToRow(const float * src, float * dst);
        };

        class ConvolutionGemmNT : public Sse3::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual String Desc() const { return "Avx::GemmNT"; }
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Sse::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Avx::Winograd" + ToStr(_block) + "x3"; }
        };

        class ConvolutionDirectNchw : public Sse::ConvolutionDirectNchw
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Avx::DirectNchw"; }
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectNhwc : public Sse::ConvolutionDirectNhwc
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Avx::DirectNhwc"; }
        
            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Sse::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual String Desc() const { return "Avx::DepthwiseDotProduct"; }
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionNhwcDirect : public Sse::ConvolutionNhwcDirect
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Avx::NhwcDirect"; }
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void NhwcRun(size_t M, size_t N, size_t K, const float * A, const float * B, float * C);
        void NhwcReorderB(size_t M, size_t N, size_t K, const float * B, float * pB);
        size_t NhwcBufferSize(size_t M, size_t N, size_t K);

        class ConvolutionGemmNN : public Avx::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
            virtual String Desc() const { return "Avx2::GemmNN"; }
        protected:
            virtual void ImgToCol(const float * src, float * dst);
        private:
            Array32i _index, _nose, _tail, _start;
        };

        class ConvolutionGemmNT : public Avx::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual String Desc() const { return "Avx2::GemmNT"; }
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Avx::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Avx2::Winograd" + ToStr(_block) + "x3"; }
        };

        class ConvolutionDirectNchw : public Avx::ConvolutionDirectNchw
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Avx2::DirectNchw"; }
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectNhwc : public Avx::ConvolutionDirectNhwc
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Avx2::DirectNhwc"; }
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionNhwcDirect : public Avx::ConvolutionNhwcDirect
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Avx2::NhwcDirect"; }
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst);

        class ConvolutionGemmNN : public Avx2::ConvolutionGemmNN
        {
        public:
            ConvolutionGemmNN(const ConvParam & p);
            virtual String Desc() const { return "Avx512f::GemmNN"; }
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
            virtual String Desc() const { return "Avx512f::GemmNT"; }
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Avx2::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Avx512f::Winograd" + ToStr(_block) + "x3"; }
        };

        class ConvolutionDirectNchw : public Avx2::ConvolutionDirectNchw
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Avx512f::DirectNchw"; }

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectNhwc : public Avx2::ConvolutionDirectNhwc
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Avx512f::DirectNhwc"; }
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionNhwcDirect : public Avx2::ConvolutionNhwcDirect
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Avx512f::NhwcDirect"; }
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
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
            virtual String Desc() const { return "Neon::GemmNN"; }
        };

        class ConvolutionGemmNT : public Base::ConvolutionGemmNT
        {
        public:
            ConvolutionGemmNT(const ConvParam & p);
            virtual String Desc() const { return "Neon::GemmNT"; }

            static bool Preferable(const ConvParam & p);
        protected:
            virtual void GemmAndBias(const float * src, float * dst);
        };

        class ConvolutionWinograd : public Base::ConvolutionWinograd
        {
        public:
            ConvolutionWinograd(const ConvParam & p);
            virtual String Desc() const { return "Neon::Winograd" + ToStr(_block) + "x3"; }

            static bool Preferable(const ConvParam & p);
        };

        class ConvolutionDirectNchw : public Base::ConvolutionDirectNchw
        {
        public:
            ConvolutionDirectNchw(const ConvParam & p);
            virtual String Desc() const { return "Neon::DirectNchw"; }

            static bool Preferable(const ConvParam & p);

        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDirectNhwc : public Base::ConvolutionDirectNhwc
        {
        public:
            ConvolutionDirectNhwc(const ConvParam & p);
            virtual String Desc() const { return "Neon::DirectNhwc"; }

            static bool Preferable(const ConvParam & p);
        protected:
            virtual ConvolutionBiasActivationPtr SetConvolutionBiasActivation();
        };

        class ConvolutionDepthwiseDotProduct : public Base::ConvolutionDepthwiseDotProduct
        {
        public:
            ConvolutionDepthwiseDotProduct(const ConvParam & p);
            virtual String Desc() const { return "Neon::DepthwiseDotProduct"; }
            virtual void Forward(const float * src, float * buf, float * dst);
        };

        class ConvolutionNhwcDirect : public Base::ConvolutionNhwcDirect
        {
        public:
            ConvolutionNhwcDirect(const ConvParam & p);
            virtual String Desc() const { return "Neon::NhwcDirect"; }

            static bool Preferable(const ConvParam & p);
        };

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimConvolution_h__
