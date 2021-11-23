/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdSynetInnerProduct32f_h__
#define __SimdSynetInnerProduct32f_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
    struct InnerProductParam32f
    {
        size_t batch;
        size_t input;
        size_t output;
        SimdBool transpose;
        SimdConvolutionActivationType activation;

        InnerProductParam32f(size_t b, size_t i, size_t o, SimdBool t, SimdConvolutionActivationType a)
        {
            batch = b;
            input = i;
            output = o;
            transpose = t;
            activation = a;
        }

        bool Valid()
        {
            return 
                activation == SimdConvolutionActivationIdentity;
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << input << "x" << output << "-" << transpose;
            return ss.str();
        }

        int64_t Flop() const
        {
            return int64_t(batch) * input * output * 2;
        }
#endif
    };

    class SynetInnerProduct32f : public Deletable
    {
    public:
        SynetInnerProduct32f(const InnerProductParam32f & p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
        }

        const InnerProductParam32f & Param() const
        {
            return _param;
        }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t InternalBufferSize() const
        {
            return 0;
        }

        virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            _weight = weight;
            if (internal)
                *internal = SimdFalse;
            _bias = bias;
            _params = params;
        }

        virtual void Forward(const float * src, float * dst) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const String& func);
#endif

    protected:
        InnerProductParam32f _param;
        const float * _weight, * _bias, * _params;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
    };

    namespace Base
    {
        class SynetInnerProduct32fGemm : public SynetInnerProduct32f
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const { return _cbWeight.size; }
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float * src, float * dst);

        protected:
            typedef void(*GemmPtr)(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);
            typedef void(*BiasAndActivationPtr)(const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, SimdBool trans, float* dst);
            typedef void(*ProdPtr)(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst);
            typedef void(*CbPackPtr)(size_t M, size_t N, size_t K, const float* B, float* pB, GemmKernelType type, bool compatibility);
            typedef void(*CbRunPtr)(size_t M, size_t N, size_t K, const float* A, const float* B, float* C, GemmKernelType type, bool compatibility);

            float _0, _1;
            GemmPtr _gemm;
            BiasAndActivationPtr _biasAndActivation;
            ProdPtr _prod;
            size_t _M, _N, _K, _ldW, _ldS, _ldD;
            Array32f _cbWeight;
            CbPackPtr _cbPack;
            CbRunPtr _cbRun;
        };

        class SynetInnerProduct32fProd : public SynetInnerProduct32f
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::Prod"; }
            virtual size_t InternalBufferSize() const { return _rWeight.size + _rBias.size; }
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float* src, float* dst);

            static bool Preferable(const InnerProductParam32f& p);

        protected:
            typedef void(*ProdPtr)(const float* src, const float* weight, const float* bias, size_t input, size_t output, float* dst);

            ProdPtr _prod;
            Array32f _rWeight, _rBias;
            size_t _F, _N, _K;

            void SetSize(size_t F);
            void ReorderWeight(const float* src, float* dst);
        };

        void * SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetInnerProduct32fGemm : public Base::SynetInnerProduct32fGemm
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetInnerProduct32fProd : public Base::SynetInnerProduct32fProd
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }
#endif//SIMD_SSE41_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class SynetInnerProduct32fGemm : public Sse41::SynetInnerProduct32fGemm
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx"; }
        };

        class SynetInnerProduct32fProd : public Sse41::SynetInnerProduct32fProd
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx"; }
        };

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetInnerProduct32fGemm : public Avx::SynetInnerProduct32fGemm
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetInnerProduct32fProd : public Avx::SynetInnerProduct32fProd
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class SynetInnerProduct32fGemm : public Avx2::SynetInnerProduct32fGemm
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx512f"; }
        };

        class SynetInnerProduct32fProd : public Avx2::SynetInnerProduct32fProd
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);

            virtual String Ext() const { return "Avx512f"; }
        };

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetInnerProduct32fGemm : public Base::SynetInnerProduct32fGemm
        {
        public:
            SynetInnerProduct32fGemm(const InnerProductParam32f& p);

            virtual String Ext() const { return "Neon"; }
        };


        class SynetInnerProduct32fProd : public Base::SynetInnerProduct32fProd
        {
        public:
            SynetInnerProduct32fProd(const InnerProductParam32f& p);

            virtual String Ext() const { return "Neon"; }
        };

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimdSynetInnerProduct32f_h__
