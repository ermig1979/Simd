/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdSynetInnerProduct16b_h__
#define __SimdSynetInnerProduct16b_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdSynetConvParam.h"

namespace Simd
{
    struct InnerProductParam16b
    {
        size_t M, N, K;
        SimdTensorDataType typeA, typeB, typeC;
        SimdBool transB, constB, bias;

        InnerProductParam16b(size_t m, size_t n, size_t k, 
            SimdTensorDataType ta, SimdTensorDataType tb, SimdTensorDataType tc,
            SimdBool t, SimdBool c, SimdBool b)
            : M(m), N(n), K(k)
            , typeA(ta), typeB(tb), typeC(tc)
            , transB(t), constB(c), bias(b)
        {
        }

        bool Valid()
        {
            return 
                (typeA == SimdTensorData32f || typeA == SimdTensorData16b) &&
                (typeB == SimdTensorData32f || typeB == SimdTensorData16b) &&
                (typeC == SimdTensorData32f || typeC == SimdTensorData16b);
        }

        String Info() const
        {
            std::stringstream ss;
            ss << M << "x" << N << "x" << K << "-";
            ss << ToChar(typeA) << ToChar(typeB) << ToChar(typeC) << "-";
            ss << (transB ? "t" : "n") << (constB ? "1" : "2") << (bias ? "b" : "o");
            return ss.str();
        }

        int64_t Flop() const
        {
            return int64_t(M) * N * K * 2;
        }
    };

    //-------------------------------------------------------------------------------------------------

    class SynetInnerProduct16b : public Deletable
    {
    public:
        SynetInnerProduct16b(const InnerProductParam16b& p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
        }

        const InnerProductParam16b& Param() const
        {
            return _param;
        }

        virtual size_t InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() + _bias.RawSize();
        }

        virtual size_t ExternalBufferSize() const
        {
            return 0;
        }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual void SetParams(const float* weight, const float* bias) = 0;
        virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:
        InnerProductParam16b _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        Array8u _buffer;
        Array16u _weight;
        Array32f _bias;
        mutable String _info;

        uint8_t* Buffer(uint8_t* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetInnerProduct16bRef : public SynetInnerProduct16b
        {
        public:
            SynetInnerProduct16bRef(const InnerProductParam16b& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias);
            virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C);

        protected:
            void GemmAndBias(const uint16_t* A, const uint16_t* B, float* C);

            size_t _sizeA, _sizeB, _sizeC;
        };

        //class SynetInnerProduct16bGemm : public SynetInnerProduct16b
        //{
        //public:
        //    SynetInnerProduct16bGemm(const InnerProductParam16b& p);
        //    virtual String Ext() const { return "Base"; }
        //    virtual String Desc() const;
        //    virtual void SetParams(const float* weight, SimdBool* internal, const float* bias);
        //    virtual void Forward(const uint8_t* A, const uint8_t* B, const uint8_t* bias, uint8_t* C);

        //protected:
        //    //typedef void(*GemmPtr)(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);
        //    typedef void(*BiasAndActivationPtr)( const float* bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float* params, SimdBool trans, float* dst);
        //    //typedef void(*ProdPtr)(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst);
        //    //typedef void(*CbPackPtr)(size_t M, size_t N, size_t K, const float* B, float* pB, GemmKernelType type, bool compatibility);
        //    ///typedef void(*CbRunPtr)(size_t M, size_t N, size_t K, const float* A, const float* B, float* C, GemmKernelType type, bool compatibility);

        //    float _0, _1;
        //    GemmPtr _gemm;
        //    BiasAndActivationPtr _biasAndActivation;
        //    ProdPtr _prod;
        //    size_t _M, _N, _K, _ldW, _ldS, _ldD;
        //    Array32f _cbWeight;
        //    CbPackPtr _cbPack;
        //    CbRunPtr _cbRun;
        //};

        //class SynetInnerProduct16bProd : public SynetInnerProduct16b
        //{
        //public:
        //    SynetInnerProduct16bProd(const InnerProductParam16b& p);
        //    virtual String Ext() const { return "Base"; }
        //    virtual String Desc() const { return Ext() + "::Prod"; }
        //    virtual size_t InternalBufferSize() const { return _rWeight.size + _rBias.size; }
        //    virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
        //    virtual void Forward(const float* src, float* dst);

        //    static bool Preferable(const InnerProductParam16b& p);

        //protected:
        //    typedef void(*ProdPtr)(const float* src, const float* weight, const float* bias, size_t input, size_t output, float* dst);

        //    ProdPtr _prod;
        //    Array32f _rWeight, _rBias;
        //    size_t _F, _N, _K;

        //    void SetSize(size_t F);
        //    void ReorderWeight(const float* src, float* dst);
        //};

        //-------------------------------------------------------------------------------------------------

        void * SynetInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif

}

#endif
