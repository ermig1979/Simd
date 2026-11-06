/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdSynetDynamicQuantizedInnerProduct32f_h__
#define __SimdSynetDynamicQuantizedInnerProduct32f_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdSynetConvParam.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct DynamicQuantizedInnerProductParam
    {
        size_t M, N, K;
        SimdBool bias;
        SimdConvolutionActivationType activation;

        DynamicQuantizedInnerProductParam(size_t m, size_t n, size_t k,
            SimdBool b, SimdConvolutionActivationType a)
            : M(m), N(n), K(k)
            , bias(b), activation(a)
        {
        }

        bool Valid()
        {
            return true;
        }

        String Info(bool detail = true) const
        {
            std::stringstream ss;
            ss << M << "x" << K << "-" << N << " ";
            ss << (bias ? "b" : "o");
            if (detail)
                ss << "-" << ToStr(activation);
            return ss.str();
        }

        int64_t Flop() const
        {
            return int64_t(M) * N * K * 2;
        }
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetDynamicQuantizedInnerProduct32f : public Deletable
        {
        public:
            SynetDynamicQuantizedInnerProduct32f(const DynamicQuantizedInnerProductParam& p);

            const DynamicQuantizedInnerProductParam& Param() const { return _param; }

            virtual String Ext() const = 0;
            virtual String Desc() const = 0;

            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;

            virtual void SetParams(const int8_t* weight, const float* scale, const float* bias, const float* params);

            virtual void Forward(const float* A, uint8_t* buf, float* C) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer* Perf(const char* func);
#endif

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

            const char* Info() const
            {
                _info = Desc();
                return _info.c_str();
            }

            typedef void (*MinMax32fPtr)(const float* src, size_t size, float* min, float* max);

        protected:
            virtual void SetWeight(const int8_t* weight) = 0;
            void AnalyseInput(const float* src, size_t size, int32_t *corr, float *norm);

            DynamicQuantizedInnerProductParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer* _perf;
#endif
            mutable String _info;
            Array8u _buffer;
            Array8i _weight;
            Array32i _sums;
            Array32f _scale, _bias, _params;
            size_t _sizeA, _sizeS, _sizeB, _aN;
            MinMax32fPtr _minMax32f;
            float _aScale;
            uint8_t _aZero;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetDynamicQuantizedInnerProduct32fRef : public SynetDynamicQuantizedInnerProduct32f
        {
        public:
            SynetDynamicQuantizedInnerProduct32fRef(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float* A, uint8_t* buf, float* C);

        protected:
            virtual void SetWeight(const int8_t* weight);
        };

        //-------------------------------------------------------------------------------------------------

        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public SynetDynamicQuantizedInnerProduct32f
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float* A, uint8_t* buf, float* C);

            static bool Preferable(const DynamicQuantizedInnerProductParam& p);

            struct AlgParam
            {
                size_t F, microM, microN, microK;
                size_t macroM, macroN, macroK;
                size_t aM, aN, aK, sN, reorder;
            };

            typedef void(*ConvertPtr)(const float* src, float norm, uint8_t zero, const DynamicQuantizedInnerProductParam& p,
                const AlgParam& a, size_t N, uint8_t* dst);
            typedef void(*GemmBodyPtr)(const uint8_t* A, const DynamicQuantizedInnerProductParam& p, const AlgParam& a,
                size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum);
            typedef void(*GemmLastPtr)(const uint8_t* A, const DynamicQuantizedInnerProductParam& p, const AlgParam& a, 
                size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf, 
                const int32_t* corr, const float* norm, const float* bias, const float* params, float* C);

        protected:
            virtual void SetWeight(const int8_t* weight);
            void SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3);

            AlgParam _alg;
            ConvertPtr _convert;
            GemmBodyPtr _gemmBody;
            GemmLastPtr _gemmLast;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Base::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Sse41::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Avx2::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Avx512bw::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Avx512vnni"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE)  
    namespace AmxBf16
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Avx512vnni::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "AmxBf16"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Base::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Neon"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif

#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        class SynetDynamicQuantizedInnerProduct32fGemmV0 : public Base::SynetDynamicQuantizedInnerProduct32fGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Sve2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
    }
#endif
}

#endif
