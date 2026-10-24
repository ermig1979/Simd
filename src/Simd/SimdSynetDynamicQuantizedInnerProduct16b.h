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
#ifndef __SimdSynetDynamicQuantizedInnerProduct16b_h__
#define __SimdSynetDynamicQuantizedInnerProduct16b_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdSynetConvParam.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct DynamicQuantizedInnerProduct16bParam
    {
        size_t M, N, K;
        SimdTensorDataType typeA, typeB, typeC;
        SimdBool constB, bias, overflow;
        SimdConvolutionActivationType activation;

        DynamicQuantizedInnerProduct16bParam(size_t m, size_t n, size_t k, SimdTensorDataType ta, SimdTensorDataType tb, 
            SimdTensorDataType tc, SimdBool c, SimdBool b, SimdConvolutionActivationType a, SimdBool o)
            : M(m), N(n), K(k), typeA(ta), typeB(tb), typeC(tc)
            , constB(c), bias(b), activation(a), overflow(o)
        {
        }

        bool Valid()
        {
            if (typeA != SimdTensorData32f && typeA != SimdTensorData16b)
                return false;
            if (typeC != SimdTensorData32f && typeC != SimdTensorData16b)
                return false;
            if (constB)
                return typeB == SimdTensorData32f || typeB == SimdTensorData8i;
            else
                return typeB == SimdTensorData32f || typeB == SimdTensorData16b;
        }

        String Info(bool detail = true) const
        {
            std::stringstream ss;
            ss << M << "x" << K << "-" << N << " ";
            ss << ToChar(typeA) << ToChar(typeB) << ToChar(typeC) << " ";
            ss << (constB ? "1" : "2") << (bias ? "b" : "o");
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
        class SynetDynamicQuantizedInnerProduct16b : public Deletable
        {
        public:
            SynetDynamicQuantizedInnerProduct16b(const DynamicQuantizedInnerProduct16bParam& p);

            const DynamicQuantizedInnerProduct16bParam& Param() const { return _param; }

            virtual String Ext() const = 0;
            virtual String Desc() const = 0;

            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;

            virtual void SetParams(const uint8_t* weight, const float* scale, const float* bias, const float* params);

            virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C) = 0;

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

        protected:
            virtual void SetWeight(const int8_t* weight) = 0;
            virtual void QuantizeB(const float* weight) = 0;

            DynamicQuantizedInnerProduct16bParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer* _perf;
#endif
            mutable String _info;
            Array8u _buffer;
            Array8i _weight;
            Array32i _sums;
            Array32f _scale, _bias, _params;
            size_t _sizeA, _sizeB, _sizeC, _sizeT, _aN;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetDynamicQuantizedInnerProduct16bRef : public SynetDynamicQuantizedInnerProduct16b
        {
        public:
            SynetDynamicQuantizedInnerProduct16bRef(const DynamicQuantizedInnerProduct16bParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C);

        protected:
            virtual void SetWeight(const int8_t* weight);
            virtual void QuantizeB(const float* weight);
        };

        //-------------------------------------------------------------------------------------------------

        class SynetDynamicQuantizedInnerProduct16bGemmV0 : public SynetDynamicQuantizedInnerProduct16b
        {
        public:
            SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C);

            static bool Preferable(const DynamicQuantizedInnerProduct16bParam& p);

            struct AlgParam
            {
                size_t F, microM, microN, microK;
                size_t macroM, macroN, macroK;
                size_t aM, aN, aK, cN, eA, eB, eC, reorder;
            };

            typedef void(*ConvertAPtr)(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
                const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t *zero);
            typedef void(*ConvertBPtr)(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
                const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums);
            typedef void(*GemmBodyPtr)(const uint8_t* A, const DynamicQuantizedInnerProduct16bParam& p, const AlgParam& a,
                size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum);
            typedef void(*GemmLastPtr)(const uint8_t* A, const DynamicQuantizedInnerProduct16bParam& p, const AlgParam& a,
                size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf, const int32_t* aZero, 
                const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C);

        protected:
            virtual void SetWeight(const int8_t* weight);
            virtual void QuantizeB(const float* weight);
            void SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3);

            AlgParam _alg;
            ConvertAPtr _convertA; 
            ConvertBPtr _convertB;
            GemmBodyPtr _gemmBody;
            GemmLastPtr _gemmLast;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, 
            SimdTensorDataType typeC, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetDynamicQuantizedInnerProduct16bGemmV0 : public Base::SynetDynamicQuantizedInnerProduct16bGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB,
            SimdTensorDataType typeC, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetDynamicQuantizedInnerProduct16bGemmV0 : public Sse41::SynetDynamicQuantizedInnerProduct16bGemmV0
        {
        public:
            SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB,
            SimdTensorDataType typeC, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {

    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {

    }
#endif

#if defined(SIMD_AMXBF16_ENABLE)  
    namespace AmxBf16
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {

    }
#endif

#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {

    }
#endif
}

#endif
