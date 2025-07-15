/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdSynetQuantizedInnerProduct_h__
#define __SimdSynetQuantizedInnerProduct_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdSynetConvParam.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct QuantizedInnerProductParam
    {
        size_t M, N, K;
        SimdTensorDataType typeA, typeB, typeC;
        SimdBool transB, constB, bias;

        QuantizedInnerProductParam(size_t m, size_t n, size_t k,
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
                (typeA == SimdTensorData32f || typeA == SimdTensorData8u) &&
                (typeB == SimdTensorData32f || typeB == SimdTensorData8i) &&
                (typeC == SimdTensorData32f || typeC == SimdTensorData8u);
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

    //------------------------------------------------------------------------------------------------

    class SynetQuantizedInnerProduct : public Deletable
    {
    public:
        SynetQuantizedInnerProduct(const QuantizedInnerProductParam& p);

        const QuantizedInnerProductParam & Param() const { return _param; }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const;
        virtual size_t InternalBufferSize() const;

        virtual void SetParams(const float* aScale, const uint8_t* aZero, const int8_t* b, const float* bScale, const int32_t* bias, const float* cScale, const uint8_t* cZero);

        virtual void Forward(const uint8_t * A, const uint8_t* B, uint8_t * buf, uint8_t * C) = 0;

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
        virtual void SetB(const int8_t* b) = 0;
        virtual void SetBias(const int8_t* b, const int32_t* bias);
        virtual void SetOther();

        QuantizedInnerProductParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
        Array8u _buffer, _aZero;
        Array8i _b;
        Array32i _bias, _cZero;
        Array32f _bScale, _norm; 
        float _aScale, _cScale;
        bool _a8u, _c8u;
        size_t _sizeA, _sizeB, _sizeC, _elemA, _elemB, _elemC;
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetQuantizedInnerProductRef : public SynetQuantizedInnerProduct
        {
        public:
            SynetQuantizedInnerProductRef(const QuantizedInnerProductParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C);

        protected:
            virtual void SetB(const int8_t* b);
            void Gemm(const uint8_t* A, const int8_t* B, int32_t* C);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedInnerProductInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias);
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
}

#endif
