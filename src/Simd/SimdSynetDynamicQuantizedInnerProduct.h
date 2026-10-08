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
#ifndef __SimdSynetDynamicQuantizedInnerProduct_h__
#define __SimdSynetDynamicQuantizedInnerProduct_h__

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
        class SynetDynamicQuantizedInnerProduct : public Deletable
        {
        public:
            SynetDynamicQuantizedInnerProduct(const DynamicQuantizedInnerProductParam& p);

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
            typedef void (*SynetQuantizeLinearPtr)(const float* src, size_t size, const float* norm, int32_t zero, uint8_t* dst);

        protected:
            virtual void SetWeight(const int8_t* weight) = 0;
            void SetInputScaleZero(const float* src, size_t size);

            DynamicQuantizedInnerProductParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer* _perf;
#endif
            mutable String _info;
            Array8u _buffer;
            Array8i _weight;
            Array32i _sums;
            Array32f _scale, _bias, _params;
            size_t _sizeA, _sizeB, _sizeC, _sizeS, _aN;
            MinMax32fPtr _minMax32f;
            float _aScale;
            uint8_t _aZero;
            SynetQuantizeLinearPtr _synetQuantizeLinear;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetDynamicQuantizedInnerProductRef : public SynetDynamicQuantizedInnerProduct
        {
        public:
            SynetDynamicQuantizedInnerProductRef(const DynamicQuantizedInnerProductParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const float* A, uint8_t* buf, float* C);

        protected:
            virtual void SetWeight(const int8_t* weight);
        };


        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProductInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);
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

#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
    }
#endif
}

#endif
