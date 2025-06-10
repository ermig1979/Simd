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
#ifndef __SimdSynetQuantizedConvolution_h__
#define __SimdSynetQuantizedConvolution_h__

#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    SIMD_INLINE bool ValidQuantized(const ConvParam& param)
    {
        if (!param.Valid(SimdTensorData8u, SimdTensorData8u))
            return false;
        return true;
    }

    //------------------------------------------------------------------------------------------------

    class SynetQuantizedConvolution : public Deletable
    {
    public:
        SynetQuantizedConvolution(const ConvParam& p);

        const ConvParam & Param() const { return _param; }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const;
        virtual size_t InternalBufferSize() const;

        virtual void SetParams(const float* srcScale, const uint8_t* srcZero, const int8_t* weight, const float* weightScale, const int32_t* bias, const float* params, const float* dstScale, const uint8_t* dstZero);

        virtual void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst);

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func);
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:
        virtual void SetWeight(const int8_t* weight) = 0;
        virtual void SetBias(const int32_t* bias) = 0;
        virtual void SetOther() = 0;

        virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst) = 0;

        ConvParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
        Array8u _buffer, _srcZero, _dstZero;
        Array8i _weight;
        Array32i _bias;
        Array32f _weightScale, _norm, _params; 
        float _srcScale, _dstScale;
        size_t _merge, _sizeS, _sizeD;
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetQuantizedConvolutionGemmNN : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionGemmNN(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNN"; }
            virtual size_t ExternalBufferSize() const;

        protected:
            virtual void SetWeight(const int8_t* weight);
            virtual void SetBias(const int32_t* bias);
            virtual void SetOther();

            virtual void Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            bool _skipConv;
            size_t _ldW, _ldS, _ldD, _grW, _grS, _grD, _siC, _siK, _siS, _siD, _sizeB;
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
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

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
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
