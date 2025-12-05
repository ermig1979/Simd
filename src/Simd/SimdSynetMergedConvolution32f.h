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
#ifndef __SimdSynetMergedConvolution32f_h__
#define __SimdSynetMergedConvolution32f_h__

#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    class SynetMergedConvolution32f : public Deletable
    {
    public:
        SynetMergedConvolution32f(const MergConvParam& p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
        }

        virtual const MergConvParam& Param() const
        { 
            return _param; 
        }

        virtual size_t ExternalBufferSize() const = 0;

        virtual size_t InternalBufferSize() const = 0;

        virtual void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params) = 0;

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        virtual Base::PerformanceMeasurer* Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        virtual const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

        virtual String Desc() const = 0;

        virtual String Ext() const = 0;

    protected:
        MergConvParam _param;
        Array32f _buffer;

        float* Buffer(float* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

    private:
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* _perf;
#endif        
        mutable String _info;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetMergedConvolution32f : public Simd::SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32f(const MergConvParam& p);

            virtual String Desc() const { return Ext() + "-fp32"; }
            virtual String Ext() const { return "Base"; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params);
            virtual void Forward(const float* src, float* buf, float* dst);

            typedef void(*ConvolutionPtr)(const float* src, const SimdConvolutionParameters& p, size_t maC, size_t yBeg, size_t yEnd,
                const size_t * bufH, const float* weight, const float* bias, const float* params, float* dst, int first);

        protected:
            virtual void ReorderFirstWeight(const float* src, float* dst) const {}
            virtual void ReorderSecondWeight(const float* src, float* dst) const {}
            virtual void ReorderThirdWeight(const float* src, float* dst) const {}

            ConvolutionPtr _convolution[4];
            size_t _sizeS, _sizeD, _sizeB[2];
            Array32f _rWeight[3], _rBias[3], _rParams[3];
            const float * _weight[3], * _bias[3], * _params[3];

            size_t _miC, _maC, _yStep[2], _bufH[2], _dp[2], _dw[3];
        };

        class SynetMergedConvolution32fCdc : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam& p);

            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
            virtual void ReorderThirdWeight(const float* src, float* dst) const;
        };

        class SynetMergedConvolution32fCd : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
        };

        class SynetMergedConvolution32fDc : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetDepthwise(const ConvParam& p, bool last, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution32fCdc : public Base::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution32fCd : public Base::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution32fDc : public Base::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetDepthwise(const ConvParam& p, bool last, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution32fCdc : public Sse41::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution32fCd : public Sse41::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution32fDc : public Sse41::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        bool SetDepthwise3x3(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        bool SetDepthwise7x7(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution32fCdc : public Avx2::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Avx2::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fDc : public Avx2::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetMergedConvolution32fCdc : public Base::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam& p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Base::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam& p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };
        
        class SynetMergedConvolution32fDc : public Base::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam& p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif
}
#endif
