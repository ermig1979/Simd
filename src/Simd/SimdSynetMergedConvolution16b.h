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
#ifndef __SimdSynetMergedConvolution16b_h__
#define __SimdSynetMergedConvolution16b_h__

#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdArray.h"

namespace Simd
{
    class SynetMergedConvolution16b : public Deletable
    {
    public:
        virtual const MergConvParam& Param() const = 0;

        virtual size_t ExternalBufferSize() const = 0;

        virtual size_t InternalBufferSize() const = 0;

        virtual void SetParams(const float* const* weight, const float* const* bias, const float* const* params) = 0;

        virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        virtual Base::PerformanceMeasurer* Perf(const char* func) = 0;
#endif
        virtual const char* Info() const = 0;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetMergedConvolution16b : public Simd::SynetMergedConvolution16b
        {
        public:
            SynetMergedConvolution16b(const MergConvParam& p);

            virtual const MergConvParam& Param() const {  return _param; };
            virtual String Desc() const { return Ext(); }
            virtual String Ext() const { return "Base"; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float* const* weight, const float* const* bias, const float* const* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            virtual Base::PerformanceMeasurer* Perf(const char* func);
#endif
            virtual const char* Info() const;

            struct AlgParam
            {
                size_t miC, maC, miK, yStep[3], yStart[3], bufH[3], dp[2], dw[3], elem[2];
            };

            typedef void(*ConvertToBf16Ptr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

            typedef void(*ConvertToFp32Ptr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst);

            typedef void(*InputConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const uint16_t* weight, const float* bias, const float* params, float* dst);

            typedef void(*DepthwiseConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const float* weight, const float* bias, const float* params, uint8_t* dst);

            typedef void(*OutputConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                int zero, const uint16_t* weight, const float* bias, const float* params, float* sum, uint8_t* dst);

        protected:
            void SetInputWeight(const float* src, const ConvParam& p);
            void SetDepthwiseWeight(const float* src, const ConvParam& p);
            void SetOutputWeight(const float* src, const ConvParam& p);
            void SetBias(const float* src, const ConvParam& p, Array32f& dst);
            void SetParams(const float* src, const ConvParam& p, Array32f& dst);
            uint8_t* Buffer(uint8_t* buffer);

            MergConvParam _param;
            mutable String _info;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer* _perf;
#endif
            bool _dw0, _src16b, _dst16b;
            ConvertToBf16Ptr _toBf16;
            ConvertToFp32Ptr _toFp32;
            InputConvolutionPtr _input;
            DepthwiseConvolutionPtr _depthwise;
            OutputConvolutionPtr _output[2];
            size_t _sizeS, _sizeD, _sizeB[4];
            AlgParam _alg;
            Array8u _buffer;
            Array16u _weightI, _weightO;
            Array32f _weightD, _bias[3], _params[3];
        };

        class SynetMergedConvolution16bCdc : public SynetMergedConvolution16b
        {
        public:
            SynetMergedConvolution16bCdc(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };

        class SynetMergedConvolution16bCd : public SynetMergedConvolution16b
        {
        public:
            SynetMergedConvolution16bCd(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };

        class SynetMergedConvolution16bDc : public SynetMergedConvolution16b
        {
        public:
            SynetMergedConvolution16bDc(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution16b::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution16b::OutputConvolutionPtr* output);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution16bCdc : public Base::SynetMergedConvolution16bCdc
        {
        public:
            SynetMergedConvolution16bCdc(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution16bCd : public Base::SynetMergedConvolution16bCd
        {
        public:
            SynetMergedConvolution16bCd(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution16bDc : public Base::SynetMergedConvolution16bDc
        {
        public:
            SynetMergedConvolution16bDc(const MergConvParam& p);
            virtual String Ext() const { return "Sse41"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution16b::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution16b::OutputConvolutionPtr* output);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution16bCdc : public Sse41::SynetMergedConvolution16bCdc
        {
        public:
            SynetMergedConvolution16bCdc(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution16bCd : public Sse41::SynetMergedConvolution16bCd
        {
        public:
            SynetMergedConvolution16bCd(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution16bDc : public Sse41::SynetMergedConvolution16bDc
        {
        public:
            SynetMergedConvolution16bDc(const MergConvParam& p);
            virtual String Ext() const { return "Avx2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution16b::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution16b::OutputConvolutionPtr* output);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution16bCdc : public Avx2::SynetMergedConvolution16bCdc
        {
        public:
            SynetMergedConvolution16bCdc(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetMergedConvolution16bCd : public Avx2::SynetMergedConvolution16bCd
        {
        public:
            SynetMergedConvolution16bCd(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetMergedConvolution16bDc : public Avx2::SynetMergedConvolution16bDc
        {
        public:
            SynetMergedConvolution16bDc(const MergConvParam& p);
            virtual String Ext() const { return "Avx512bw"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
        void SetInput(const ConvParam& p, Base::SynetMergedConvolution16b::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam& p, Base::SynetMergedConvolution16b::OutputConvolutionPtr* output);

        //-------------------------------------------------------------------------------------------------

        class SynetMergedConvolution16bCdc : public Avx512bw::SynetMergedConvolution16bCdc
        {
        public:
            SynetMergedConvolution16bCdc(const MergConvParam& p);
            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetMergedConvolution16bCd : public Avx512bw::SynetMergedConvolution16bCd
        {
        public:
            SynetMergedConvolution16bCd(const MergConvParam& p);
            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetMergedConvolution16bDc : public Avx512bw::SynetMergedConvolution16bDc
        {
        public:
            SynetMergedConvolution16bDc(const MergConvParam& p);
            virtual String Ext() const { return "AmxBf16"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif
}
#endif
