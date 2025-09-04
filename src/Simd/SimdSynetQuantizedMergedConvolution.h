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
#ifndef __SimdSynetMergedQuantizedConvolution_h__
#define __SimdSynetMergedQuantizedConvolution_h__

#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
    class SynetQuantizedMergedConvolution : public Deletable
    {
    public:
        SynetQuantizedMergedConvolution(const MergConvParam& p);

        const MergConvParam & Param() const { return _param; }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const;
        virtual size_t InternalBufferSize() const;

        virtual void SetParams(const float* ioScale, const uint8_t* ioZero, const int8_t* const* weight, const float* const* weightScale, const int32_t* const* bias);

        virtual void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func);
#endif

        uint8_t* Buffer(uint8_t* buffer);

        const char* Info() const;

    protected:
        virtual void SetInput(const int8_t* weight, const ConvParam& p, Array8i & dst) = 0;
        virtual void SetDepthwise(const int8_t* weight, const ConvParam& p, Array8i& dst) = 0;
        virtual void SetOutput(const int8_t* weight, const ConvParam& p, Array8i& dst) = 0;
        virtual void SetZero(uint8_t zero, const ConvParam& p, Array8u& dst) = 0;
        virtual void SetBias(const int8_t* weight, const int32_t* bias, int32_t zero, const ConvParam& p, Array32i & dst);
        virtual void SetNorm(const float* weightScale, float srcScale, float dstScale, const ConvParam& p, Array32f& dst);

        MergConvParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
        Array8u _buffer, _dwSrcZero;
        Array8i _weight[3];
        Array32i _bias[3];
        Array32f _norm[3];
        float _ioScale[5], _srcNorm, _dstNorm, _addScale;
        int32_t _ioZero[5], _addZero, _srcBias, _dstBias;
        size_t _batch, _merge, _count, _sizeS, _sizeD;
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetQuantizedMergedConvolutionRef : public Simd::SynetQuantizedMergedConvolution
        {
        public:
            SynetQuantizedMergedConvolutionRef(const MergConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::Ref"; }
            virtual size_t ExternalBufferSize() const;

        protected:
            virtual void SetInput(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetDepthwise(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetOutput(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetZero(uint8_t zero, const ConvParam& p, Array8u& dst);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            void Depthwise(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const int8_t* weight, int32_t* dst);
            void AddSrc(const uint8_t* src, uint8_t* dst);

            size_t _sizeB;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolution : public Simd::SynetQuantizedMergedConvolution
        {
        public:
            SynetQuantizedMergedConvolution(const MergConvParam& p);

            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext(); }
            virtual size_t ExternalBufferSize() const;

            struct AlgParam
            {
                size_t miC, maC, miK, dbE;
                size_t dsStep, dbStep, ddStep, dsStart;
                size_t isH, dsH, dbH, ddH;
                size_t isB, idB, dsB, dbB, ddB, odB;
                size_t dbW, dwC, dwStep, dwSize, iwStep, owStep;
            };

            typedef void(*InputPreprocessPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst);

            typedef void(*InputConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t * sum, uint8_t* dst);

            typedef void(*DepthwisePreprocessPtr)(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, uint8_t* dst);

            typedef void(*DepthwiseConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst);

            typedef void(*OutputConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* sum, uint8_t* dst);

            typedef void(*AddInputToOutputPtr)(const uint8_t* a, int aBias, float aNorm, const uint8_t* b, int bBias, float bNorm, 
                const ConvParam& p, size_t yBeg, size_t yEnd, float dNorm, int dZero, uint8_t* dst);

        protected:
            virtual void SetInput(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetDepthwise(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetOutput(const int8_t* weight, const ConvParam& p, Array8i& dst);
            virtual void SetZero(uint8_t zero, const ConvParam& p, Array8u& dst);

            AlgParam _alg;
            InputPreprocessPtr _inputPreprocess;
            InputConvolutionPtr _inputConvolution;
            DepthwisePreprocessPtr _depthwisePreprocess;
            DepthwiseConvolutionPtr _depthwiseConvolution;
            OutputConvolutionPtr _outputConvolution[2];
            AddInputToOutputPtr _addInputToOutput;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCdc : public SynetQuantizedMergedConvolution
        {
        public:
            SynetQuantizedMergedConvolutionCdc(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK, size_t dbE);
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCd : public SynetQuantizedMergedConvolution
        {
        public:
            SynetQuantizedMergedConvolutionCd(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK, size_t dbE);
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionDc : public SynetQuantizedMergedConvolution
        {
        public:
            SynetQuantizedMergedConvolutionDc(const MergConvParam& p);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const MergConvParam& p);

        protected:
            void SetSize(size_t miC, size_t miK, size_t dbE);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void SetInputPreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputPreprocessPtr& func);

        void SetInputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputConvolutionPtr & func);

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func);

        void SetDepthwiseConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwiseConvolutionPtr& func);

        void SetOutputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::OutputConvolutionPtr *funcs);

        void SetAddInputToOutput(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::AddInputToOutputPtr& func);

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCdc : public Base::SynetQuantizedMergedConvolutionCdc
        {
        public:
            SynetQuantizedMergedConvolutionCdc(const MergConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCd : public Base::SynetQuantizedMergedConvolutionCd
        {
        public:
            SynetQuantizedMergedConvolutionCd(const MergConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionDc : public Base::SynetQuantizedMergedConvolutionDc
        {
        public:
            SynetQuantizedMergedConvolutionDc(const MergConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void SetInputPreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputPreprocessPtr& func);

        void SetInputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputConvolutionPtr& func);

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func);

        void SetDepthwiseConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwiseConvolutionPtr& func);

        void SetOutputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::OutputConvolutionPtr* funcs);

        void SetAddInputToOutput(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::AddInputToOutputPtr& func);

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCdc : public Sse41::SynetQuantizedMergedConvolutionCdc
        {
        public:
            SynetQuantizedMergedConvolutionCdc(const MergConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCd : public Sse41::SynetQuantizedMergedConvolutionCd
        {
        public:
            SynetQuantizedMergedConvolutionCd(const MergConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionDc : public Sse41::SynetQuantizedMergedConvolutionDc
        {
        public:
            SynetQuantizedMergedConvolutionDc(const MergConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void SetInputPreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputPreprocessPtr& func);

        void SetInputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputConvolutionPtr& func);

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func);

        void SetDepthwiseConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwiseConvolutionPtr& func);

        void SetOutputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::OutputConvolutionPtr* funcs);

        void SetAddInputToOutput(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::AddInputToOutputPtr& func);

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCdc : public Avx2::SynetQuantizedMergedConvolutionCdc
        {
        public:
            SynetQuantizedMergedConvolutionCdc(const MergConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionCd : public Avx2::SynetQuantizedMergedConvolutionCd
        {
        public:
            SynetQuantizedMergedConvolutionCd(const MergConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedMergedConvolutionDc : public Avx2::SynetQuantizedMergedConvolutionDc
        {
        public:
            SynetQuantizedMergedConvolutionDc(const MergConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add);
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
