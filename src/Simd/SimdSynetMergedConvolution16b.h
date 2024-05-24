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

        virtual void SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params) = 0;

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

            virtual String Desc() const { return Ext(); }
            virtual String Ext() const { return "Base"; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            struct AlgParam
            {
                size_t miC, maC, miK, yStep[3], yStart[3], bufH[3], dp[2], dw[3];
            };

            typedef void(*ConvertPtr)(const float* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

            typedef void(*InputConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const uint16_t* weight, const float* bias, const float* params, float* dst);

            typedef void(*DepthwiseConvolutionPtr)(const float* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const float* weight, const float* bias, const float* params, uint16_t* dst);

            typedef void(*OutputConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const uint16_t* weight, const float* bias, const float* params, float* dst, int zero);

        protected:
            void SetInputWeight(const float* src, const ConvParam& p);
            void SetDepthwiseWeight(const float* src, const ConvParam& p);
            void SetOutputWeight(const float* src, const ConvParam& p);
            void SetBias(const float* src, const ConvParam& p, Array32f& dst);
            void SetParams(const float* src, const ConvParam& p, Array32f& dst);
            uint8_t* Buffer(uint8_t* buffer);

            MergConvParam _param;
            bool _dw0, _src16b, _dst16b;
            ConvertPtr _convert;
            InputConvolutionPtr _input;
            DepthwiseConvolutionPtr _depthwise;
            OutputConvolutionPtr _output[2];
            size_t _sizeS, _sizeD, _sizeB[4];
            size_t _elemS, _elemD, _stepS, _stepD;
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

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility);
    }
}
#endif
