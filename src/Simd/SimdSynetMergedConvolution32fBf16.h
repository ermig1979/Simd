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
#ifndef __SimdSynetMergedConvolution32fBf16_h__
#define __SimdSynetMergedConvolution32fBf16_h__

#include "Simd/SimdSynetMergedConvolution32f.h"

namespace Simd
{
    namespace Base
    {
        class SynetMergedConvolution32fBf16 : public Simd::SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fBf16(const MergConvParam32f& p);

            virtual String Desc() const { return Ext() + "-bf16"; }
            virtual String Ext() const { return "Base"; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params);
            virtual void Forward(const float* src, float* buf, float* dst);

            struct AlgParam
            {
                size_t miC, maC, miK, yStep[3], yStart[3], bufH[3], dp[2], dw[3];
            };

            typedef void(*ConvertPtr)(const float* src, const ConvParam32f& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

            typedef void(*InputConvolutionPtr)(const uint16_t* src, const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const uint16_t* weight, const float* bias, const float* params, float* dst);

            typedef void(*DepthwiseConvolutionPtr)(const float* src, const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const float* weight, const float* bias, const float* params, uint16_t* dst);

            typedef void(*OutputConvolutionPtr)(const uint16_t* src, const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
                const uint16_t* weight, const float* bias, const float* params, float* dst, int zero);

        protected:
            void SetInputWeight(const float* src, const ConvParam32f& p);
            void SetDepthwiseWeight(const float* src, const ConvParam32f& p);
            void SetOutputWeight(const float* src, const ConvParam32f& p);
            void SetBias(const float* src, const ConvParam32f& p, Array32f & dst);
            void SetParams(const float* src, const ConvParam32f& p, Array32f& dst);

            bool _dw0;
            ConvertPtr _convert;
            InputConvolutionPtr _input;
            DepthwiseConvolutionPtr _depthwise;
            OutputConvolutionPtr _output[2];
            size_t _sizeS, _sizeD, _sizeB[3];
            AlgParam _alg;
            Array16u _weightI, _weightO;
            Array32f _weightD, _bias[3], _params[3];
        };

        class SynetMergedConvolution32fBf16Cdc : public SynetMergedConvolution32fBf16
        {
        public:
            SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };

        class SynetMergedConvolution32fBf16Cd : public SynetMergedConvolution32fBf16
        {
        public:
            SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };

        class SynetMergedConvolution32fBf16Dc : public SynetMergedConvolution32fBf16
        {
        public:
            SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t miC, size_t miK);
        };
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void SetInput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr* output);

        class SynetMergedConvolution32fBf16Cdc : public Base::SynetMergedConvolution32fBf16Cdc
        {
        public:
            SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution32fBf16Cd : public Base::SynetMergedConvolution32fBf16Cd
        {
        public:
            SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetMergedConvolution32fBf16Dc : public Base::SynetMergedConvolution32fBf16Dc
        {
        public:
            SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void SetInput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr* output);

        class SynetMergedConvolution32fBf16Cdc : public Sse41::SynetMergedConvolution32fBf16Cdc
        {
        public:
            SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution32fBf16Cd : public Sse41::SynetMergedConvolution32fBf16Cd
        {
        public:
            SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetMergedConvolution32fBf16Dc : public Sse41::SynetMergedConvolution32fBf16Dc
        {
        public:
            SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void ConvertFp32ToBf16(const float* src, const ConvParam32f& p, const Base::SynetMergedConvolution32fBf16::AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

        void SetInput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr* output);

        class SynetMergedConvolution32fBf16Cdc : public Avx2::SynetMergedConvolution32fBf16Cdc
        {
        public:
            SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetMergedConvolution32fBf16Cd : public Avx2::SynetMergedConvolution32fBf16Cd
        {
        public:
            SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetMergedConvolution32fBf16Dc : public Avx2::SynetMergedConvolution32fBf16Dc
        {
        public:
            SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p);

            virtual String Ext() const { return "Avx512bw"; }
        };
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
        void SetInput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::InputConvolutionPtr& input);

        void SetDepthwise(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::DepthwiseConvolutionPtr& depthwise);

        void SetOutput(const ConvParam32f& p, Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr* output);

        class SynetMergedConvolution32fBf16Cdc : public Avx512bw::SynetMergedConvolution32fBf16Cdc
        {
        public:
            SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetMergedConvolution32fBf16Cd : public Avx512bw::SynetMergedConvolution32fBf16Cd
        {
        public:
            SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetMergedConvolution32fBf16Dc : public Avx512bw::SynetMergedConvolution32fBf16Dc
        {
        public:
            SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility);
    }
#endif
}
#endif
