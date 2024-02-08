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
#ifndef __SimdSynetConvolution32fBf16_h__
#define __SimdSynetConvolution32fBf16_h__

#include "Simd/SimdSynetConvolution32f.h"

namespace Simd
{
    namespace Base
    {
        class SynetConvolution32fBf16Gemm : public SynetConvolution32f
        {
        public:
            SynetConvolution32fBf16Gemm(const ConvParam32f& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::Bf16Gemm"; }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float* src, float* buf, float* dst);

        protected:
            void ImgToCol(const float* src, uint16_t* dst);
            void ImgToRow(const float* src, uint16_t* dst);
            void GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc);

            Array16u _weight;
            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fBf16Nhwc : public SynetConvolution32f
        {
        public:
            SynetConvolution32fBf16Nhwc(const ConvParam32f& p);
            virtual size_t InternalBufferSize() const;

        protected:
            void SetBias(const float* bias, size_t align);
            void SetParams(const float* params, size_t align);

            Array16u _weight;
            Array32f _bias, _params;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fBf16NhwcGemm : public SynetConvolution32fBf16Nhwc
        {
        public:
            SynetConvolution32fBf16NhwcGemm(const ConvParam32f& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const ConvParam32f& p);

            struct AlgParam
            {
                size_t batch, K, M;
                size_t microD, microM, microK;
                size_t macroD, macroH, macroK;
                size_t bufD, bufM, bufK;
            };

            typedef void(*ConvertPtr)(const float* src, const ConvParam32f& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

            typedef void(*ConvolutionPtr)(const uint16_t* src, const ConvParam32f& p, size_t dstC, size_t dstH,
                size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst);

        protected:
            void SetAlgParam(size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3);
            void SetWeight(const float* weight);
            void Forward(const float* src, uint16_t* buf, float* dst);

            AlgParam _alg;
            ConvertPtr _convert;
            ConvolutionPtr _convolutions[2];
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution32fBf16NhwcOld : public SynetConvolution32f
        {
        public:
            SynetConvolution32fBf16NhwcOld(const ConvParam32f& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params);
            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const ConvParam32f& p);

            struct AlgParam
            {
                int mode;
                size_t microC, microD, macroH, macroC, macroD;
                size_t batch, srcH, srcW;
            };

            typedef void(*ConvertPtr)(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, size_t micC, uint16_t* dst);

            typedef void(*ConvolutionPtr)(const uint16_t* src, const ConvParam32f& p, size_t dstC, size_t dstH, 
                size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst);

        protected:
            void SetAlgParam(size_t microD, size_t microHW, size_t microC, size_t L1, size_t L2, size_t L3);
            int PreferableMode(size_t microD, size_t microHW, size_t microC, size_t L1, size_t L2, size_t L3);
            void SetWeight(const float* weight);
            void SetBias(const float* bias);
            void SetParams(const float* params);
            void ForwardConv(const float* src, uint16_t* buf, float* dst);
            void ForwardGemm(const float* src, uint16_t* buf, float* dst);
            size_t Offset(size_t yBeg, size_t cBeg, size_t cEnd);

            Array16u _weight;
            Array32f _bias, _params;
            AlgParam _alg;
            ConvertPtr _convert;
            ConvolutionPtr _convolutions[2];
        };
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetConvolution32fBf16NhwcGemm : public Base::SynetConvolution32fBf16NhwcGemm
        {
        public:
            SynetConvolution32fBf16NhwcGemm(const ConvParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution32fBf16NhwcOld : public Base::SynetConvolution32fBf16NhwcOld
        {
        public:
            SynetConvolution32fBf16NhwcOld(const ConvParam32f& p);

            virtual String Ext() const { return "Sse41"; }
        };
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetConvolution32fBf16NhwcGemm : public Sse41::SynetConvolution32fBf16NhwcGemm
        {
        public:
            SynetConvolution32fBf16NhwcGemm(const ConvParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution32fBf16NhwcOld : public Sse41::SynetConvolution32fBf16NhwcOld
        {
        public:
            SynetConvolution32fBf16NhwcOld(const ConvParam32f& p);

            virtual String Ext() const { return "Avx2"; }
        };
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetConvolution32fBf16NhwcGemm : public Avx2::SynetConvolution32fBf16NhwcGemm
        {
        public:
            SynetConvolution32fBf16NhwcGemm(const ConvParam32f& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //-----------------------------------------------------------------------------------------

        void ConvolutionBf16NhwcConvertConv(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, size_t micC, uint16_t* dst);

        void ConvolutionBf16NhwcConvertGemm(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, size_t micC, uint16_t* dst);

        //-----------------------------------------------------------------------------------------

        class SynetConvolution32fBf16NhwcOld : public Avx2::SynetConvolution32fBf16NhwcOld
        {
        public:
            SynetConvolution32fBf16NhwcOld(const ConvParam32f& p);

            virtual String Ext() const { return "Avx512bw"; }
        };
    }
#endif

#if defined(SIMD_AVX512BF16_ENABLE) && 0    
    namespace Avx512bf16
    {
        void ConvolutionBf16NhwcConvertConv(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, size_t micC, uint16_t* dst);

        void ConvolutionBf16NhwcConvertGemm(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, size_t micC, uint16_t* dst);

        //-----------------------------------------------------------------------------------------

        class SynetConvolution32fBf16Nhwc : public Avx512bw::SynetConvolution32fBf16Nhwc
        {
        public:
            SynetConvolution32fBf16Nhwc(const ConvParam32f& p);

            virtual String Ext() const { return "Avx512bf16"; }
        };

        //-----------------------------------------------------------------------------------------

        void* SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        class SynetConvolution32fBf16NhwcGemm : public Avx512bw::SynetConvolution32fBf16NhwcGemm
        {
        public:
            SynetConvolution32fBf16NhwcGemm(const ConvParam32f& p);

            virtual String Ext() const { return "AmxBf16"; }
        };
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif
}

#endif
