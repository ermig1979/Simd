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
#ifndef __SimdSynetConvolution16b_h__
#define __SimdSynetConvolution16b_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"
#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
    class SynetConvolution16b : public Deletable
    {
    public:
        SynetConvolution16b(const ConvParam& p);

        const ConvParam& Param() const
        {
            return _param;
        }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const
        {
            return 1;
        }

        virtual size_t InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() +
                _bias.RawSize() + _params.RawSize();
        }

        virtual void SetParams(const float* weight, const float* bias, const float* params) = 0;

        virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst) = 0;

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

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func);
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:

        ConvParam _param;
        Array8u _buffer;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* _perf;
#endif
        mutable String _info;
        Array16u _weight;
        Array32f _bias, _params;
        bool _src16b, _dst16b, _is1x1;
        size_t _elemS, _elemD, _stepS, _stepD;

        void SetBias(const float* bias, size_t align);
        void SetParams(const float* params, size_t align);
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetConvolution16bGemm : public SynetConvolution16b
        {
        public:
            SynetConvolution16bGemm(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::Gemm"; }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

        protected:
            void ImgToCol(const uint16_t* src, uint16_t* dst);
            void ImgToRow(const uint16_t* src, uint16_t* dst);

            void GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc);

            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution16bNhwcGemm : public SynetConvolution16b
        {
        public:
            SynetConvolution16bNhwcGemm(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t batch, K, M;
                size_t F, microD, microM, microK;
                size_t macroD, macroH, macroK;
                size_t bufD, bufM, bufK, elem, dB;
                int reorderType, sumBuf;
            };

            typedef void(*ConvertPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst);

            typedef void(*ConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH,
                size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* sum, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3);
            virtual void SetWeight(const float* weight);
            void Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst);

            AlgParam _alg;
            ConvertPtr _convert;
            ConvolutionPtr _convolutions[2];
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution16bNhwcDirect : public SynetConvolution16b
        {
        public:
            SynetConvolution16bNhwcDirect(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t batch, srcC, srcH, srcW, dstC, K;
                size_t F, microD, microS, microC;
                size_t macroD, macroH, macroC;
                size_t bufS, bufD, elem;
            };

            typedef void(*PreprocessPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint16_t* dst);

            typedef void(*ConvolutionPtr)(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, float* dst);

            typedef void(*PostprocessPtr)(const float* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3);
            virtual void SetWeight(const float* weight);
            void Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst);

            AlgParam _alg;
            PreprocessPtr _preprocess;
            ConvolutionPtr _convolution;
            PostprocessPtr _postprocess;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution16bNhwcDepthwise : public SynetConvolution16b
        {
        public:
            SynetConvolution16bNhwcDepthwise(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p);

            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst);

        protected:
            size_t _sizeS, _sizeD;
            Array32f _weight;
            ConvolutionPtr _convolution;
        };

        //-------------------------------------------------------------------------------------------------

        class SynetConvolution16bNchwGemm : public SynetConvolution16b
        {
        public:
            SynetConvolution16bNchwGemm(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t K, N;
                size_t F, microD, microN, microK;
                size_t macroD, macroH, macroK;
                size_t bufD, bufN, bufK, elem;
                int reorderType, sumBuf;
            };

            typedef void(*ConvertPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst);

            typedef void(*ConvolutionPtr)(const uint16_t* weight, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH,
                size_t K, int zero, const uint16_t* src, const float* bias, const float* params, float* sum, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3);
            virtual void SetWeight(const float* weight);
            void Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst);

            AlgParam _alg;
            ConvertPtr _convert;
            ConvolutionPtr _convolutions[2];
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetConvolution16bNhwcGemm : public Base::SynetConvolution16bNhwcGemm
        {
        public:
            SynetConvolution16bNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution16bNhwcDirect : public Base::SynetConvolution16bNhwcDirect
        {
        public:
            SynetConvolution16bNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution16bNhwcDepthwise : public Base::SynetConvolution16bNhwcDepthwise
        {
        public:
            SynetConvolution16bNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        class SynetConvolution16bNchwGemm : public Base::SynetConvolution16bNchwGemm
        {
        public:
            SynetConvolution16bNchwGemm(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetConvolution16bNhwcGemm : public Sse41::SynetConvolution16bNhwcGemm
        {
        public:
            SynetConvolution16bNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution16bNhwcDirect : public Sse41::SynetConvolution16bNhwcDirect
        {
        public:
            SynetConvolution16bNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution16bNhwcDepthwise : public Sse41::SynetConvolution16bNhwcDepthwise
        {
        public:
            SynetConvolution16bNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        class SynetConvolution16bNchwGemm : public Sse41::SynetConvolution16bNchwGemm
        {
        public:
            SynetConvolution16bNchwGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetConvolution16bNhwcGemm : public Avx2::SynetConvolution16bNhwcGemm
        {
        public:
            SynetConvolution16bNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution16bNhwcDirect : public Avx2::SynetConvolution16bNhwcDirect
        {
        public:
            SynetConvolution16bNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };


        class SynetConvolution16bNhwcDepthwise : public Avx2::SynetConvolution16bNhwcDepthwise
        {
        public:
            SynetConvolution16bNhwcDepthwise(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        class SynetConvolution16bNchwGemm : public Avx2::SynetConvolution16bNchwGemm
        {
        public:
            SynetConvolution16bNchwGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif

#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        class SynetConvolution16bNhwcGemm : public Avx512bw::SynetConvolution16bNhwcGemm
        {
        public:
            SynetConvolution16bNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetConvolution16bNhwcDirect : public Avx512bw::SynetConvolution16bNhwcDirect
        {
        public:
            SynetConvolution16bNhwcDirect(const ConvParam& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        class SynetConvolution16bNchwGemm : public Avx512bw::SynetConvolution16bNchwGemm
        {
        public:
            SynetConvolution16bNchwGemm(const ConvParam& p);

            virtual String Ext() const { return "AmxBf16"; }
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
#endif
}

#endif
