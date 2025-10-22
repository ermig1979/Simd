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

        virtual void SetParams(const float* ioScale, const uint8_t* ioZero, const int8_t* weight, const float* weightScale, const int32_t* bias, const float* params);

        virtual void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst) = 0;

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
        virtual void SetWeight(const int8_t* weight) = 0;
        virtual void SetBias(const int8_t* weight, const int32_t* bias);
        virtual void SetOther();

        ConvParam _param;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
        Array8u _buffer, _srcZero;
        Array8i _weight;
        Array32i _bias;
        Array32f _weightScale, _norm, _params; 
        int32_t _intZero, _dstZero;
        float _srcScale, _intScale, _dstScale;
        bool _src8u, _dst8u, _is1x1;
        size_t _merge, _sizeS, _sizeD, _elemS, _elemD;
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetQuantizedConvolutionGemm : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionGemm(const ConvParam & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;

        protected:
            virtual void SetWeight(const int8_t* weight);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);
            void Forward(const uint8_t* src, uint8_t* buf, int32_t *sum, uint8_t* dst);

            bool _skipConv;
            size_t _ldW, _ldS, _ldD, _grW, _grS, _grD, _siC, _siK, _siS, _siD, _sizeB;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcGemm : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t batch, K, M;
                size_t F, microD, microM, microK;
                size_t macroD, macroH, macroK;
                size_t bufD, bufM, bufK, elem, dB;
                int reorderType, sumBuf;
            };

            typedef void(*ConvertPtr)(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst);

            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int update, const int8_t* weight, 
                const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3);

            virtual void SetWeight(const int8_t* weight);

            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);
            void Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst);

            AlgParam _alg;
            ConvertPtr _convert;
            ConvolutionPtr _convolutions[2];
        };

        //-------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0 : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p);

            struct AlgParam
            {
                size_t F, microD, microS, microC;
                size_t batch, srcC, srcH, srcW, dstC, K;
                size_t padV, padH, padE, gapV, gapH, kA;
                size_t macroD, macroH, macroC, numH, macroO;
                size_t bufS, bufD, elem;
            };

            typedef void(*PreprocessPtr)(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, int end, uint8_t* dst); 

            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int update, const int8_t* weight, int32_t* dst);

            typedef void(*PostprocessPtr)(const int32_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, 
                const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3);
            virtual void SetWeight(const int8_t* weight);
            void Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst);

            AlgParam _alg;
            Array32i _offset;
            PreprocessPtr _preprocess;
            ConvolutionPtr _convolution;
            PostprocessPtr _postprocess;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV0 : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p, size_t F);

            struct AlgParam
            {
                int32_t srcE, dstE;
            };

            typedef void(*ConvolutionPtr)(const uint8_t* src, uint32_t srcZero, const ConvParam& p, const AlgParam& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst);

        protected:
            virtual void SetWeight(const int8_t* weight);

            AlgParam _alg;
            ConvolutionPtr _convolution;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV1 : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p, size_t F);

            struct AlgParam
            {
                int32_t srcE, dstE;
                size_t F, bufC, bufH, bufW, stepH;
                int reorderType;
            };

            typedef void(*PreprocessPtr)(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, int32_t* dst);
            typedef void(*ConvolutionPtr)(const int32_t* src, const ConvParam& p, const AlgParam& a, const int32_t* weight, const int32_t* bias, const float* norm, size_t yBeg, size_t yEnd, uint32_t zero, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F);
            virtual void SetWeight(const int8_t* weight);

            AlgParam _alg;
            Array32i _weight32i;
            PreprocessPtr _preprocess;
            ConvolutionPtr _convolution;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV2 : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t InternalBufferSize() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p, size_t F);

            struct AlgParam
            {
                int32_t srcE, dstE;
                size_t F, bufC, bufH, bufW, bufR, stepH, sizeW, stepW;
                int reorderType;
            };

            typedef void(*PreprocessPtr)(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, int16_t* dst);
            typedef void(*ConvolutionPtr)(const int16_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, const int16_t* weight, 
                const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F);
            virtual void SetWeight(const int8_t* weight);

            AlgParam _alg;
            Array16i _weight16i;
            PreprocessPtr _preprocess;
            ConvolutionPtr _convolution;
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV3 : public SynetQuantizedConvolution
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV3(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const;
            virtual size_t ExternalBufferSize() const;
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

            static bool Preferable(const ConvParam& p, size_t F);

            struct AlgParam
            {
                int32_t srcE, dstE;
                size_t F, bufC, bufH, bufW, bufR, stepH, sizeW, stepW;
                int reorderType;
            };

            typedef void(*PreprocessPtr)(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst);
            typedef void(*ConvolutionPtr)(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, const int8_t* weight,
                const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst);

        protected:
            void SetAlgParam(size_t F);
            virtual void SetWeight(const int8_t* weight);

            AlgParam _alg;
            PreprocessPtr _preprocess;
            ConvolutionPtr _convolution;
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetQuantizedConvolutionNhwcGemm : public Base::SynetQuantizedConvolutionNhwcGemm
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0: public Base::SynetQuantizedConvolutionNhwcSpecV0
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV0 : public Base::SynetQuantizedConvolutionNhwcDepthwiseV0
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV1 : public Base::SynetQuantizedConvolutionNhwcDepthwiseV1
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV2 : public Base::SynetQuantizedConvolutionNhwcDepthwiseV2
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p);

            virtual String Ext() const { return "Sse41"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetQuantizedConvolutionNhwcGemm : public Sse41::SynetQuantizedConvolutionNhwcGemm
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0 : public Sse41::SynetQuantizedConvolutionNhwcSpecV0
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV0 : public Sse41::SynetQuantizedConvolutionNhwcDepthwiseV0
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV1 : public Sse41::SynetQuantizedConvolutionNhwcDepthwiseV1
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV2 : public Sse41::SynetQuantizedConvolutionNhwcDepthwiseV2
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p);

            virtual String Ext() const { return "Avx2"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetQuantizedConvolutionNhwcGemm : public Avx2::SynetQuantizedConvolutionNhwcGemm
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0 : public Avx2::SynetQuantizedConvolutionNhwcSpecV0
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV0 : public Avx2::SynetQuantizedConvolutionNhwcDepthwiseV0
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV1 : public Avx2::SynetQuantizedConvolutionNhwcDepthwiseV1
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV2 : public Avx2::SynetQuantizedConvolutionNhwcDepthwiseV2
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p);

            virtual String Ext() const { return "Avx512bw"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        class SynetQuantizedConvolutionNhwcGemm : public Avx512bw::SynetQuantizedConvolutionNhwcGemm
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0 : public Avx512bw::SynetQuantizedConvolutionNhwcSpecV0
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV0 : public Avx512bw::SynetQuantizedConvolutionNhwcDepthwiseV0
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV1 : public Avx512bw::SynetQuantizedConvolutionNhwcDepthwiseV1
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV2 : public Avx512bw::SynetQuantizedConvolutionNhwcDepthwiseV2
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcDepthwiseV3 : public Base::SynetQuantizedConvolutionNhwcDepthwiseV3
        {
        public:
            SynetQuantizedConvolutionNhwcDepthwiseV3(const ConvParam& p);

            virtual String Ext() const { return "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE)  
    namespace AmxBf16
    {
        class SynetQuantizedConvolutionNhwcGemm : public Avx512vnni::SynetQuantizedConvolutionNhwcGemm
        {
        public:
            SynetQuantizedConvolutionNhwcGemm(const ConvParam& p);

            virtual String Ext() const { return _alg.microK == 64 ? "AmxBf16" : "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        class SynetQuantizedConvolutionNhwcSpecV0 : public Avx512vnni::SynetQuantizedConvolutionNhwcSpecV0
        {
        public:
            SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p);

            virtual String Ext() const { return _alg.microC == 64 ? "AmxBf16" : "Avx512vnni"; }
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif
}

#endif
