/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdSynetDeconvolution32f_h__
#define __SimdSynetDeconvolution32f_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"
#include "Simd/SimdGemm.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct DeconvParam32f : public SimdConvolutionParameters
    {
        SimdBool trans;
        size_t batch;
        SimdGemm32fNNPtr gemm;

        DeconvParam32f(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            *((SimdConvolutionParameters*)this) = *conv;
            this->trans = (srcF == SimdTensorFormatNhwc ? SimdTrue : SimdFalse);
            this->batch = batch;
            this->gemm = gemm;
        }

        bool Valid()
        {
            return 
                dstH == strideY * (srcH - 1) + dilationY * (kernelY - 1) + 1 - padY - padH && dstH > 0 &&
                dstW == strideX * (srcW - 1) + dilationX * (kernelX - 1) + 1 - padX - padW && dstW > 0 &&
                srcT == SimdTensorData32f && dstT == SimdTensorData32f && 
                srcF == dstF && (srcF == SimdTensorFormatNchw || (srcF == SimdTensorFormatNhwc && group == 1));
        }

        SIMD_INLINE bool IsKernel(size_t value) const
        {
            return kernelY == value && kernelX == value;
        }

        SIMD_INLINE bool IsDilation(size_t value) const
        {
            return dilationY == value && dilationX == value;
        }

        SIMD_INLINE bool IsStride(size_t value) const
        {
            return strideY == value && strideX == value;
        }

        SIMD_INLINE bool IsPad(size_t value) const
        {
            return padY == value && padX == value && padH == value && padW == value;
        }

        SIMD_INLINE bool IsDepthwise() const
        {
            return srcC == group && dstC == group;
        }
        SIMD_INLINE bool Is1x1() const
        {
            return IsKernel(1) && IsDilation(1) && IsStride(1) && IsPad(0);
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << strideX << "-" << Simd::Max(padX, padW) << "-" << group << "-" << trans;
            return ss.str();
        }

        int64_t Flop() const
        {
            return int64_t(batch) * kernelY * kernelX * srcC * srcH * srcW * dstC / group * 2;
        }
#endif
    };

    class SynetDeconvolution32f : public Deletable
    {
    public:
        SynetDeconvolution32f(const DeconvParam32f & p)
            : _param(p)
            , _0(0.0f)
            , _1(1.0f)
            , _nhwcRun(0)
            , _nhwcReorderB(0)
            , _biasAndActivation(0)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
        }

        const DeconvParam32f & Param() const
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
            return _buffer.size + _nhwcWeight.size + _weightT.size;
        }

        virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            _weight = weight;
            if (internal)
                *internal = SimdFalse;
            _bias = bias;
            _params = params;
        }

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

        float * Buffer(float * buffer)
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
        Base::PerformanceMeasurer* Perf(const String& func);
#endif

        const char * Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:
        typedef void(*NhwcReorderB)(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        typedef void(*NhwcRun)(size_t M, size_t N, size_t K, const float * A, const float * B, float * C, GemmKernelType type, bool compatibility);
        typedef void(*BiasAndActivation)(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst);

        DeconvParam32f _param;
        Array32f _buffer;
        float _0, _1;
        const float * _weight, * _bias, * _params;
        RuntimeGemm _gemm;
        RuntimeGemmCb _gemmCb;
        Array32f _nhwcWeight, _weightT;
        NhwcRun _nhwcRun;
        NhwcReorderB _nhwcReorderB;
        BiasAndActivation _biasAndActivation;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer * _perf;
#endif
        mutable String _info;
    };

    namespace Base
    {
        class SynetDeconvolution32fGemmNN : public SynetDeconvolution32f
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNN" + (_merge > 1 ? "-" + ToStr(_merge) : ""); }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

        protected:
            virtual void ColToImg(const float * src, float * dst);
            virtual void RowToImg(const float * src, float * dst);

            bool _is1x1;
            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD, _merge;
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public SynetDeconvolution32f
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::NhwcDirect2x2"; }
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params);
            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const DeconvParam32f & p);

            struct AlgParam
            {
                size_t microD, macroH, macroC, macroD;
            };
            typedef void(*DeconvolutionPtr)(const float * src, const DeconvParam32f & p, const AlgParam & a, const float * weight, const float * bias, const float * params, float * dst);

        protected:
            void SetAlgParam(size_t F, size_t L1, size_t L2, size_t L3);
            void ReorderWeight(const float * src, float * dst);

            size_t _sizeS, _sizeD;
            AlgParam _alg;
            Array32f _rWeight, _rBias, _rParams;
            DeconvolutionPtr _deconvolution;
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class SynetDeconvolution32fGemmNN : public Base::SynetDeconvolution32fGemmNN
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Sse2"; }
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public Base::SynetDeconvolution32fNhwcDirect2x2
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Sse2"; }

            static bool Preferable(const DeconvParam32f & p);
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class SynetDeconvolution32fGemmNN : public Sse2::SynetDeconvolution32fGemmNN
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx"; }
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public Sse2::SynetDeconvolution32fNhwcDirect2x2
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx"; }
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetDeconvolution32fGemmNN : public Avx::SynetDeconvolution32fGemmNN
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx2"; }
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public Avx::SynetDeconvolution32fNhwcDirect2x2
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx2"; }
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class SynetDeconvolution32fGemmNN : public Avx2::SynetDeconvolution32fGemmNN
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx512f"; }
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public Avx2::SynetDeconvolution32fNhwcDirect2x2
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Avx512f"; }
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetDeconvolution32fGemmNN : public Base::SynetDeconvolution32fGemmNN
        {
        public:
            SynetDeconvolution32fGemmNN(const DeconvParam32f & p);
            virtual String Ext() const { return "Neon"; }
        };

        class SynetDeconvolution32fNhwcDirect2x2 : public Base::SynetDeconvolution32fNhwcDirect2x2
        {
        public:
            SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p);
            virtual String Ext() const { return "Neon"; }

            static bool Preferable(const DeconvParam32f & p);
        };

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimdSynetDeconvolution32f_h__
