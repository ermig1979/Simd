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
#ifndef __SimdSynetMergedConvolution32f_h__
#define __SimdSynetMergedConvolution32f_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct MergConvParam32f
    {
        SimdBool trans, add;
        size_t batch, count;
        SimdConvolutionParameters conv[3];

        MergConvParam32f(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            assert(count <= 3);
            this->add = add;
            this->batch = batch;
            this->count = count;
            for (size_t i = 0; i < count; ++i)
                this->conv[i] = convs[i];
        }

        bool Valid()
        {
            if (count < 2 || count > 3)
                return false;
            for (size_t i = 0; i < count; ++i)
            {
                SimdConvolutionParameters & c = conv[i];                
                if (c.srcT != SimdTensorData32f || c.dstT != SimdTensorData32f)
                    return false;
                if (c.srcF != SimdTensorFormatNhwc || c.dstF != SimdTensorFormatNhwc)
                    return false;
                if (c.dstH != (c.srcH + c.padY + c.padH - (c.dilationY * (c.kernelY - 1) + 1)) / c.strideY + 1 || c.dstH == 0)
                    return false;
                if (c.dstW != (c.srcW + c.padX + c.padW - (c.dilationY * (c.kernelX - 1) + 1)) / c.strideX + 1 || c.dstW == 0)
                    return false;
                if (c.kernelY != c.kernelX || !(c.kernelY == 1 || c.kernelY == 3 || c.kernelY == 5 || c.kernelY == 7))
                    return false;
                if (c.strideY != c.strideX || !(c.strideY == 1 || c.strideY == 2 || c.strideY == 3))
                    return false;
                if (c.dilationY != 1 || c.dilationX != 1)
                    return false;

                if (c.dstH == (c.srcH + c.padY + c.padH - (c.dilationY * (c.kernelY - 1) + 1) - 1) / c.strideY + 1)
                    c.padH--;
                if (c.dstW == (c.srcW + c.padX + c.padW - (c.dilationY * (c.kernelX - 1) + 1) - 1) / c.strideX + 1)
                    c.padW--;
            }
            if (count == 3)
            {
                if (conv[0].group != 1 || (conv[0].kernelY != 1 && conv[0].kernelY != 3))
                    return false;
                if (conv[1].group != conv[1].srcC || conv[1].group != conv[1].dstC || (conv[1].kernelY != 3 && conv[1].kernelY != 5 && conv[1].kernelY != 7))
                    return false;
                if (conv[2].group != 1 || conv[2].kernelY != 1 || conv[2].strideY != 1)
                    return false;
                if (add && (conv[0].srcC != conv[2].dstC || conv[0].srcH != conv[2].dstH || conv[0].srcW != conv[2].dstW))
                    return false;
            }
            else
            {
                if (conv[0].group == 1)
                {
                    if (conv[0].kernelY != 1 && conv[0].kernelY != 3)
                        return false;
                    if (conv[1].group != conv[1].srcC || conv[1].group != conv[1].dstC || (conv[1].kernelY != 3 && conv[1].kernelY != 5 && conv[1].kernelY != 7))
                        return false;
                }
                else
                {
                    if (conv[0].group != conv[0].srcC || conv[0].group != conv[0].dstC || (conv[0].kernelY != 3 && conv[0].kernelY != 5 && conv[0].kernelY != 7))
                        return false;
                    if (conv[1].group != 1 || conv[1].kernelY != 1 || conv[1].strideY != 1)
                        return false;
                }
            }
            return true;
        }

        SIMD_INLINE bool IsPad(size_t index, size_t value) const
        {
            return conv[index].padY == value && conv[index].padX == value && conv[index].padH == value && conv[index].padW == value;
        }

#ifdef SIMD_PERFORMANCE_STATISTIC
        String Info() const
        {
            std::stringstream ss;
            ss << count << ":" << batch << "x" << conv[0].srcC << "x" << conv[0].srcH << "x" << conv[0].srcW;
            for (size_t i = 0; i < count; ++i)
                ss << "-" << (conv[i].group != 1 ? String("") : ToStr(conv[i].dstC) + "x") << conv[i].kernelY << "x" << conv[i].strideY;
            return ss.str();
        }

        int64_t Flop(size_t i) const
        {
            return int64_t(batch) * conv[i].kernelY * conv[i].kernelX * conv[i].srcC * conv[i].dstH * conv[i].dstW * conv[i].dstC / conv[i].group * 2;
        }

        int64_t Flop() const
        {
            int64_t flop = 0;
            for (size_t i = 0; i < count; ++i)
                flop += Flop(i);
            return flop;
        }
#endif
    };

    class SynetMergedConvolution32f : public Deletable
    {
    public:
        virtual const MergConvParam32f & Param() const = 0;

        virtual size_t ExternalBufferSize() const = 0;

        virtual size_t InternalBufferSize() const = 0;

        virtual void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params) = 0;

        virtual void Forward(const float * src, float * buf, float * dst) = 0;

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        virtual Base::PerformanceMeasurer* Perf(const char* func) = 0;
#endif

        virtual const char* Info() const = 0;
    };

    namespace Base
    {
        class SynetMergedConvolution32f : public Simd::SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32f(const MergConvParam32f& p);

            virtual String Desc() const { return Ext(); }
            virtual String Ext() const { return "Base"; }
            virtual const MergConvParam32f & Param() const { return _param; }
            virtual size_t ExternalBufferSize() const;
            virtual size_t InternalBufferSize() const;
            virtual void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params);
            virtual void Forward(const float* src, float* buf, float* dst);

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            virtual Base::PerformanceMeasurer* Perf(const char* func);
#endif

            virtual const char* Info() const
            {
                _info = Desc();
                return _info.c_str();
            }

            typedef void(*ConvolutionPtr)(const float* src, const SimdConvolutionParameters& p, size_t maC, size_t yBeg, size_t yEnd,
                const size_t * bufH, const float* weight, const float* bias, const float* params, float* dst, int first);

        protected:
            float* GetBuffer(float* buffer);

            virtual void ReorderFirstWeight(const float* src, float* dst) const {}
            virtual void ReorderSecondWeight(const float* src, float* dst) const {}
            virtual void ReorderThirdWeight(const float* src, float* dst) const {}

            MergConvParam32f _param;
            ConvolutionPtr _convolution[4];
            size_t _sizeS, _sizeD, _sizeB[2];
            Array32f _buffer, _rWeight[3], _rBias[3], _rParams[3];
            const float * _weight[3], * _bias[3], * _params[3];

            size_t _miC, _maC, _yStep[2], _bufH[2], _dp[2], _dw[3];

        private:
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            Base::PerformanceMeasurer * _perf;
#endif        
            mutable String _info;
        };

        class SynetMergedConvolution32fCdc : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);

            virtual void Forward(const float * src, float * buf, float * dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
            virtual void ReorderThirdWeight(const float* src, float* dst) const;
        };

        class SynetMergedConvolution32fCd : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
        };

        class SynetMergedConvolution32fDc : public SynetMergedConvolution32f
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);

            virtual void Forward(const float* src, float* buf, float* dst);

            static bool Preferable(const MergConvParam32f& p);

        protected:
            void SetSize(size_t L1, size_t L2, size_t L3, size_t F);
            virtual void ReorderFirstWeight(const float* src, float* dst) const;
            virtual void ReorderSecondWeight(const float* src, float* dst) const;
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class SynetMergedConvolution32fCdc : public Base::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);
            virtual String Ext() const { return "Sse2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr * c);
        };

        class SynetMergedConvolution32fCd : public Base::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);
            virtual String Ext() const { return "Sse2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fDc : public Base::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);
            virtual String Ext() const { return "Sse2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class SynetMergedConvolution32fCdc : public Sse2::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);
            virtual String Ext() const { return "Avx"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Sse2::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fDc : public Sse2::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetMergedConvolution32fCdc : public Avx::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);
            virtual String Ext() const { return "Avx2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Avx::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fDc : public Avx::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx2"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class SynetMergedConvolution32fCdc : public Avx2::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);
            virtual String Ext() const { return "Avx512f"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Avx2::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx512f"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fDc : public Avx2::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);
            virtual String Ext() const { return "Avx512f"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetMergedConvolution32fCdc : public Base::SynetMergedConvolution32fCdc
        {
        public:
            SynetMergedConvolution32fCdc(const MergConvParam32f & p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        class SynetMergedConvolution32fCd : public Base::SynetMergedConvolution32fCd
        {
        public:
            SynetMergedConvolution32fCd(const MergConvParam32f& p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };
        
        class SynetMergedConvolution32fDc : public Base::SynetMergedConvolution32fDc
        {
        public:
            SynetMergedConvolution32fDc(const MergConvParam32f& p);
            virtual String Ext() const { return "Neon"; }

            static void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c);
        };

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetMergedConvolution32f_h__
