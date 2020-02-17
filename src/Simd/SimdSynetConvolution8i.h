/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdSynetConvolution8i_h__
#define __SimdSynetConvolution8i_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    struct ConvParam8i : public SimdConvolutionParameters
    {
        SimdBool trans;
        size_t batch;

        ConvParam8i(size_t batch, const SimdConvolutionParameters * conv)
        {
            *((SimdConvolutionParameters*)this) = *conv;
            this->trans = (srcF == SimdTensorFormatNhwc ? SimdTrue : SimdFalse);
            this->batch = batch;
        }

        bool Valid()
        {
            return 
                dstH == (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1 && dstH > 0 &&
                dstW == (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1 && dstW > 0 &&
                (srcT == SimdTensorData32f || srcT == SimdTensorData8u) && (dstT == SimdTensorData32f || dstT == SimdTensorData8u) &&
                srcF == dstF && (srcF == SimdTensorFormatNchw || srcF == SimdTensorFormatNhwc);
        }

        SIMD_INLINE bool IsKernel(size_t value) const
        {
            return kernelY == value && kernelX == value;
        }

        SIMD_INLINE bool IsKernel(size_t valueY, size_t valueX) const
        {
            return kernelY == valueY && kernelX == valueX;
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

        long long Flop() const
        {
            return batch* kernelY* kernelX* srcC* dstH* dstW* dstC / group * 2;
        }
#endif
    };

    struct CvtParam
    {
        Array8u zero;
        Array32f scale, shift, iScale, iShift;
        bool neg;

        CvtParam()
            : neg(false)
        {
        }

        void Init(const float * min, const float * max, size_t size)
        {
            zero.Resize(size);
            scale.Resize(size);
            shift.Resize(size);
            iScale.Resize(size);
            iShift.Resize(size);
            for (size_t i = 0; i < size; ++i)
            {
                assert(min[i] <= max[i]);
                if (min[i] < 0.0f)
                    neg = true;
            }
            for (size_t i = 0; i < size; ++i)
            {
                float abs = ::fmax(::fabs(min[i]), ::fabs(max[i]));
                float inv = abs / (neg ? 127.0f : 255.0f);
                if (fabs(inv) < 1e-7)
                    inv = 1.0f;
                zero[i] = (neg ? 128 : 0);
                scale[i] = float(1.0 / inv);
                shift[i] = float(zero[i]);
                iScale[i] = inv;
                iShift[i] = -float(zero[i]) * inv;
            }        
        }

        size_t Size() const
        {
            return (zero.size)*sizeof(uint8_t) + (scale.size + shift.size + iScale.size + iShift.size) * sizeof(float);
        }
    };

    class SynetConvolution8i : public Deletable
    {
    public:
        SynetConvolution8i(const ConvParam8i & p) 
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC)
            , _perf(NULL)
#endif
        {
        }

        const ConvParam8i & Param() const 
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
            return _buffer.size;
        }

        virtual void SetParams(const float* weight, const float* bias, const float* params, const float* const* stats) = 0;

        virtual void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst) = 0;

        uint8_t* Buffer(uint8_t * buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        template<class T> T * Allocate(uint8_t* & buffer, size_t size)
        {
            T* ptr = (T*)buffer;
            buffer = buffer + size*sizeof(T);
            return ptr;
        }

#if defined(SIMD_PERFORMANCE_STATISTIC)
        Base::PerformanceMeasurer* Perf(const String & func);
#endif

    protected:
        ConvParam8i _param;
        Array8u _buffer;
#if defined(SIMD_PERFORMANCE_STATISTIC)
        Base::PerformanceMeasurer * _perf;
#endif
    };

    namespace Base
    {
        class SynetConvolution8iGemmNN : public SynetConvolution8i
        {
        public:
            SynetConvolution8iGemmNN(const ConvParam8i & p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::GemmNN"; }
            virtual size_t InternalBufferSize() const;
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params, const float* const* stats);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

        protected:
            virtual void ImgToCol(const uint8_t* src, uint8_t* dst);
            virtual void ImgToRow(const uint8_t* src, uint8_t* dst);

            virtual void GemmNN(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t lda, const int8_t* weight, size_t ldb, int32_t* dst, size_t ldc);
            virtual void GemmNN(size_t D, size_t S, size_t C, size_t K, const int8_t* weight, size_t lda, const uint8_t* src, size_t ldb, int32_t* dst, size_t ldc);

            CvtParam _srcCvt, _dstCvt;
            Array8i _weight8i;
            Array32i _norm32i;
            Array32f _norm32f; 
            bool _skipConv, _src8u, _dst8u, _overflow16i;
            size_t _batch, _merge, _ldW, _ldS, _ldD, _grW, _grS, _grD, _siC, _siK, _siS, _siD, _sizeS, _sizeB, _sizeD;
        };

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv);
    }
}

#endif//__SimdSynetConvolution8i_h__
