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
#ifndef __SimdSynetConvParam_h__
#define __SimdSynetConvParam_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
    SIMD_INLINE bool IsKernel(const SimdConvolutionParameters& p, size_t value)
    {
        return p.kernelY == value && p.kernelX == value;
    }

    SIMD_INLINE bool IsDilation(const SimdConvolutionParameters& p, size_t value)
    {
        return p.dilationY == value && p.dilationX == value;
    }

    SIMD_INLINE bool IsStride(const SimdConvolutionParameters& p, size_t value)
    {
        return p.strideY == value && p.strideX == value;
    }

    SIMD_INLINE bool IsPad(const SimdConvolutionParameters& p, size_t value)
    {
        return p.padY == value && p.padX == value && p.padH == value && p.padW == value;
    }

    SIMD_INLINE bool IsDepthwise(const SimdConvolutionParameters& p)
    {
        return p.srcC == p.group && p.dstC == p.group;
    }

    SIMD_INLINE bool Is1x1(const SimdConvolutionParameters& p)
    {
        return IsKernel(p, 1) && IsDilation(p, 1) && IsStride(p, 1) && IsPad(p, 0);
    }

    SIMD_INLINE size_t NoseH(const SimdConvolutionParameters& p)
    {
        return DivHi(p.padY, p.strideY);
    }

    SIMD_INLINE size_t NoseW(const SimdConvolutionParameters& p)
    {
        return DivHi(p.padX, p.strideX);
    }

    SIMD_INLINE size_t BodyH(const SimdConvolutionParameters& p)
    {
        return (p.padY + p.srcH - (p.kernelY - 1) * p.dilationY - 1) / p.strideY + 1;
    }

    SIMD_INLINE size_t BodyW(const SimdConvolutionParameters& p)
    {
        return (p.padX + p.srcW - (p.kernelX - 1) * p.dilationX - 1) / p.strideX + 1;
    }

    //-------------------------------------------------------------------------------------------------

    struct ConvParam : public SimdConvolutionParameters
    {
        SimdBool trans;
        size_t batch;
        SimdSynetCompatibilityType compatibility;

        ConvParam()
            : SimdConvolutionParameters({ 0 })
            , trans(SimdFalse)
            , batch(0)
            , compatibility(SimdSynetCompatibilityDefault)
        {
        }

        ConvParam(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            *((SimdConvolutionParameters*)this) = *conv;
            this->trans = (srcF == SimdTensorFormatNhwc ? SimdTrue : SimdFalse);
            this->batch = batch;
            this->compatibility = compatibility;
        }

        bool Valid(SimdTensorDataType type0, SimdTensorDataType type1 = SimdTensorData32f)
        {
            return
                dstH == (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1 && dstH > 0 &&
                dstW == (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1 && dstW > 0 &&
                srcF == dstF && (srcF == SimdTensorFormatNchw || srcF == SimdTensorFormatNhwc) &&
                (srcT == type0 || srcT == type1) && (dstT == type0 || dstT == type1);
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
            return Simd::IsDepthwise(*this);
        }

        SIMD_INLINE bool Is1x1() const
        {
            return Simd::Is1x1(*this);
        }

        SIMD_INLINE size_t NoseH() const
        {
            return Simd::NoseH(*this);
        }

        SIMD_INLINE size_t NoseW() const
        {
            return Simd::NoseW(*this);
        }

        SIMD_INLINE size_t BodyH() const
        {
            return Simd::BodyH(*this);
        }

        SIMD_INLINE size_t BodyW() const
        {
            return Simd::BodyW(*this);
        }

        SIMD_INLINE size_t SizeS() const
        {
            return batch * srcC * srcH * srcW;
        }

        SIMD_INLINE size_t SizeW() const
        {
            return kernelY * kernelX * srcC * dstC / group;
        }

        SIMD_INLINE size_t SizeD() const
        {
            return batch * dstC * dstH * dstW;
        }

        SIMD_INLINE String Info() const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << Simd::Max(dilationX, dilationY) << "-" << Simd::Max(strideX, strideY);
            ss << "-" << group << "-" << trans;
            return ss.str();
        }

        SIMD_INLINE int64_t Flop() const
        {
            return int64_t(batch) * kernelY * kernelX * srcC * dstH * dstW * dstC / group * 2;
        }
    };
}

#endif
