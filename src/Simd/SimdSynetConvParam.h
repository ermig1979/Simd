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

    SIMD_INLINE String ToStr(SimdConvolutionActivationType t)
    {
        const char* cats[] = { "id", "re", "lr", "rr", "pr", "el", "hs", "mi", "hi", "sw", "ge" };
        return String(cats[t]);
    }

    SIMD_INLINE String ToChar(SimdTensorDataType t)
    {
        const char* tdts[] = { "?", "f", "i", "u", "u", "l", "l", "~", "b", "b"};
        return String(tdts[int(t) + 1]);
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

        SIMD_INLINE String Info(bool detail = false) const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << Simd::Max(dilationX, dilationY) << "-" << Simd::Max(strideX, strideY);
            ss << "-" << group << "-" << trans;
            if (detail)
            {
                ss << "-" << ToChar(srcT) << ToChar(dstT) << "-" << ToStr(activation);
            }
            return ss.str();
        }

        SIMD_INLINE int64_t Flop() const
        {
            return int64_t(batch) * kernelY * kernelX * srcC * dstH * dstW * dstC / group * 2;
        }
    };

    //-------------------------------------------------------------------------------------------------

    struct DeconvParam : public SimdConvolutionParameters
    {
        SimdBool trans;
        size_t batch;
        SimdSynetCompatibilityType compatibility;

        DeconvParam(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            *((SimdConvolutionParameters*)this) = *conv;
            this->trans = (srcF == SimdTensorFormatNhwc ? SimdTrue : SimdFalse);
            this->batch = batch;
            this->compatibility = compatibility;
        }

        bool Valid(SimdTensorDataType type0, SimdTensorDataType type1 = SimdTensorData32f)
        {
            return
                dstH == strideY * (srcH - 1) + dilationY * (kernelY - 1) + 1 - padY - padH && dstH > 0 &&
                dstW == strideX * (srcW - 1) + dilationX * (kernelX - 1) + 1 - padX - padW && dstW > 0 &&
                srcF == dstF && (srcF == SimdTensorFormatNchw || (srcF == SimdTensorFormatNhwc && group == 1)) &&
                (srcT == type0 || srcT == type1) && (dstT == type0 || dstT == type1);
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
            return Simd::IsDepthwise(*this);
        }
        SIMD_INLINE bool Is1x1() const
        {
            return Simd::Is1x1(*this);
        }

        String Info(bool detail = false) const
        {
            std::stringstream ss;
            ss << batch << "x" << srcC << "x" << srcH << "x" << srcW;
            ss << "-" << dstC << "x" << kernelY << "x" << kernelX;
            ss << "-" << Simd::Max(dilationX, dilationY) << "-" << Simd::Max(strideX, strideY);
            ss << "-" << group << "-" << trans;
            if (detail)
            {
                ss << "-" << ToChar(srcT) << ToChar(dstT) << "-" << ToStr(activation);
            }
            return ss.str();
        }

        int64_t Flop() const
        {
            return int64_t(batch) * kernelY * kernelX * srcC * srcH * srcW * dstC / group * 2;
        }
    };

    //-------------------------------------------------------------------------------------------------

    struct MergConvParam
    {
        ConvParam conv[3];
        size_t count;
        SimdBool add;

        MergConvParam(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault)
        {
            assert(count <= 3);
            this->count = count;
            for (size_t i = 0; i < count; ++i)
                this->conv[i] = ConvParam(batch, convs + i, compatibility);
            this->add = add;
        }

        bool Valid(SimdTensorDataType type0, SimdTensorDataType type1 = SimdTensorData32f)
        {
            if (count < 2 || count > 3)
                return false;
            for (size_t i = 0; i < count; ++i)
            {
                ConvParam& c = conv[i];
                if (!c.Valid(type0, type1))
                    return false;
                if (c.srcF != SimdTensorFormatNhwc || c.dstF != SimdTensorFormatNhwc)
                    return false;
                if (c.dstH != (c.srcH + c.padY + c.padH - (c.dilationY * (c.kernelY - 1) + 1)) / c.strideY + 1 || c.dstH == 0)
                    return false;
                if (c.dstW != (c.srcW + c.padX + c.padW - (c.dilationY * (c.kernelX - 1) + 1)) / c.strideX + 1 || c.dstW == 0)
                    return false;
                if (c.kernelY != c.kernelX || !(c.kernelY == 1 || c.kernelY == 3 || c.kernelY == 5 || c.kernelY == 7))
                    return false;
                if (/*c.strideY != c.strideX ||*/ !(c.strideY == 1 || c.strideY == 2 || c.strideY == 3))
                    return false;
                if (c.dilationY != 1 || c.dilationX != 1)
                    return false;

                if (c.dstH == (c.srcH + c.padY + c.padH - (c.dilationY * (c.kernelY - 1) + 1) - 1) / c.strideY + 1)
                    c.padH--;
                if (c.dstW == (c.srcW + c.padX + c.padW - (c.dilationY * (c.kernelX - 1) + 1) - 1) / c.strideX + 1)
                    c.padW--;
                if (c.IsDepthwise() && i != count - 1 && type1 != SimdTensorData32f)
                    c.dstT = type1;
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

        SIMD_INLINE String Info(bool detail = false) const
        {
            std::stringstream ss;
            ss << count << ":" << conv[0].batch << "x" << conv[0].srcC << "x" << conv[0].srcH << "x" << conv[0].srcW;
            for (size_t i = 0; i < count; ++i)
                ss << "-" << (conv[i].group != 1 ? String("") : ToStr(conv[i].dstC) + "x") << conv[i].kernelY << "x" << conv[i].strideY;
            if (detail)
            {
                ss << "-" << ToChar(conv[0].srcT) << ToChar(conv[count - 1].dstT);
                if (count == 3)
                    ss << add;
            }
            return ss.str();
        }

        SIMD_INLINE int64_t Flop() const
        {
            int64_t flop = 0;
            for (size_t i = 0; i < count; ++i)
                flop += conv[i].Flop();
            return flop;
        }
    };
}

#endif
