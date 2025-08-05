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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetQuantizedConvolutionNhwcDepthwiseV0::SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _convolution = 0;
            AlgParam& a = _alg;
            a.srcE = _elemS;
            a.dstE = _elemD;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV0";
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV0::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            for (size_t b = 0; b < p.batch; b += _merge)
            {
                _convolution(src, _srcZero[0], p, a, _weight.data, _bias.data, _norm.data, _dstZero[0], dst);
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV0::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            _weight.Assign(weight, _weight.size);
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV0::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.group >= F;
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV1::SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _convolution = 0;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV1::Desc() const
        {
            const AlgParam& a = _alg;
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV1-" << a.reorderType;
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV1::InternalBufferSize() const
        {
            return SynetQuantizedConvolution::InternalBufferSize() + _weight32i.RawSize();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufC * a.bufH * a.bufW * sizeof(int32_t);
            return size;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            int32_t* buf = Allocate<int32_t>(buf8, a.bufC * a.bufH * a.bufW);
            for (size_t b = 0; b < p.batch; b += 1)
            {
                for (size_t yBeg = 0; yBeg < p.dstH;)
                {
                    size_t yEnd = Simd::Min(yBeg + a.stepH, p.dstH);
                    _preprocess(src, _srcZero[0], p, a, yBeg, yEnd, buf);
                    _convolution(buf, p, a, _weight32i.data, _bias.data, _norm.data, yBeg, yEnd, _dstZero[0], dst);
                    yBeg = yEnd;
                }
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::SetAlgParam(size_t F)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            a.srcE = _elemS;
            a.dstE = _elemD;
            a.F = F;
            a.bufC = AlignHi(p.srcC, F);
            a.bufW = p.srcW + p.padX + p.padW;
            a.bufH = Pow2Hi(p.kernelY);
            a.stepH = 1;
            a.reorderType = 0;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::SetWeight(const int8_t* src)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t kernel = p.kernelY * p.kernelX;
            _weight32i.Resize(kernel * a.bufC);
            int32_t* dst = _weight32i.data;
            if (a.reorderType == 0)
            {
                for (size_t k = 0; k < kernel; ++k)
                {
                    size_t c = 0;
                    for (; c < p.srcC; ++c)
                        dst[c] = src[c];
                    for (; c < a.bufC; ++c)
                        dst[c] = 0;
                    src += p.srcC;
                    dst += a.bufC;
                }
            }
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV1::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.IsDilation(1) && p.group >= F;
        }
     }
#endif
}
