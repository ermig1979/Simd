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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template<SimdConvolutionActivationType type, UpdateType update> void DirectConvolution(const float* src, const SimdConvolutionParameters& p,
            size_t maC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    if (bias)
                        memcpy(buf.data, bias, dstC * sizeof(float));
                    else
                        memset(buf.data, 0, dstC * sizeof(float));
                    for (size_t ky = 0; ky < kernelY; ++ky)
                    {
                        size_t sy = dy * strideY + ky - padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; ++kx)
                            {
                                size_t sx = dx * strideX + kx - padX;
                                if (sx < p.srcW)
                                {
                                    const float* pw = weight + (ky * kernelX + kx) * srcC * dstC;
                                    const float* ps = src + (sy * srcW + sx) * srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += dstC;
                                    }
                                }
                            }
                        }
                    }
                    if (update == UpdateAdd)
                    {
                        for (size_t dc = 0; dc < dstC; ++dc)
                            dst[dc] = Activate<type>(dst[dc] + buf[dc], params, dc);
                    }
                    else
                    {
                        for (size_t dc = 0; dc < dstC; ++dc)
                            dst[dc] = Activate<type>(buf[dc], params, dc);
                    }
                    dst += p.dstC;
                }
            }
        }

        template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t index, SynetMergedConvolution32fCdc::ConvolutionPtr * convolution)
        {
            switch (index)
            {
            case 0:
                if(p.conv[0].group == 1)
                    convolution[0] = DirectConvolution<type, UpdateSet>;
                else
                    convolution[0] = DepthwiseConvolution<type>;
                break;
            case 1:
                if (p.conv[1].group == 1)
                    convolution[1] = DirectConvolution<type, UpdateSet>;
                else
                    convolution[1] = DepthwiseConvolution<type>;
                break;
            case 2:
                if (p.add)
                    convolution[2] = DirectConvolution<type, UpdateAdd>;
                else
                    convolution[2] = DirectConvolution<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fBf16::SynetMergedConvolution32fBf16(const MergConvParam32f& p)
           : Simd::SynetMergedConvolution32f(p)
        {
            _convert = NULL, _input = NULL, _depthwise = NULL, _output[0] = NULL, _output[1] = NULL;
            _sizeB[0] = 0, _sizeB[1] = 0, _sizeB[2] = 0;
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            size_t size0 = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            size_t size1 = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            _dw0 = beg.group != 1;
            _1x1 = beg.kernelY == 1 && beg.strideY == 1;
        }

        size_t SynetMergedConvolution32fBf16::ExternalBufferSize() const
        {
            return _sizeB[1] + (_sizeB[0] + _sizeB[2]) / 2;
        }

        size_t SynetMergedConvolution32fBf16::InternalBufferSize() const
        {
            size_t size = _buffer.size + _weightD.size;
            size += (_weightI.size + _weightO.size) / 2;
            for (size_t i = 0; i < _param.count; ++i)
                size += _bias[i].size + _params[i].size;
            return size;
        }

        void SynetMergedConvolution32fBf16::SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params)
        {
            const MergConvParam32f& p = _param;
            if (_dw0)
            {
                SetDepthwiseWeight(weight[0], p.conv[0]);
                SetOutputWeight(weight[1], p.conv[1]);
            }
            else
            {
                SetInputWeight(weight[0], p.conv[0]);
                SetDepthwiseWeight(weight[1], p.conv[1]);
                if(p.count > 2)
                    SetOutputWeight(weight[2], p.conv[2]);
            }
            for (size_t i = 0; i < p.count; ++i)
            {
                if (internal)
                    internal[i] = SimdTrue;
                SetBias(bias[i], p.conv[i], _bias[i]);
                SetParams(params[i], p.conv[i], _params[i]);
            }
        }

        void SynetMergedConvolution32fBf16::SetInputWeight(const float* src, const SimdConvolutionParameters& p)
        {

        }

        void SynetMergedConvolution32fBf16::SetDepthwiseWeight(const float* src, const SimdConvolutionParameters& p)
        {

        }

        void SynetMergedConvolution32fBf16::SetOutputWeight(const float* src, const SimdConvolutionParameters& p)
        {

        }

        void SynetMergedConvolution32fBf16::SetBias(const float* src, const SimdConvolutionParameters& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            dst.Resize(AlignHiAny(p.dstC, a.miC), true);
            if (src)
                memcpy(dst.data, src, p.dstC * sizeof(float));
        }

        void SynetMergedConvolution32fBf16::SetParams(const float* src, const SimdConvolutionParameters& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                dst.Resize(AlignHiAny(p.dstC, a.miC), true);
            else
                dst.Resize(2, true);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                dst.data[0] = -FLT_MAX;
                dst.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationRelu:
                dst.data[0] = 0;
                dst.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationLeakyRelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    dst.data[d] = src[0];
                break;
            case SimdConvolutionActivationRestrictRange:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationPrelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    dst.data[d] = src[d];
                break;
            case SimdConvolutionActivationElu:
                dst.data[0] = src[0];
                break;
            case SimdConvolutionActivationHswish:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationMish:
                dst.data[0] = src[0];
                break;
            case SimdConvolutionActivationHardSigmoid:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationSwish:
                dst.data[0] = src[0];
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
