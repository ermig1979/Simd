/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdMergedConvolution.h"
#include "Simd/SimdConvolution.h"
#include "Simd/SimdConvolutionCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        template<SimdConvolutionActivationType type, UpdateType update> void DirectConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t maC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
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
                                    const float * pw = weight + (ky*kernelX + kx)*srcC*dstC;
                                    const float * ps = src + (sy*srcW + sx)*srcC;
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
                    for (size_t dc = 0; dc < dstC; ++dc)
                        Update<update>(dst + dc, Activate<type>(buf[dc], params, dc));
                    dst += p.dstC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t maC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float * pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float * ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum, params, c);
                    }
                    dst += srcC;
                }
            }
        }

        template <SimdConvolutionActivationType type> void SetConvolutionPtr(const MergConvParam & p, size_t index, MergedConvolution::ConvolutionPtr convolution[3])
        {
            switch (index)
            {
            case 0: 
                convolution[0] = DirectConvolution<type, UpdateSet>;
                break;
            case 1:
                convolution[1] = DepthwiseConvolution<type>;
                break;
            case 2:
                if(p.add)
                    convolution[2] = DirectConvolution<type, UpdateAdd>;
                else
                    convolution[2] = DirectConvolution<type, UpdateSet>;
                break;
            default: 
                assert(0);
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : _param(p), _base(true)
        {
            _sizeS = p.conv[0].srcH*p.conv[0].srcW*p.conv[0].srcC;
            _sizeD = p.conv[2].dstH*p.conv[2].dstW*p.conv[2].dstC;
            _sizeB[0] = p.conv[1].srcH*p.conv[1].srcW*p.conv[1].srcC;
            _sizeB[1] = p.conv[1].dstH*p.conv[1].dstW*p.conv[1].dstC;
            for (size_t i = 0; i < _param.count; ++i)
            {
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: SetConvolutionPtr<SimdConvolutionActivationIdentity>(_param, i, _convolution); break;
                case SimdConvolutionActivationRelu: SetConvolutionPtr<SimdConvolutionActivationRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: SetConvolutionPtr<SimdConvolutionActivationLeakyRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationRestrictRange: SetConvolutionPtr<SimdConvolutionActivationRestrictRange>(_param, i, _convolution); break;
                case SimdConvolutionActivationPrelu: SetConvolutionPtr<SimdConvolutionActivationPrelu>(_param, i, _convolution); break;
                default: assert(0);
                }
            }       
        }

        void MergedConvolution::SetSize(size_t L1, size_t L2, size_t L3, size_t F)
        {
            const MergConvParam & p = _param;
            _miC = F;
            _maC = p.conv[0].dstC;
            for (size_t yStep = p.conv[1].dstH; yStep >= 1; yStep--)
            {
                _yStep[1] = Simd::Max<size_t>(1, yStep);
                for (_bufH[1] = 1; _bufH[1] < _yStep[1]; _bufH[1] *= 2);
                _yStep[0] = _yStep[1] * p.conv[1].strideY;
                for (_bufH[0] = 1; _bufH[0] < (_yStep[1] - 1) * p.conv[1].strideY + p.conv[1].kernelY; _bufH[0] *= 2);
                _sizeB[0] = _bufH[0] * p.conv[0].dstW * AlignHi(p.conv[0].dstC, F);
                _sizeB[1] = _bufH[1] * p.conv[1].dstW * AlignHi(p.conv[1].dstC, F);
                if ((_sizeB[0] + _sizeB[1]) * sizeof(float) <= L2)
                    break;
            }
            _rWeight[0].Resize(AlignHiAny(p.conv[0].dstC, 2 * _miC)*p.conv[0].kernelY*p.conv[0].kernelX*p.conv[0].srcC);
            _rWeight[1].Resize(AlignHiAny(p.conv[1].dstC, _miC)*p.conv[1].kernelY*p.conv[1].kernelX);
            _rWeight[2].Resize(AlignHiAny(p.conv[2].dstC, 2 * _miC)*p.conv[2].kernelY*p.conv[2].kernelX*p.conv[2].srcC);
            _base = false;
        }

        float * MergedConvolution::GetBuffer(float * buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        void MergedConvolution::ReorderInputWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[0];
            size_t size = p.kernelY*p.kernelX*p.srcC, dstC = p.dstC, micD = _miC*2;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void MergedConvolution::ReorderDepthwiseWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[1];
            size_t dstC = p.dstC, size = p.kernelY*p.kernelX, micD = _miC;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void MergedConvolution::ReorderOutputWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[2];
            size_t size = p.kernelY*p.kernelX*p.srcC, dstC = p.dstC, micD = _miC * 2;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        size_t MergedConvolution::ExternalBufferSize() const
        {
            return _sizeB[0] + _sizeB[1];
        }

        size_t MergedConvolution::InternalBufferSize() const
        {
            return _buffer.size + _rWeight[0].size + _rWeight[1].size + _rWeight[2].size;
        }

        void MergedConvolution::SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params)
        {
            for (size_t i = 0; i < _param.count; ++i)
            {
                if (_rWeight[i].data)
                {
                    switch (i)
                    {
                    case 0: ReorderInputWeight(weight[i], _rWeight[i].data); break;
                    case 1: ReorderDepthwiseWeight(weight[i], _rWeight[i].data); break;
                    case 2: ReorderOutputWeight(weight[i], _rWeight[i].data); break;
                    default: assert(0);
                    }
                    _weight[i] = _rWeight[i].data;
                    if (internal)
                        internal[i] = SimdTrue;
                }
                else
                {
                    _weight[i] = weight[i];
                    if (internal)
                        internal[i] = SimdFalse;
                }
                _bias[i] = bias[i];
                _params[i] = params[i];
            }
        }

        void MergedConvolution::Forward(const float * src, float * buf, float * dst)
        {
            const MergConvParam & p = _param;
            float * buf0 = GetBuffer(buf);
            float * buf1 = buf0 + _sizeB[0];
            for (size_t b = 0; b < p.batch; ++b)
            {
                if (_base)
                {
                    _convolution[0](src, p.conv[0], 0, 0, p.conv[0].dstH, _bufH, _weight[0], _bias[0], _params[0], buf0);
                    _convolution[1](buf0, p.conv[1], 0, 0, p.conv[1].dstH, _bufH, _weight[1], _bias[1], _params[1], buf1);
                    _convolution[2](buf1, p.conv[2], 0, 0, p.conv[2].dstH, _bufH, _weight[2], _bias[2], _params[2], dst);
                }
                else
                {
                    for (size_t yBeg1 = 0, yBeg0 = 0; yBeg1 < p.conv[1].dstH;)
                    {
                        size_t yEnd1 = Simd::Min(yBeg1 + _yStep[1], p.conv[1].dstH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + _yStep[0], (_yStep[1] - 1)*p.conv[1].strideY + p.conv[1].kernelY - p.conv[1].padY, p.conv[0].dstH);
                        _convolution[0](src, p.conv[0], _maC, yBeg0, yEnd0, _bufH, _weight[0], _bias[0], _params[0], buf0);
                        _convolution[1](buf0, p.conv[1], _maC, yBeg1, yEnd1, _bufH, _weight[1], _bias[1], _params[1], buf1);
                        _convolution[2](buf1, p.conv[2], _maC, yBeg1, yEnd1, _bufH, _weight[2], _bias[2], _params[2], dst);
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new Base::MergedConvolution(param);
        }
    }
}
