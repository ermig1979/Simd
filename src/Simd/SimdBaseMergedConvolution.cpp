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
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float * params, size_t offset);

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<SimdConvolutionActivationType type, UpdateType update> void DirectConvolutionOld(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
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

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionOld(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
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

        const size_t F = 1;

        template<SimdConvolutionActivationType type> void InputConvolution(const float * src, const SimdConvolutionParameters & p, 
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == 1);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            size_t dC = (dstC + F - 1)/F, dstM = (bufH[0] - 1), dstS = bufH[0]*dstW*F;
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
                    float * pDst = dst + ((dy&dstM)*dstW + dx)*F;
                    for (size_t dc = 0; dc < dstC; ++dc)
                        pDst[dstS * dc] = Activate<type>(buf[dc], params, dc);
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            size_t sC = (srcC + F - 1) / F, srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW*F, dstS = bufH[1] * dstW*F;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    float * pDst = dst + ((dy&dstM)*dstW + dx)*F;
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const float * pSrc = src + (sy&srcM)*srcW*F + c*srcS;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float * pw = weight + (ky * kernelX + kx) * srcC + c;
                                        sum += pSrc[sx] * pw[0];
                                    }
                                }
                            }
                        }
                        pDst[dstS * c] = Activate<type>(sum, params, c);
                    }
                }
            }
        }

        template<SimdConvolutionActivationType type, UpdateType update> void OutputConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t sC = (srcC + F - 1) / F, srcM = (bufH[1] - 1), srcS = bufH[1] * srcW*F;
            Array32f buf(dstC);
            dst += yBeg * p.dstW * p.dstC;
            for (size_t y = yBeg; y < yEnd; ++y)
            {
                for (size_t x = 0; x < dstW; ++x)
                {
                    if (bias)
                        memcpy(buf.data, bias, dstC * sizeof(float));
                    else
                        memset(buf.data, 0, dstC * sizeof(float));
                    const float * pw = weight;
                    const float * ps = src + ((y&srcM)*srcW + x)*F;
                    for (size_t sc = 0; sc < srcC; ++sc)
                    {
                        float s = ps[sc*srcS];
                        for (size_t dc = 0; dc < dstC; ++dc)
                            buf[dc] += s * pw[dc];
                        pw += dstC;
                    }
                    for (size_t dc = 0; dc < dstC; ++dc)
                        Update<update>(dst + dc, Activate<type>(buf[dc], params, dc));
                    dst += p.dstC;
                }
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p, bool old)
            : Simd::MergedConvolution(p)
        {
            _old = old;
            _sizeS = p.conv[0].srcH*p.conv[0].srcW*p.conv[0].srcC;
            _sizeD = p.conv[2].dstH*p.conv[2].dstW*p.conv[2].dstC;
            if (_old)
            {
                _sizeB[0] = p.conv[1].srcH*p.conv[1].srcW*p.conv[1].srcC;
                _sizeB[1] = p.conv[1].dstH*p.conv[1].dstW*p.conv[1].dstC;
                switch (p.conv[0].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[0] = DirectConvolutionOld<SimdConvolutionActivationIdentity, UpdateSet>; break;
                case SimdConvolutionActivationRelu: _convolution[0] = DirectConvolutionOld<SimdConvolutionActivationRelu, UpdateSet>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[0] = DirectConvolutionOld<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[0] = DirectConvolutionOld<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
                case SimdConvolutionActivationPrelu: _convolution[0] = DirectConvolutionOld<SimdConvolutionActivationPrelu, UpdateSet>; break;
                default: assert(0);
                }
                switch (p.conv[1].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[1] = DepthwiseConvolutionOld<SimdConvolutionActivationIdentity>; break;
                case SimdConvolutionActivationRelu: _convolution[1] = DepthwiseConvolutionOld<SimdConvolutionActivationRelu>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[1] = DepthwiseConvolutionOld<SimdConvolutionActivationLeakyRelu>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[1] = DepthwiseConvolutionOld<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationPrelu: _convolution[1] = DepthwiseConvolutionOld<SimdConvolutionActivationPrelu>; break;
                default: assert(0);
                }
                if (p.add)
                {
                    switch (p.conv[2].activation)
                    {
                    case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationIdentity, UpdateAdd>; break;
                    case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationRelu, UpdateAdd>; break;
                    case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationLeakyRelu, UpdateAdd>; break;
                    case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationRestrictRange, UpdateAdd>; break;
                    case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationPrelu, UpdateAdd>; break;
                    default: assert(0);
                    }
                }
                else
                {
                    switch (p.conv[2].activation)
                    {
                    case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationIdentity, UpdateSet>; break;
                    case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationRelu, UpdateSet>; break;
                    case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
                    case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
                    case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionOld<SimdConvolutionActivationPrelu, UpdateSet>; break;
                    default: assert(0);
                    }
                }
            }
            else
            {
                _F = 1;
                _yStep[1] = 1;
                for (_bufH[1] = 1; _bufH[1] < _yStep[1]; _bufH[1] *= 2);
                _yStep[0] = _yStep[1] * p.conv[1].strideY;
                for (_bufH[0] = 1; _bufH[0] < (_yStep[1] - 1) * p.conv[1].strideY + p.conv[1].kernelY; _bufH[0] *= 2);
                _bufC[0] = (p.conv[0].dstC + _F - 1) / _F;
                _bufC[1] = (p.conv[1].dstC + _F - 1) / _F;
                _sizeB[0] = _bufC[0] * _bufH[0] * p.conv[0].dstW * _F;
                _sizeB[1] = _bufC[1] * _bufH[1] * p.conv[1].dstW * _F;

                switch (p.conv[0].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[0] = InputConvolution<SimdConvolutionActivationIdentity>; break;
                case SimdConvolutionActivationRelu: _convolution[0] = InputConvolution<SimdConvolutionActivationRelu>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[0] = InputConvolution<SimdConvolutionActivationLeakyRelu>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[0] = InputConvolution<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationPrelu: _convolution[0] = InputConvolution<SimdConvolutionActivationPrelu>; break;
                default: assert(0);
                }
                switch (p.conv[1].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[1] = DepthwiseConvolution<SimdConvolutionActivationIdentity>; break;
                case SimdConvolutionActivationRelu: _convolution[1] = DepthwiseConvolution<SimdConvolutionActivationRelu>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[1] = DepthwiseConvolution<SimdConvolutionActivationLeakyRelu>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[1] = DepthwiseConvolution<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationPrelu: _convolution[1] = DepthwiseConvolution<SimdConvolutionActivationPrelu>; break;
                default: assert(0);
                }
                if (p.add)
                {
                    switch (p.conv[2].activation)
                    {
                    case SimdConvolutionActivationIdentity: _convolution[2] = OutputConvolution<SimdConvolutionActivationIdentity, UpdateAdd>; break;
                    case SimdConvolutionActivationRelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationRelu, UpdateAdd>; break;
                    case SimdConvolutionActivationLeakyRelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationLeakyRelu, UpdateAdd>; break;
                    case SimdConvolutionActivationRestrictRange: _convolution[2] = OutputConvolution<SimdConvolutionActivationRestrictRange, UpdateAdd>; break;
                    case SimdConvolutionActivationPrelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationPrelu, UpdateAdd>; break;
                    default: assert(0);
                    }
                }
                else
                {
                    switch (p.conv[2].activation)
                    {
                    case SimdConvolutionActivationIdentity: _convolution[2] = OutputConvolution<SimdConvolutionActivationIdentity, UpdateSet>; break;
                    case SimdConvolutionActivationRelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationRelu, UpdateSet>; break;
                    case SimdConvolutionActivationLeakyRelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
                    case SimdConvolutionActivationRestrictRange: _convolution[2] = OutputConvolution<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
                    case SimdConvolutionActivationPrelu: _convolution[2] = OutputConvolution<SimdConvolutionActivationPrelu, UpdateSet>; break;
                    default: assert(0);
                    }
                }
            }
        }

        size_t MergedConvolution::ExternalBufferSize() const
        {
            return _sizeB[0] + _sizeB[1];
        }

        size_t MergedConvolution::InternalBufferSize() const
        {
            return _buffer.size + _weightR[0].size + _weightR[1].size + _weightR[2].size;
        }

        void MergedConvolution::SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params)
        {
            Simd::MergedConvolution::SetParams(weight, internal, bias, params);
        }

        void MergedConvolution::Forward(const float * src, float * buf, float * dst)
        {
            const MergConvParam & p = _param;
            float * buf0 = Buffer(buf);
            float * buf1 = buf0 + _sizeB[0];
            for (size_t b = 0; b < p.batch; ++b)
            {
                if (_old)
                {
                    _convolution[0](src, p.conv[0], 0, p.conv[0].dstH, _bufH, _weight[0], _bias[0], _params[0], buf0);
                    _convolution[1](buf0, p.conv[1], 0, p.conv[1].dstH, _bufH, _weight[1], _bias[1], _params[1], buf1);
                    _convolution[2](buf1, p.conv[2], 0, p.conv[2].dstH, _bufH, _weight[2], _bias[2], _params[2], dst);
                }
                else
                {
                    for (size_t yBeg1 = 0, yBeg0 = 0; yBeg1 < p.conv[1].dstH;)
                    {
                        size_t yEnd1 = Simd::Min(yBeg1 + _yStep[1], p.conv[1].dstH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + _yStep[0], (_yStep[1] - 1)*p.conv[1].strideY + p.conv[1].kernelY - p.conv[1].padY, p.conv[0].dstH);
                        _convolution[0](src, p.conv[0], yBeg0, yEnd0, _bufH, _weight[0], _bias[0], _params[0], buf0);
                        _convolution[1](buf0, p.conv[1], yBeg1, yEnd1, _bufH, _weight[1], _bias[1], _params[1], buf1);
                        _convolution[2](buf1, p.conv[2], yBeg1, yEnd1, _bufH, _weight[2], _bias[2], _params[2], dst);
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
            return new Base::MergedConvolution(param, true);
        }
    }
}
