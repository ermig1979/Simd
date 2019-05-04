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

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<::SimdConvolutionActivationType type, UpdateType update> void DirectConvolutionBiasActivation(
            const float * src, const SimdConvolutionParameters & p, size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
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
                                size_t sx = dx * strideX + kx  - padX;
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

        template<::SimdConvolutionActivationType type> void DepthwiseConvolutionBiasActivation(
            const float * src, const SimdConvolutionParameters & p, size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
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

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Simd::MergedConvolution(p)
        {
            for (size_t i = 0; i < p.count; ++i)
            {
                _sizeS[i] = p.conv[i].srcH*p.conv[i].srcW*p.conv[i].srcC;
                _sizeD[i] = p.conv[i].dstH*p.conv[i].dstW*p.conv[i].dstC;
            }
            switch (p.conv[0].activation)
            {
            case SimdConvolutionActivationIdentity: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateSet>; break;
            case SimdConvolutionActivationRelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateSet>; break;
            case SimdConvolutionActivationLeakyRelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
            case SimdConvolutionActivationRestrictRange: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
            case SimdConvolutionActivationPrelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateSet>; break;
            default: assert(0);
            }
            switch (p.conv[1].activation)
            {
            case SimdConvolutionActivationIdentity: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationPrelu>; break;
            default: assert(0);
            }
            if (p.add)
            {
                switch (p.conv[2].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateAdd>; break;
                case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateAdd>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateAdd>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateAdd>; break;
                case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateAdd>; break;
                default: assert(0);
                }
            }
            else
            {
                switch (p.conv[2].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateSet>; break;
                case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateSet>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
                case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateSet>; break;
                default: assert(0);
                }
            }
        }

        size_t MergedConvolution::ExternalBufferSize() const
        {
            const MergConvParam & p = _param;
            return _sizeD[0] + _sizeD[1];
        }

        size_t MergedConvolution::InternalBufferSize() const
        {
            return _buffer.size;
        }

        void MergedConvolution::SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params)
        {
            Simd::MergedConvolution::SetParams(weight, internal, bias, params);
        }

        void MergedConvolution::Forward(const float * src, float * buf, float * dst)
        {
            const MergConvParam & p = _param;
            float * buf0 = Buffer(buf);
            float * buf1 = buf0 + _sizeD[0];
            for (size_t b = 0; b < p.batch; ++b)
            {
                _convolution[0](src, p.conv[0], 0, p.conv[0].dstH, _weight[0], _bias[0], _params[0], buf0);
                _convolution[1](buf0, p.conv[1], 0, p.conv[1].dstH, _weight[1], _bias[1], _params[1], buf1);
                _convolution[2](buf1, p.conv[2], 0, p.conv[2].dstH, _weight[2], _bias[2], _params[2], dst);
                src += _sizeS[0];
                dst += _sizeD[2];
            }
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new MergedConvolution(param);
        }
    }
}
