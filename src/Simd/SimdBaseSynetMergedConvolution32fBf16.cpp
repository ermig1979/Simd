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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        typedef SynetMergedConvolution32fBf16::AlgParam AlgParam;
        typedef SynetMergedConvolution32fBf16::ConvertPtr ConvertPtr;
        typedef SynetMergedConvolution32fBf16::InputConvolutionPtr InputPtr;
        typedef SynetMergedConvolution32fBf16::DepthwiseConvolutionPtr DepthwisePtr;
        typedef SynetMergedConvolution32fBf16::OutputConvolutionPtr OutputPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, UpdateType update> void DirectConvolutionBf16(const float* src, 
            const SimdConvolutionParameters& p, const uint16_t * weight, const float* bias, const float* params, float* dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = 0; dy < p.dstH; ++dy)
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
                                    const uint16_t* pw = weight + (ky * kernelX + kx) * srcC * dstC;
                                    const float* ps = src + (sy * srcW + sx) * srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        float s = RoundToBFloat16(ps[sc]);
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += s * BFloat16ToFloat32(pw[dc]);
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

        template<SimdConvolutionActivationType type, UpdateType update> void InputConvolutionBf16(const uint16_t* src, const SimdConvolutionParameters& p, 
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            DirectConvolutionBf16<type, update>((const float*)src, p, weight, bias, params, dst);
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionBf16(const float* src, const SimdConvolutionParameters& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint16_t* dst)
        {
            DepthwiseConvolution<type>(src, p, maC, yBeg, yEnd, a.bufH, weight, bias, params, (float*)dst, 0);
        }

        template<SimdConvolutionActivationType type, UpdateType update> void OutputConvolutionBf16(const uint16_t* src, const SimdConvolutionParameters& p, 
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst, int zero)
        {
            DirectConvolutionBf16<type, update>((const float*)src, p, weight, bias, params, dst);
        }

        template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t index, InputPtr & input, DepthwisePtr & depthwise, OutputPtr & output)
        {
            switch (index)
            {
            case 0:
                if (p.conv[0].group == 1)
                    input = InputConvolutionBf16<type, UpdateSet>;
                else
                    depthwise = DepthwiseConvolutionBf16<type>;
                break;
            case 1:
                if (p.conv[1].group == 1)
                    output = OutputConvolutionBf16<type, UpdateSet>;
                else
                    depthwise = DepthwiseConvolutionBf16<type>;
                break;
            case 2:
                if (p.add)
                    output = OutputConvolutionBf16<type, UpdateAdd>;
                else
                    output = OutputConvolutionBf16<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fBf16::SynetMergedConvolution32fBf16(const MergConvParam32f& p)
           : Simd::SynetMergedConvolution32f(p)
        {
            memset(&_alg, 0, sizeof(_alg));
            _convert = NULL, _input = NULL, _depthwise = NULL, _output[0] = NULL, _output[1] = NULL;
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            _dw0 = beg.group != 1;
            _1x1 = beg.kernelY == 1 && beg.strideY == 1;
            _sizeB[0] = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            _sizeB[1] = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            _sizeB[2] = 0;
            for (size_t i = 0; i < p.count; ++i)
            {
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(_param, i, _input, _depthwise, _output[0]); break;
                default: assert(0);
                }
            }
        }

        size_t SynetMergedConvolution32fBf16::ExternalBufferSize() const
        {
            if (_alg.miC)
                return _sizeB[1] + (_sizeB[0] + _sizeB[2]) / 2;
            else
                return _sizeB[1] + _sizeB[0];
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
            assert(p.group == 1);
            if (_alg.miC)
            {
                size_t F = _alg.miC * 2, C = DivHi(p.srcC, 2), D = DivHi(p.dstC, F), K = p.kernelY * p.kernelX;
                _weightI.Resize(K * C * D * F * 2, true);
                uint16_t* dst = _weightI.data;
                for (size_t d = 0; d < D; d++)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        for (size_t c = 0; c < C; ++c)
                        {
                            const float* ps = src + (k * p.srcC + c * 2) * p.dstC + d * F;
                            for (size_t f = 0; f < F; ++f)
                            {
                                for (size_t i = 0; i < 2; ++i)
                                {
                                    if (d * F + f < p.dstC && c * 2 + i < p.srcC)
                                        *(dst++) = Float32ToBFloat16(ps[i * p.dstC]);
                                    else
                                        *(dst++) = 0;
                                }
                                ps++;
                            }
                        }
                    }
                }            
            }
            else
            {
                _weightI.Resize(p.kernelY * p.kernelX * p.srcC * p.dstC, true);
                Float32ToBFloat16(src, _weightI.size, _weightI.data);
            }
        }

        void SynetMergedConvolution32fBf16::SetDepthwiseWeight(const float* src, const SimdConvolutionParameters& p)
        {
            assert(p.srcC == p.dstC && p.srcC == p.group);
            if (_alg.miC)
            {
                size_t D = p.dstC, K = p.kernelY * p.kernelX, F = _alg.miC;
                _weightD.Resize(AlignHiAny(D, F) * K);
                float* dst = _weightD.data;
                for (size_t d = 0; d < D; d += F)
                {
                    size_t n = Simd::Min(F, D - d);
                    for (size_t k = 0; k < K; k++)
                    {
                        size_t i = 0;
                        for (; i < n; ++i)
                            dst[i] = src[k * D + d + i];
                        for (; i < F; ++i)
                            dst[i] = 0;
                        dst += F;
                    }
                }
            }
            else
                _weightD.Assign(src, p.kernelY * p.kernelX * p.srcC * p.dstC / p.group);
        }

        void SynetMergedConvolution32fBf16::SetOutputWeight(const float* src, const SimdConvolutionParameters& p)
        {
            assert(p.group == 1 && p.kernelX == 1 && p.kernelY == 1 && p.strideX == 1 && p.strideY == 1);
            if (_alg.miC)
            {
                size_t F = _alg.miC * 2, C = DivHi(p.srcC, 2), D = DivHi(p.dstC, F), M = DivHi(_alg.maC, 2);
                _weightO.Resize(C * D * F * 2, true);
                uint16_t* dst = _weightO.data;
                for (size_t cB = 0; cB < C; cB += M)
                {
                    size_t cE = Simd::Min(C, cB + M);
                    for (size_t d = 0; d < D; d++)
                    {
                        for (size_t c = cB; c < cE; ++c)
                        {
                            const float* ps = src + c * 2 * p.dstC + d * F;
                            for (size_t f = 0; f < F; ++f)
                            {
                                for (size_t i = 0; i < 2; ++i)
                                {
                                    if (d * F + f < p.dstC && c * 2 + i < p.srcC)
                                        *(dst++) = Float32ToBFloat16(ps[i * p.dstC]);
                                    else
                                        *(dst++) = 0;
                                }
                                ps++;
                            }
                        }
                    }
                }
            }
            else
            {
                _weightO.Resize(p.kernelY * p.kernelX * p.srcC * p.dstC, true);
                Float32ToBFloat16(src, _weightO.size, _weightO.data);
            }
        }

        void SynetMergedConvolution32fBf16::SetBias(const float* src, const SimdConvolutionParameters& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            dst.Resize(AlignHiAny(p.dstC, Simd::Max(size_t(1), a.miC)), true);
            if (src)
                memcpy(dst.data, src, p.dstC * sizeof(float));
        }

        void SynetMergedConvolution32fBf16::SetParams(const float* src, const SimdConvolutionParameters& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                dst.Resize(AlignHiAny(p.dstC, Simd::Max(size_t(1), a.miC)), true);
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

        void SynetMergedConvolution32fBf16::Forward(const float* src, float* buf, float* dst)
        {
            uint8_t* buffer = (uint8_t*)Buffer(buf);
            float* buf0 = Allocate<float>(buffer, _sizeB[0]);
            float* buf1 = Allocate<float>(buffer, _sizeB[1]);
            const MergConvParam32f& p = _param;
            const SimdConvolutionParameters& c0 = p.conv[0];
            const SimdConvolutionParameters& c1 = p.conv[1];
            const SimdConvolutionParameters& c2 = p.conv[2];
            const AlgParam& a = _alg;
            for (size_t b = 0; b < p.batch; ++b)
            {
                if (_dw0)
                {
                    _depthwise(src, c0, a, 0, 0, c0.dstH, _weightD.data, _bias[0].data, _params[0].data, (uint16_t*)buf0);
                    _output[0]((uint16_t*)buf0, c1, a, 0, 0, c1.dstH, _weightO.data, _bias[1].data, _params[1].data, dst, 0);
                }
                else
                {
                    _input((uint16_t*)src, c0, a, 0, 0, c0.dstH, _weightI.data, _bias[0].data, _params[0].data, buf0);
                    if (p.count > 2)
                    {
                        _depthwise(buf0, c1, a, 0, 0, c1.dstH, _weightD.data, _bias[1].data, _params[1].data, (uint16_t*)buf1);
                        _output[0]((uint16_t*)buf1, c2, a, 0, 0, c2.dstH, _weightO.data, _bias[2].data, _params[2].data, dst, 0);
                    }
                    else
                        _depthwise(buf0, c1, a, 0, 0, c1.dstH, _weightD.data, _bias[1].data, _params[1].data, (uint16_t*)dst);
                }
                src += _sizeS;
                dst += _sizeD;
            }
        };
    }
#endif
}
