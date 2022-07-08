/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdCpu.h"

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

        template<SimdConvolutionActivationType type, UpdateType update> void DirectBf16(const float* src, 
            const ConvParam32f& p, const uint16_t * weight, const float* bias, const float* params, float* dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC, srcC2 = AlignLo(srcC, 2);
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    if (update == UpdateAdd)
                        memcpy(buf.data, dst, dstC * sizeof(float));
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
                    for (size_t dc = 0; dc < dstC; ++dc)
                        dst[dc] = Activate<type>(buf[dc] + bias[dc], params, dc);
                    dst += p.dstC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseBf16(const float* src, 
            const ConvParam32f& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = 0; 
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
                                        const float* pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float* ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum + bias[c], params, c);
                    }
                    dst += srcC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolutionBf16(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            DirectBf16<type, UpdateSet>((const float*)src, p, weight, bias, params, dst);
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionBf16(const float* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint16_t* dst)
        {
            DepthwiseBf16<type>(src, p, weight, bias, params, (float*)dst);
        }

        template<SimdConvolutionActivationType type, UpdateType update> void OutputConvolutionBf16(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst, int zero)
        {
            DirectBf16<type, update>((const float*)src, p, weight, bias, params, dst);
        }

        template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t index, InputPtr & input, DepthwisePtr & depthwise, OutputPtr & output)
        {
            switch (index)
            {
            case 0:
                if (p.conv[0].group == 1)
                    input = InputConvolutionBf16<type>;
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
            const ConvParam32f& beg = p.conv[0];
            const ConvParam32f& end = p.conv[p.count - 1];
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
                return _sizeB[1] + (_sizeB[0] + _sizeB[2]) / 2 + (SIMD_ALIGN * 2 / 4);
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

        void SynetMergedConvolution32fBf16::SetInputWeight(const float* src, const ConvParam32f& p)
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

        void SynetMergedConvolution32fBf16::SetDepthwiseWeight(const float* src, const ConvParam32f& p)
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

        void SynetMergedConvolution32fBf16::SetOutputWeight(const float* src, const ConvParam32f& p)
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

        void SynetMergedConvolution32fBf16::SetBias(const float* src, const ConvParam32f& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            dst.Resize(AlignHiAny(p.dstC, Simd::Max(size_t(1), a.miC)), true);
            if (src)
                memcpy(dst.data, src, p.dstC * sizeof(float));
        }

        void SynetMergedConvolution32fBf16::SetParams(const float* src, const ConvParam32f& p, Array32f& dst)
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
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            const ConvParam32f& c2 = p.conv[2];
            const AlgParam& a = _alg;
            for (size_t b = 0; b < c0.batch; ++b)
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
                        if (p.add)
                            memcpy(dst, src, sizeof(float) * _sizeS);
                        _output[0]((uint16_t*)buf1, c2, a, 0, 0, c2.dstH, _weightO.data, _bias[2].data, _params[2].data, dst, 0);
                    }
                    else
                        _depthwise(buf0, c1, a, 0, 0, c1.dstH, _weightD.data, _bias[1].data, _params[1].data, (uint16_t*)dst);
                }
                src += _sizeS;
                dst += _sizeD;
            }
        };

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cdc::SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p)
            : SynetMergedConvolution32fBf16(p)
        {
        }

        void SynetMergedConvolution32fBf16Cdc::Forward(const float* src, float* buf, float* dst)
        {
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            const ConvParam32f& c2 = p.conv[2];
            const AlgParam& a = _alg;

            uint8_t* buffer = (uint8_t*)Buffer(buf);
            uint16_t* buf0 = Allocate<uint16_t>(buffer, _sizeB[0]);
            SetGap(buffer);
            float* buf1 = Allocate<float>(buffer, _sizeB[1]);
            uint16_t* buf2 = Allocate<uint16_t>(buffer, _sizeB[2]);
            SetGap(buffer);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        _convert(src, c0, yBeg0, yEnd0, buf0, a.bufH[0]);
                        _input(buf0, c0, a, maC, yBeg1, yEnd1, _weightI.data + c * a.dw[0], 
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf1);
                        _depthwise(buf1, c1, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[1], 
                            _bias[1].data + c, _params[1].data + c * a.dp[1], buf2);
                        if (p.add && c == 0)
                        {
                            size_t offset = yBeg1 * p.conv[2].dstW * p.conv[2].dstC, size = (yEnd1 - yBeg1) * p.conv[2].dstW * p.conv[2].dstC;
                            memcpy(dst + offset, src + offset, sizeof(float) * size);
                        }
                        if (c + maC == C)
                            _output[0](buf2, c2, a, maC, yBeg2, yEnd2, _weightO.data + c * a.dw[2], 
                                _bias[2].data, _params[2].data, dst, (maC != C || p.add) ? 0 : 1);
                        else
                            _output[1](buf2, c2, a, maC, yBeg2, yEnd2, _weightO.data + c * a.dw[2], 
                                _bias[2].data, _params[2].data, dst, (c != 0 || p.add) ? 0 : 1);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fBf16Cdc::Preferable(const MergConvParam32f& p)
        {
            return p.count == 3;
        }

        void SynetMergedConvolution32fBf16Cdc::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            const ConvParam32f& c2 = p.conv[2];
            AlgParam& a = _alg;

            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 3; ++i)
            {
                const ConvParam32f& c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC * 2 : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.dstC / count, 2 * a.miC);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1] * c0.strideY;
                a.yStart[0] = Simd::Min((a.yStart[1] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);
                a.bufH[0] = Pow2Hi(Simd::Max((a.yStep[1] - 1) * c0.strideY + c0.kernelY, a.yStart[0]));

                _sizeB[0] = a.bufH[0] * p.conv[0].srcW * p.conv[0].srcC;
                _sizeB[1] = a.bufH[1] * p.conv[1].srcW * a.maC;
                _sizeB[2] = a.bufH[2] * p.conv[1].dstW * a.maC;
                if (_sizeB[0] * 2 + _sizeB[1] * 4 + _sizeB[2] * 2 <= L2)
                    break;
            }
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX * c0.srcC;
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.dw[2] = AlignHiAny(c2.dstC, 2 * a.miC);
            
            ((ConvParam32f&)c1).dstT = SimdTensorData16b;
            ((ConvParam32f&)c2).srcT = SimdTensorData16b;
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cd::SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p)
            : SynetMergedConvolution32fBf16(p)
        {
        }

        void SynetMergedConvolution32fBf16Cd::Forward(const float* src, float* buf, float* dst)
        {
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            const AlgParam& a = _alg;

            uint8_t* buffer = (uint8_t*)Buffer(buf);
            uint16_t* buf0 = Allocate<uint16_t>(buffer, _sizeB[0]);
            SetGap(buffer);
            float* buf1 = Allocate<float>(buffer, _sizeB[1]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        _convert(src, c0, yBeg0, yEnd0, buf0, a.bufH[0]);
                        _input(buf0, c0, a, maC, yBeg1, yEnd1, _weightI.data + c * a.dw[0],
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf1);
                        _depthwise(buf1, c1, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[1],
                            _bias[1].data + c, _params[1].data + c * a.dp[1], (uint16_t*)dst);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fBf16Cd::Preferable(const MergConvParam32f& p)
        {
            return p.count == 2 && p.conv[0].group == 1;
        }

        void SynetMergedConvolution32fBf16Cd::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            AlgParam& a = _alg;

            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam32f& c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC * 2 : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.dstC / count, 2 * a.miC);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1] * c0.strideY;
                a.yStart[0] = Simd::Min((a.yStart[1] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);
                a.bufH[0] = Pow2Hi(Simd::Max((a.yStep[1] - 1) * c0.strideY + c0.kernelY, a.yStart[0]));

                _sizeB[0] = a.bufH[0] * p.conv[0].srcW * p.conv[0].srcC;
                _sizeB[1] = a.bufH[1] * p.conv[1].srcW * a.maC;
                if (_sizeB[0] * 2 + _sizeB[1] * 4 <= L2)
                    break;
            }
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX * c0.srcC;
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.dw[2] = 0;
            a.bufH[2] = 0;
            _sizeB[2] = 0;
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Dc::SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p)
            : SynetMergedConvolution32fBf16(p)
        {
        }

        void SynetMergedConvolution32fBf16Dc::Forward(const float* src, float* buf, float* dst)
        {
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            const AlgParam& a = _alg;

            uint8_t* buffer = (uint8_t*)Buffer(buf);
            uint16_t* buf2 = Allocate<uint16_t>(buffer, _sizeB[2]);
            SetGap(buffer);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c0.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c0.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c0.srcH);
                        _depthwise(src + c, c0, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[0], _bias[0].data + c,
                            _params[0].data + c * a.dp[0], buf2);
                        if (c + maC == C)
                            _output[0](buf2, c1, a, maC, yBeg2, yEnd2, _weightO.data + c * a.dw[1],
                                _bias[1].data, _params[1].data, dst, maC != C ? 0 : 1);
                        else
                            _output[1](buf2, c1, a, maC, yBeg2, yEnd2, _weightO.data + c * a.dw[1],
                                _bias[1].data, _params[1].data, dst, c != 0 ? 0 : 1);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fBf16Dc::Preferable(const MergConvParam32f& p)
        {
            return p.count == 2 && p.conv[1].group == 1;
        }

        void SynetMergedConvolution32fBf16Dc::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam32f& p = _param;
            const ConvParam32f& c0 = p.conv[0];
            const ConvParam32f& c1 = p.conv[1];
            AlgParam& a = _alg;

            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam32f& c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC * 2 : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.srcC / count, 2 * a.miC);

            for (size_t yStep = c0.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c0.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);

                _sizeB[2] = a.bufH[2] * p.conv[1].srcW * a.maC;
                if (_sizeB[2] * 2 <= L2)
                    break;
            }
            a.bufH[0] = 0;
            a.bufH[1] = 0;
            _sizeB[0] = 0;
            _sizeB[1] = 0;
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX;
            a.dw[1] = AlignHiAny(c1.dstC, 2 * a.miC);

            ((ConvParam32f&)c0).dstT = SimdTensorData16b;
            ((ConvParam32f&)c1).srcT = SimdTensorData16b;
        }
    }
#endif
}
