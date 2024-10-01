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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
#if defined(SIMD_RUNTIME_DISABLE)
            _old.enable = true;
#else
            _old.enable = false;
#endif
            if (p.IsDilation(1))
            {
                if (p.srcC <= 3)
                    _old.enable = true;
                if (p.SizeW()*sizeof(float) > Base::AlgCacheL3() * 1.0)
                    _old.enable = true;
            }
            _old.convolution = NULL;
        }

        size_t SynetConvolution32fNhwcDirect::InternalBufferSize() const
        {
            size_t size = _buffer.size + _rWeight.size + _rBias.size + _rParams.size;
            size += _old.weight.size;
            return size;
        }

        void SynetConvolution32fNhwcDirect::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            SynetConvolution32f::SetParams(weight, internal, bias, params);
            if (_old.enable && _old.weight.data)
            {
                OldReorderWeight(weight, _old.weight.data);
                _weight = _old.weight.data;
                if (internal)
                    *internal = SimdTrue;
            }
            else
            if (_rWeight.data)
            {
                ReorderWeight(weight, _rWeight.data);
                _weight = _rWeight.data;
                if (internal)
                    *internal = SimdTrue;
            }
            if (_rBias.data)
            {
                if (bias)
                    memcpy(_rBias.data, bias, _param.dstC * sizeof(float));
                _bias = _rBias.data;
            }
            if (_rParams.data)
            {
                const ConvParam& p = _param;
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity:
                    _rParams.data[0] = -FLT_MAX;
                    _rParams.data[1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationRelu:
                    _rParams.data[0] = 0;
                    _rParams.data[1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationLeakyRelu:
                    for (size_t d = 0; d < p.dstC; ++d)
                        _rParams.data[d] = params[0];
                    break;
                case SimdConvolutionActivationRestrictRange:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationPrelu:
                    for (size_t d = 0; d < p.dstC; ++d)
                        _rParams.data[d] = params[d];
                    break;
                case SimdConvolutionActivationElu:
                    _rParams.data[0] = params[0];
                    break;
                case SimdConvolutionActivationHswish:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationMish:
                    _rParams.data[0] = params[0];
                    break;
                case SimdConvolutionActivationHardSigmoid:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationSwish:
                    _rParams.data[0] = params[0];
                    break;
                case SimdConvolutionActivationGelu:
                    break;
                default:
                    assert(0);
                }
                _params = _rParams.data;
            }
        }

        void SynetConvolution32fNhwcDirect::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                if(_old.enable)
                    _old.convolution(src, _param, _old.alg, _weight, _bias, _params, dst);
                else
                _run.Run(RunArgs(src, _param, _weight, _bias, _params, dst));
                src += _sizeS;
                dst += _sizeD;
            }
        }

        void SynetConvolution32fNhwcDirect::Forward(const float* src, const ConvParam& p, const AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (sc + macroC == p.srcC)
                            a.convolutions[TermLast](src + sc, p, a, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            a.convolutions[TermInterim](src + sc, p, a, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += a.F * macroC;
                }
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                weight += p.kernelY * p.kernelY * p.srcC * macroD - p.srcC * a.F;
            }
        }

        void SynetConvolution32fNhwcDirect::SetAlgParam(size_t F, size_t N, AlgParam & alg)
        {
            const ConvParam& p = _param;
            alg.F = F;
            alg.microD = F*N;
            alg.macroC = Simd::Min(Base::AlgCacheL1() / sizeof(float) / p.kernelY / p.kernelX / alg.microD, p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                alg.macroH = macroH;
                if (alg.macroC * p.srcW * (alg.macroH * p.strideY + p.kernelY * p.dilationY - 1) * sizeof(float) <= Base::AlgCacheL2())
                    break;
            }
            alg.macroD = Simd::RestrictRange(AlignLoAny(Base::AlgCacheL3() / sizeof(float) / p.kernelY / p.kernelX / alg.macroC, alg.microD), 
                alg.microD, AlignHiAny(p.dstC, alg.microD));
            alg.stepW = p.kernelY * p.kernelX * p.srcC * alg.F;
            _rWeight.Resize(DivHi(p.dstC, alg.F)*alg.stepW);
            _rBias.Resize(AlignHiAny(p.dstC, alg.F), true);
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, alg.F), true);
            else
                _rParams.Resize(2, true);
        }

        void SynetConvolution32fNhwcDirect::ReorderWeight(const float* src, float* dst)
        {
            const ConvParam& p = _param;
            const AlgParam & a = _run.At(0).alg;
            for (size_t dc = 0; dc < p.dstC; dc += a.F)
            {
                size_t F = Simd::Min(p.dstC, dc + a.F) - dc;
                const float* psrc = src;
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        for (size_t sc = 0; sc < p.srcC; ++sc)
                        {
                            size_t f = 0;
                            for (; f < F; ++f)
                                *(dst++) = psrc[f];
                            for (; f < a.F; ++f)
                                *(dst++) = 0.0f;
                            psrc += p.dstC;
                        }
                    }
                }
                src += F;
            }
        }

        void SynetConvolution32fNhwcDirect::OldSetAlgParam(size_t F)
        {
            const ConvParam& p = _param;
            AlgParam & a = _old.alg;
            a.F = F;
            a.microD = a.F*2;
            a.macroC = Simd::Min(Base::AlgCacheL1() / sizeof(float) / p.kernelY / p.kernelX / a.microD, p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                a.macroH = macroH;
                if (a.macroC * p.srcW * (a.macroH * p.strideY + p.kernelY * p.dilationY - 1) * sizeof(float) <= Base::AlgCacheL2())
                    break;
            }
            a.macroD = Simd::RestrictRange(AlignLoAny(Base::AlgCacheL3() / sizeof(float) / p.kernelY / p.kernelX / a.macroC, a.microD), 
                a.microD, AlignHiAny(p.dstC, a.microD));
            _old.weight.Resize(AlignHiAny(p.dstC, a.microD) * p.kernelY * p.kernelX * p.srcC);
            _rBias.Resize(AlignHiAny(p.dstC, a.microD), true);
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, a.microD), true);
            else
                _rParams.Resize(2, true);
        }

        void SynetConvolution32fNhwcDirect::OldReorderWeight(const float* src, float* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _old.alg;
            for (size_t da = 0; da < p.dstC; da += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, da + a.macroD) - da;
                for (size_t sa = 0; sa < p.srcC; sa += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sa + a.macroC) - sa;
                    for (size_t di = 0; di < macroD; di += a.microD)
                    {
                        size_t microD = Simd::Min(macroD, di + a.microD) - di;
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                for (size_t si = 0; si < macroC; si++)
                                {
                                    const float* s = src + ((ky * p.kernelX + kx) * p.srcC + sa + si) * p.dstC + da + di;
                                    size_t i = 0;
                                    for (; i < microD; i++)
                                        dst[i] = s[i];
                                    for (; i < a.microD; i++)
                                        dst[i] = 0;
                                    dst += a.microD;
                                }
                            }
                        }
                    }
                }
            }
        }

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam & p)
        {
            return false;
        }
    }
#endif
}
