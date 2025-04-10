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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution16bNhwcSpecV0::SynetConvolution16bNhwcSpecV0(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _preprocess = 0;
            _convolution = 0;
            _postprocess = 0;
        }

        String SynetConvolution16bNhwcSpecV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV0";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetConvolution16bNhwcSpecV0::SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microD = microD;
            a.microS = microS;
            a.microC = microC;
            a.srcC = AlignHi(p.srcC, a.microC);
            a.srcH = p.srcH + p.padY + p.padH;
            a.srcW = p.srcW + p.padX + p.padW;
            a.dstC = AlignHi(p.dstC, a.F);
            a.K = p.kernelX * p.kernelY;

            a.macroC = Simd::RestrictRange(AlignLo(L1 / a.microD / a.K / 2, a.microC), a.microC, a.srcC);
            a.batch = 1;
            size_t bufSize = a.srcC * a.srcH * a.srcW * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroC / a.srcW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / a.K / 2, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));
            a.numH = DivHi(p.dstH * a.batch, a.macroH);
            a.elem = _elemD;
            a.bufS = a.batch * a.srcH * a.srcW * a.srcC + a.F * a.F;
            a.bufD = (a.batch * a.srcH * a.srcW + a.numH * a.F) * a.macroD;

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;
        }

        size_t SynetConvolution16bNhwcSpecV0::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufS * sizeof(uint16_t);
            size += a.bufD * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcSpecV0::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcSpecV0::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.srcC * a.dstC * a.K, true);
            uint16_t* dst = _weight.data;
            for (size_t mad = 0; mad < p.dstC; mad += _alg.F)
            {
                for (size_t mac = 0; mac < p.srcC; mac += a.microC)
                {
                    for (size_t k = 0; k < a.K; k++)
                    {
                        for (size_t c = 0; c < a.microC; c += 2)
                        {
                            const float* src = weight + (k * p.srcC + mac + c) * p.dstC + mad;
                            for (size_t d = 0; d < a.F; ++d)
                            {
                                for (size_t i = 0; i < 2; ++i)
                                {
                                    if (mad + d < p.dstC && mac + c + i < p.srcC)
                                        *(dst++) = Float32ToBFloat16(src[i * p.dstC]);
                                    else
                                        *(dst++) = 0;
                                }
                                src++;
                            }
                         }
                    }
                }
            }
        }

        void SynetConvolution16bNhwcSpecV0::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufS = a.bufS ? Allocate<uint16_t>(buf8, a.bufS) : NULL;
            float* bufD = a.bufD ? Allocate<float>(buf8, a.bufD) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* buf = bufS ? bufS : (uint16_t*)src;
                float* sum = bufD ? bufD : (float*)dst;
                Forward(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcSpecV0::Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * a.batch, padY = (p.kernelY - 1) / 2, dstHb = a.srcH * a.batch + 1 - p.kernelY;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const uint16_t* weight = _weight.data + mad * a.srcC * a.K;
                for (size_t mac = 0; mac < a.srcC; mac += a.macroC)
                {
                    size_t macroC = Simd::Min(a.srcC, mac + a.macroC) - mac;
                    for (size_t dyBeg = 0, dyTime = 0; dyBeg < dstH; dyTime++)
                    {
                        size_t dyEnd = Simd::Min(dyBeg + a.macroH, dstH);
                        if (mad == 0 && mac == 0)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = a.srcH * a.srcW * a.microC;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _preprocess(src + b * dS, p, a, 0, p.dstH, buf + b * dB);
                            }
                            else
                                _preprocess(src, p, a, dyBeg, dyEnd, buf);
                        }
                        if (a.batch > 1)
                        {
                            _convolution(buf + mac * a.srcH * a.srcW * a.batch, p, a, macroD, dstHb, macroC, mac == 0 ? 1 : 0, weight, sum);
                        }
                        else
                        {
                            _convolution(buf + mac * a.srcH * a.srcW + dyBeg * a.srcW * a.microC, p, a, macroD, dyEnd - dyBeg,
                                macroC, mac == 0 ? 1 : 0, weight, sum + (dyBeg * a.srcW + dyTime * a.F) * a.macroD);
                        }
                        if (mac + macroC == a.srcC)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = a.srcH * a.srcW * a.macroD;
                                size_t dD = p.dstH * p.dstW * p.dstC * a.elem;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _postprocess(sum + b * dS, p, a, macroD, 0, p.dstH, bias, params, dst + b * dD);
                            }
                            else
                                _postprocess(sum + dyTime * a.F * a.macroD, p, a, macroD, dyBeg, dyEnd, bias, params, dst);
                        }
                        dyBeg = dyEnd;
                    }
                    weight += macroC * a.K * a.F;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetConvolution16bNhwcSpecV0::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) && p.dstC >= 4;
        }
    }
#endif
}
