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
        SynetConvolution16bNhwcGemm::SynetConvolution16bNhwcGemm(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
        }

        String SynetConvolution16bNhwcGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemm";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetConvolution16bNhwcGemm::SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.M = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = F;
            a.microD = microD;
            a.microM = microM;
            a.microK = microK;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD / 2, a.microK), a.microK, a.bufK);
            a.batch = 1;
            size_t bufSize = a.M * a.bufK * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroK / 2, a.microD), a.microD, a.bufD);
            a.bufM = a.batch * p.dstH * AlignHi(p.dstW, a.F);
            a.elem = _elemD;
            a.reorderType = 0;
            a.sumBuf = (_dst16b && a.macroK < a.K) || a.microK > 2 ? 1 : 0;
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;
            a.dB = (a.sumBuf ? a.macroD : p.dstC);

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;
        }

        size_t SynetConvolution16bNhwcGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            if(_convert)
                size += a.bufM * a.bufK * sizeof(uint16_t);
            if (a.sumBuf)
                size += a.macroD * a.bufM * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcGemm::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t D = DivHi(p.dstC, _alg.F);
            _weight.Resize(a.bufK * a.bufD, true);
            uint16_t* dst = _weight.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t k = 0; k < a.bufK; k += 2)
                {
                    const float* src = weight + k * p.dstC + d * _alg.F;
                    for (size_t f = 0; f < _alg.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (d * _alg.F + f < p.dstC && k + i < a.K)
                                *(dst++) = Float32ToBFloat16(src[i * p.dstC]);
                            else
                                *(dst++) = 0;
                        }
                        src++;
                    }
                }
            }
        }

        void SynetConvolution16bNhwcGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufB = _convert ? Allocate<uint16_t>(buf8, a.bufM * a.bufK) : NULL;
            float* bufS = a.sumBuf ? Allocate<float>(buf8, a.macroD * a.bufM) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* buf = _convert ? bufB : (uint16_t*)src;
                float* sum = a.sumBuf ? bufS : (float*)dst;
                Forward(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemm::Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * a.batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                const uint16_t* weight = _weight.data + dc * a.bufK;
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    for (size_t yBeg = 0; yBeg < dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                        size_t bufOffs = (a.macroK < a.bufK || _convert == NULL) ? 
                            yBeg * (_convert ? AlignHi(p.dstW, a.F) : p.dstW) * a.bufK + (a.reorderType ? mak * a.F : mak) : 0;
                        size_t sumOffs = a.macroK < a.bufK ? yBeg * p.dstW * a.dB : 0;
                        size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                        if (dc == 0 && mak == 0 && _convert)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = p.dstH * p.dstW * a.bufK;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _convert(src + b * dS, p, a, 0, p.dstH, buf + b * dB);
                            }
                            else
                                _convert(src, p, a, yBeg, yEnd, buf + bufOffs);
                        }
                        if (mak + macroK == a.bufK)
                            _convolutions[1](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 1 : 0,
                                weight, bias, params, sum + sumOffs, dst + dstOffs);
                        else
                            _convolutions[0](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, mak == 0 ? 1 : 0,
                                weight, bias, params, sum + sumOffs, dst + dstOffs);
                        yBeg = yEnd;
                    }
                    weight += macroK * a.F;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
                if (!a.sumBuf)
                    sum += macroD;
            }
        }

        bool SynetConvolution16bNhwcGemm::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }
    }
#endif
}
