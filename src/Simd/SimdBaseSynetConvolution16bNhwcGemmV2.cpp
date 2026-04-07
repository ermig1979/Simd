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
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution16bNhwcGemmV2::SynetConvolution16bNhwcGemmV2(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
        }

        String SynetConvolution16bNhwcGemmV2::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemmV2";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetConvolution16bNhwcGemmV2::SetAlgParam()
        {
            const int L1 = int(Base::AlgCacheL1()), L2 = int(Base::AlgCacheL2() * 0.5), L3 = int(Base::AlgCacheL3());
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.M = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = 16;
            a.microD = 32;
            a.microM = 32;
            a.microK = 32;
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

        size_t SynetConvolution16bNhwcGemmV2::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 4 * 16 * 16 * sizeof(float);
            if(_convert)
                size += a.bufM * a.bufK * sizeof(uint16_t);
            if (a.sumBuf)
                size += a.macroD * a.bufM * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcGemmV2::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcGemmV2::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t D = DivHi(p.dstC, _alg.microD);
            _weight.Resize(a.bufK * a.bufD, true);
            uint16_t* dst = _weight.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t k = 0; k < a.bufK; k += 2)
                {
                    const float* src = weight + k * p.dstC + d * _alg.microD;
                    for (size_t f = 0; f < _alg.microD; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (d * _alg.microD + f < p.dstC && k + i < a.K)
                                *(dst++) = Float32ToBFloat16(src[i * p.dstC]);
                            else
                                *(dst++) = 0;
                        }
                        src++;
                    }
                }
            }
        }

        void SynetConvolution16bNhwcGemmV2::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufT = _convert ? Allocate<uint16_t>(buf8, a.bufM * a.bufK) : NULL;
            float* bufS = a.sumBuf ? Allocate<float>(buf8, a.macroD * a.bufM) : NULL;
            float* bufD = Allocate<float>(buf8, 1024);
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* tmp = _convert ? bufT : (uint16_t*)src;
                float* sum = a.sumBuf ? bufS : (float*)dst;
                Forward(src, tmp, sum, bufD, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemmV2::Forward(const uint8_t* src, uint16_t* tmp, float* sum, float* buf, uint8_t* dst)
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
                        size_t tmpOffs = (a.macroK < a.bufK || _convert == NULL) ? 
                            yBeg * (_convert ? AlignHi(p.dstW, a.F) : p.dstW) * a.bufK + (a.reorderType ? mak * a.F : mak) : 0;
                        size_t sumOffs = a.macroK < a.bufK ? yBeg * (a.microK > 2 ? AlignHi(p.dstW, a.F) : p.dstW)* a.dB : 0;
                        size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                        if (dc == 0 && mak == 0 && _convert)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = p.dstH * p.dstW * a.bufK;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _convert(src + b * dS, p, a, 0, p.dstH, tmp + b * dB);
                            }
                            else
                                _convert(src, p, a, yBeg, yEnd, tmp + tmpOffs);
                        }
                        if (mak + macroK == a.bufK)
                            _convolutions[1](tmp + tmpOffs, p, a, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 1 : 0,
                                weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
                        else
                            _convolutions[0](tmp + tmpOffs, p, a, macroD, yEnd - yBeg, macroK, mak == 0 ? 1 : 0,
                                weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
                        yBeg = yEnd;
                    }
                    weight += macroK * a.microD;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
                if (!a.sumBuf)
                    sum += macroD;
            }
        }

        bool SynetConvolution16bNhwcGemmV2::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }
    }
#endif
}
