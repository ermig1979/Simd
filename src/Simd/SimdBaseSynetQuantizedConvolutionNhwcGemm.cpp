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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
        }

        String SynetQuantizedConvolutionNhwcGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemm";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcGemm::SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3)
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
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD, a.microK), a.microK, a.bufK);
            a.batch = 1;
            size_t bufSize = a.M * a.bufK;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2 && (a.bufK > 512 || microK == 4 || batch * a.M <= 32 * microM))
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroK, a.microD), a.microD, a.bufD);
            a.bufM = a.batch * p.dstH * AlignHi(p.dstW, a.F);
            a.elem = _elemD;
            a.reorderType = 0;
            a.sumBuf = (_dst8u && a.macroK < a.K) || a.microK > 4 ? 1 : 0;
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;
            a.dB = (a.sumBuf ? a.macroD : p.dstC);

            _merge = a.batch;
        }

        size_t SynetQuantizedConvolutionNhwcGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            if (_convert)
                size += a.bufM * a.bufK * sizeof(uint8_t);
            if (a.sumBuf)
                size += a.macroD * a.bufM * sizeof(int32_t);
            return size;
        }

        void SynetQuantizedConvolutionNhwcGemm::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t D = DivHi(p.dstC, _alg.F);
            _weight.Resize(a.bufK * a.bufD, true);
            int8_t* dst = _weight.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t k = 0; k < a.bufK; k += 4)
                {
                    const int8_t* src = weight + k * p.dstC + d * _alg.F;
                    for (size_t f = 0; f < _alg.F; ++f)
                    {
                        for (size_t i = 0; i < 4; ++i)
                        {
                            if (d * _alg.F + f < p.dstC && k + i < a.K)
                                *(dst++) = src[i * p.dstC];
                            else
                                *(dst++) = 0;
                        }
                        src++;
                    }
                }
            }
        }

        void SynetQuantizedConvolutionNhwcGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* bufB = _convert ? Allocate<uint8_t>(buf8, a.bufM * a.bufK) : NULL;
            int32_t* bufS = a.sumBuf ? Allocate<int32_t>(buf8, a.macroD * a.bufM) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint8_t* buf = _convert ? bufB : (uint8_t*)src;
                int32_t* sum = a.sumBuf ? bufS : (int32_t*)dst;
                Forward(src, buf, sum, dst);
                src += _sizeS * a.batch * _elemS;
                dst += _sizeD * a.batch * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcGemm::Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int32_t* sBias = _bias.data;
            const float* sNorm = _norm.data;
            const float* params = _params.data;
            float dNorm = 1.0f / _dstScale;
            size_t dstH = p.dstH * a.batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                const int8_t* weight = _weight.data + dc * a.bufK;
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    for (size_t yBeg = 0; yBeg < dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                        size_t bufOffs = (a.macroK < a.bufK || _convert == NULL) ?
                            yBeg * (_convert ? AlignHi(p.dstW, a.F) : p.dstW) * a.bufK + (a.reorderType ? mak * a.F : mak) : 0;
                        size_t sumOffs = a.macroK < a.bufK ? yBeg * (a.microK > 4 ? AlignHi(p.dstW, a.F) : p.dstW) * a.dB : 0;
                        size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                        if (dc == 0 && mak == 0 && _convert)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = p.dstH * p.dstW * a.bufK;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _convert(src + b * dS, _srcZero[0], p, a, 0, p.dstH, buf + b * dB);
                            }
                            else
                                _convert(src, _srcZero[0], p, a, yBeg, yEnd, buf + bufOffs);
                        }
                        if (mak + macroK == a.bufK)
                            _convolutions[1](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 0 : 1,
                                weight, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero, sum + sumOffs, dst + dstOffs);
                        else
                            _convolutions[0](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, mak == 0 ? 0 : 1,
                                weight, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero, sum + sumOffs, dst + dstOffs);
                        yBeg = yEnd;
                    }
                    weight += macroK * a.F;
                }
                sBias += macroD;
                sNorm += macroD;
                if (p.activation == SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
                if (!a.sumBuf)
                    sum += macroD;
            }
        }

        bool SynetQuantizedConvolutionNhwcGemm::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }
    }
#endif
}
