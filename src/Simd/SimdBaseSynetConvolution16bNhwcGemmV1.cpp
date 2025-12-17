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
#include "Simd/SimdMemory.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution16bNhwcGemmV1::SynetConvolution16bNhwcGemmV1(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _convert = 0;
            _convolution = 0;
        }

        String SynetConvolution16bNhwcGemmV1::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemmV1";
            desc << "-" << _alg.microM / 16 << "x" << _alg.microD / 16;
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetConvolution16bNhwcGemmV1::SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3)
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
            a.batch = 1;
            size_t bufSize = a.M * a.bufK * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.bufK / p.dstW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.bufK / 2, a.microD), a.microD, a.bufD);
            a.bufM = a.batch * p.dstH * AlignHi(p.dstW, a.F);
            a.elem = _elemD;

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;
        }

        size_t SynetConvolution16bNhwcGemmV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            if(_convert)
                size += a.bufM * a.bufK * sizeof(uint16_t);
            size += a.microD * a.microM * sizeof(float) * 2;
            return size;
        }

        void SynetConvolution16bNhwcGemmV1::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcGemmV1::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t F = _alg.microD, D = DivHi(p.dstC, F);
            _weight.Resize(a.bufK * a.bufD, true);
            uint16_t* dst = _weight.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t k = 0; k < a.bufK; k += 2)
                {
                    const float* src = weight + k * p.dstC + d * F;
                    for (size_t f = 0; f < F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (d * F + f < p.dstC && k + i < a.K)
                                *(dst++) = Float32ToBFloat16(src[i * p.dstC]);
                            else
                                *(dst++) = 0;
                        }
                        src++;
                    }
                }
            }
        }

        void SynetConvolution16bNhwcGemmV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufB = _convert ? Allocate<uint16_t>(buf8, a.bufM * a.bufK) : NULL;
            float* bufS = Allocate<float>(buf8, a.microD * a.microM);
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* buf = _convert ? bufB : (uint16_t*)src;
                Forward(src, buf, bufS, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemmV1::Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * a.batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                const uint16_t* weight = _weight.data + dc * a.bufK;
                for (size_t yBeg = 0; yBeg < dstH;)
                {
                    size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                    size_t bufOffs = (_convert == NULL || a.macroD < p.dstC) ? yBeg * (_convert ? AlignHi(p.dstW, a.F) : p.dstW) * a.bufK : 0;
                    size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                    if (dc == 0 && _convert)
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
                    _convolution(buf + bufOffs, p, a, macroD, yEnd - yBeg, weight, bias, params, sum, dst + dstOffs);
                    yBeg = yEnd;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetConvolution16bNhwcGemmV1::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1 && 1 &&
                ((Aligned(p.dstC, 64) && p.dstH * p.dstW >= 16 && p.srcC >= 64 && p.dstT == SimdTensorData16b) ||
                    (p.srcC >= 128 && p.dstT == SimdTensorData16b) || 
                    (p.srcC >= 256 && p.dstT == SimdTensorData32f)) && p.srcC <= 512;
        }
    }
#endif
}
