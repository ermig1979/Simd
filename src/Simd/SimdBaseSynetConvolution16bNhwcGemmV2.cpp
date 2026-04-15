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
            _conv1x1 = 0;
            _convAny = 0;
            _gemm[0] = 0;
            _gemm[1] = 0;
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
                    if (batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.batch = DivHi(p.batch, DivHi(p.batch, a.batch));
            a.macroM = Simd::RestrictRange(AlignLoAny(L2 / a.macroK / 2, a.microM), a.microM, a.batch * a.M);
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroK / 2, a.microD), a.microD, a.bufD);
            if (_is1x1)
            {
                if (a.macroK == a.bufK && a.macroD == a.bufD && a.macroM > 256)
                    a.macroM = 256;
            }
            else
            {
                a.isAlMaH = (a.macroH == p.dstH * a.batch || Aligned(a.macroH * p.dstW, a.microM))
                    && (a.batch == 1 || Aligned(p.dstH * p.dstW, a.microM));
                if (!a.isAlMaH && a.batch == 1)
                {

                    size_t hAlign = a.microM / Pow2Divider(p.dstW);
                    //std::cout << " a.macroH " << a.macroH << " hAlign " << hAlign << std::endl;
                    if (hAlign < a.macroH)
                    {
                        a.macroH = AlignLo(a.macroH, hAlign);
                        if (a.macroK == a.bufK && a.macroD == a.bufD && a.macroH * p.dstW > 256)
                            a.macroH = Simd::Min(a.macroH, AlignHi(DivHi(256, p.dstW), hAlign));
                        a.isAlMaH = Aligned(a.macroH * p.dstW, a.microM);
                    }
                }
            }
            a.bufM = AlignHi(a.batch * p.dstH * p.dstW, a.F);
            a.elem = _elemD;
            a.tmpBuf = !_src16b || !_is1x1 || a.K != a.bufK;
            a.sumBuf = (_dst16b || a.bufD != p.dstC || a.bufM != a.batch * p.dstH * p.dstW) && a.macroK < a.bufK;
            a.reorderType = a.tmpBuf != 0 && (_is1x1 || (Aligned(p.srcC, a.microK) && a.isAlMaH));
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;
            a.dB = (a.sumBuf ? a.macroD : p.dstC);

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;

            //std::cout << " p.srcC " << p.srcC << std::endl;
            //std::cout << " a.bufK " << a.bufK << std::endl;
            //std::cout << " p.dstC " << p.dstC << std::endl;
            //std::cout << " a.macroK " << a.macroK << std::endl;
            //std::cout << " a.macroD " << a.macroD << std::endl;
            //std::cout << " a.macroM " << a.macroM << std::endl;
            //std::cout << " a.macroH " << a.macroH << std::endl;
            //std::cout << " p.dstW " << p.dstW << std::endl;
            //std::cout << " a.tmpBuf " << a.tmpBuf << std::endl;
            //std::cout << " a.sumBuf " << a.sumBuf << std::endl;
            //std::cout << " a.reorderType " << a.reorderType << std::endl;
            //std::cout << " a.isAlMaH " << a.isAlMaH << std::endl;
        }

        size_t SynetConvolution16bNhwcGemmV2::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 2048 * sizeof(float);
            if(a.tmpBuf)
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

        void SynetConvolution16bNhwcGemmV2::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufT = a.tmpBuf ? Allocate<uint16_t>(buf8, a.bufM * a.bufK) : NULL;
            float* bufS = a.sumBuf ? Allocate<float>(buf8, a.macroD * a.bufM) : NULL;
            float* bufD = Allocate<float>(buf8, 2048);
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint16_t* tmp = a.tmpBuf ? bufT : (uint16_t*)src;
                float* sum = a.sumBuf ? bufS : (float*)dst;
                size_t batch = Simd::Min(p.batch, b + a.batch) - b;
                if(_is1x1)
                    Forward1x1(src, tmp, batch, sum, bufD, dst);
                else
                    ForwardAny(src, tmp, batch, sum, bufD, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemmV2::Forward1x1(const uint8_t* src, uint16_t* tmp, size_t batch, float* sum, float* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t M = batch * p.dstH * p.dstW;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                const uint16_t* weight = _weight.data + dc * a.bufK;
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    for (size_t i = 0; i < M; i += a.macroM)
                    {
                        size_t macroM = Simd::Min(M, i + a.macroM) - i;
                        size_t tmpOffs = (a.macroK == a.bufK && a.tmpBuf) ? 0 : i * a.bufK + (a.reorderType ? mak * a.F : mak);
                        size_t sumOffs = a.macroK < a.bufK ? i * a.dB : 0;
                        size_t dstOffs = i * p.dstC * _elemD;
                        if (dc == 0 && mak == 0 && a.tmpBuf)
                        {
                            size_t srcOffs = (i * p.srcC + mak) * _elemS;
                            _conv1x1(src + srcOffs, p, a, macroM, tmp + tmpOffs);
                        }
                        if (mak + macroK == a.bufK)
                            _gemm[1](tmp + tmpOffs, p, a, macroD, macroM, macroK, mak == 0 ? 1 : 0, weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
                        else
                            _gemm[0](tmp + tmpOffs, p, a, macroD, macroM, macroK, mak == 0 ? 1 : 0, weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
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

        void SynetConvolution16bNhwcGemmV2::ForwardAny(const uint8_t* src, uint16_t* tmp, size_t batch, float* sum, float* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                const uint16_t* weight = _weight.data + dc * a.bufK;
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    for (size_t yBeg = 0, i = 0; yBeg < dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                        size_t tmpOffs = (a.macroK == a.bufK && a.isAlMaH) ? 0 : i * a.bufK + (a.reorderType ? mak * a.F : mak);
                        size_t sumOffs = a.macroK < a.bufK ? i * a.dB : 0;
                        size_t dstOffs = i * p.dstC * _elemD;
                        if (dc == 0 && mak == 0 && a.tmpBuf)
                        {
                            if (batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = p.dstH * p.dstW * a.bufK;
                                for (size_t b = 0; b < batch; ++b)
                                    _convAny(src + b * dS, p, a, 0, p.dstH, tmp + b * dB);
                            }
                            else
                            {
                                size_t tmpOffs = (a.macroK == a.bufK && a.isAlMaH) ? 0 : yBeg * p.dstW * a.bufK + (a.reorderType ? mak * a.F : mak);
                                _convAny(src, p, a, yBeg, yEnd, tmp + tmpOffs);
                            }
                        }
                        size_t macroM = yEnd * p.dstW - i;
                        if (yEnd < dstH)
                            macroM = AlignLo(macroM, a.microM);
                        if (mak + macroK == a.bufK)
                            _gemm[1](tmp + tmpOffs, p, a, macroD, macroM, macroK, mak == 0 ? 1 : 0, weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
                        else
                            _gemm[0](tmp + tmpOffs, p, a, macroD, macroM, macroK, mak == 0 ? 1 : 0, weight, bias, params, sum + sumOffs, buf, dst + dstOffs);
                        yBeg = yEnd;
                        i += macroM;
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

        bool SynetConvolution16bNhwcGemmV2::Preferable(const ConvParam& p)
        {
            static int choise = 1;
            size_t K = p.srcC * p.kernelX * p.kernelY, M = p.batch * p.dstH * p.dstW;
            return p.trans != 0 && p.group == 1 
                && (!p.IsKernel(1) || K < 32 || K > 128 || M < 16)
                && 1;// ((choise++) & 1);
        }
    }
#endif
}
