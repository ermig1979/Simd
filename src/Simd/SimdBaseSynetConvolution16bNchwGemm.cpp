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
        typedef Base::SynetConvolution16bNchwGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNchwGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const float* src = ((float*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (j = 0; j < NF; j += a.F)
            {
                for (k = 0; k < K2; k += 2)
                {
                    const float* src0 = src + k * dS, * src1 = src0 + dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = Float32ToBFloat16(src0[f]);
                        *dst++ = Float32ToBFloat16(src1[f]);
                    }
                }
                for (; k < K; k += 2)
                {
                    const float* src0 = src + k * dS, * src1 = src0 + dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = Float32ToBFloat16(src0[f]);
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                src += a.F;
            }
            if (j < N)
            {
                size_t tail = N - j, f;
                for (k = 0; k < K2; k += 2)
                {
                    const float* src0 = src + k * dS, * src1 = src0 + dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = Float32ToBFloat16(src0[f]);
                        *dst++ = Float32ToBFloat16(src1[f]);
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < K; k += 2)
                {
                    const float* src0 = src + k * dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = Float32ToBFloat16(src0[f]);
                        *dst++ = 0;
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
            }
        }

        static void Reorder16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const uint16_t* src = ((uint16_t*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (j = 0; j < NF; j += a.F)
            {
                for (k = 0; k < K2; k += 2)
                {
                    const uint16_t* src0 = src + k * dS, * src1 = src0 + dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = src1[f];
                    }
                }
                for (; k < K; k += 2)
                {
                    const uint16_t* src0 = src + k * dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                src += a.F;
            }
            if (j < N)
            {
                size_t tail = N - j, f;
                for (k = 0; k < K2; k += 2)
                {
                    const uint16_t* src0 = src + k * dS, * src1 = src0 + dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = src1[f];
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < K; k += 2)
                {
                    const uint16_t* src0 = src + k * dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = 0;
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNchwGemm::SynetConvolution16bNchwGemm(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
            if (_src16b)
            {
                if (_is1x1)
                    _convert = Reorder16bNchwGemm1x1;
                //else
                //    _convert = Reorder16bNhwcGemm;
            }
            else
            {
                if (_is1x1)
                    _convert = Convert16bNchwGemm1x1;
                //else
                //    _convert = Convert16bNhwcGemm;
            }
        }

        String SynetConvolution16bNchwGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NchwGemm";
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetConvolution16bNchwGemm::SetAlgParam(size_t F, size_t microD, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.N = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = F;
            a.microD = microD;
            a.microN = microN;
            a.microK = microK;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD / 2, a.microK), a.microK, a.bufK);
            a.macroH = Simd::RestrictRange(L3 / a.macroK / p.dstW / 2, size_t(1), p.dstH);
            a.macroD = Simd::RestrictRange(AlignLoAny(L2 / a.macroK / 2, a.microD), a.microD, a.bufD);
            a.bufN = p.dstH * AlignHi(p.dstW, a.F);
            a.elem = _elemD;
            a.reorderType = 0;
            a.sumBuf = (_dst16b && a.macroK < a.K) || a.microK > 2 ? 1 : 0;
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;

            _stepS = p.srcH * p.srcW * p.srcC * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * _elemD;
        }

        size_t SynetConvolution16bNchwGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = a.bufN * a.bufK * sizeof(uint16_t);
            if (a.sumBuf)
                size += a.macroD * a.bufN * sizeof(float);
            return size;
        }

        void SynetConvolution16bNchwGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNchwGemm::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.bufK * a.bufD, true);
            uint16_t* dst = _weight.data;
            for (size_t mak = 0; mak < a.bufK; mak += a.macroK)
            {
                size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                for (size_t d = 0; d < a.bufD; d += 1)
                {
                    const float* src = weight + d * a.K + mak;
                    for (size_t k = 0; k < macroK; k += 1)
                    {
                        if (d < p.dstC && mak + k < a.K)
                            *(dst++) = Float32ToBFloat16(src[k]);
                        else
                            *(dst++) = 0;
                    }
                }
            }
        }

        void SynetConvolution16bNchwGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* bufB = Allocate<uint16_t>(buf8, a.bufN * a.bufK);
            float* bufS = a.sumBuf ? Allocate<float>(buf8, a.macroD * a.bufN) : NULL;
            for (size_t b = 0; b < p.batch; b += 1)
            {
                uint16_t* buf = _convert ? bufB : (uint16_t*)src;
                float* sum = a.sumBuf ? bufS : (float*)dst;
                Forward(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNchwGemm::Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            for (size_t yBeg = 0; yBeg < p.dstH;)
            {
                size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                if(!_is1x1)
                    _convert(src, p, a, yBeg, yEnd, 0, p.srcC, buf);
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    if (_is1x1)
                        _convert(src, p, a, yBeg, yEnd, mak, mak + macroK, buf);
                    size_t bufOffs = _is1x1 ? 0 : mak * a.F;
                    for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
                    {
                        size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                        size_t sumOffs = a.macroK < a.bufK ? (dc * p.dstH + yBeg) * p.dstW : 0;
                        size_t dstOffs = (dc * p.dstH + yBeg) * p.dstW * _elemD;
                        const uint16_t* weight = _weight.data + a.bufD * mak + dc * macroK;
                        if (mak + macroK == a.bufK)
                            _convolutions[1](weight, p, a, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 1 : 0,
                                buf + bufOffs, bias, params, sum + sumOffs, dst + dstOffs);
                        else
                            _convolutions[0](weight, p, a, macroD, yEnd - yBeg, macroK, mak == 0 ? 1 : 0,
                                buf + bufOffs, bias, params, sum + sumOffs, dst + dstOffs);
                    }
                }
                yBeg = yEnd;
            }
        }

        bool SynetConvolution16bNchwGemm::Preferable(const ConvParam& p)
        {
            return p.trans == 0 && p.group == 1 && Is1x1(p);
        }
    }
#endif
}
