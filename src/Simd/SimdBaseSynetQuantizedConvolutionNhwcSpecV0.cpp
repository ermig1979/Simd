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
        SynetQuantizedConvolutionNhwcSpecV0::SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _convolution = 0;
            _postprocess = 0;
        }

        String SynetQuantizedConvolutionNhwcSpecV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV0";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcSpecV0::SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microD = microD;
            a.microS = microS;
            a.microC = microC;
            a.srcC = AlignHi(p.srcC, a.microC);
            a.padV = Simd::Max(p.padY, p.padH);
            a.padH = Simd::Max(p.padX, p.padW);
            a.srcH = p.srcH + a.padV;
            a.srcW = p.srcW + a.padH;
            a.gapV = a.srcH - p.dstH;
            a.gapH = a.srcW - p.dstW;
            a.dstC = AlignHi(p.dstC, a.F);
            a.kA = p.kernelX * p.kernelY;
            a.K = a.srcC * a.kA;
            a.padE = a.srcW * a.padV + a.padH * Simd::Max<size_t>(1, a.padV) + a.microC;

            a.macroC = Simd::RestrictRange(AlignLo(L1 / a.microD / a.kA, a.microC), a.microC, a.srcC);
            a.batch = 1;
            size_t bufSize = a.srcC * a.srcH * a.srcW;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2 && (microC == 4 || batch * a.srcH * a.srcW <= 32 * microS))
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroC / a.srcW, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / a.kA, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));

            a.numH = DivHi(p.dstH * a.batch, a.macroH);
            a.bufD = (a.batch * a.srcH * a.srcW + a.numH * a.F) * a.macroD;

            a.macroO = DivHi(a.macroC, a.microC) * a.kA;

            a.elem = _elemD;
            a.bufS = (a.batch * a.srcH * a.srcW + a.padE) * a.srcC + a.microC * a.F;
            _merge = a.batch;

            int dX = (int)a.microC, dY = (int)a.srcW * dX, dC = int(a.batch * a.srcH * a.srcW + a.padE) * dX;
            _offset.Resize(DivHi(a.K, a.microC));
            for (size_t c = 0, offsS = 0, i = 0; c < a.srcC; c += dX, offsS += dC)
                for (size_t y = 0, offsY = offsS; y < p.kernelY; y += 1, offsY += dY)
                    for (size_t offsX = offsY, endX = offsY + p.kernelX * dX; offsX < endX; offsX += dX, i++)
                        _offset[i] = (int)offsX;
        }

        size_t SynetQuantizedConvolutionNhwcSpecV0::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufS * sizeof(uint8_t);
            size += a.bufD * sizeof(int32_t);
            return size;
        }

        size_t SynetQuantizedConvolutionNhwcSpecV0::InternalBufferSize() const
        {
            return SynetQuantizedConvolution::InternalBufferSize() + _offset.RawSize();
        }

        void SynetQuantizedConvolutionNhwcSpecV0::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.K * a.dstC, true);
            int8_t* dst = _weight.data;
            for (size_t mad = 0; mad < p.dstC; mad += _alg.F)
            {
                for (size_t mac = 0; mac < p.srcC; mac += a.microC)
                {
                    for (size_t k = 0; k < a.kA; k++)
                    {
                        for (size_t c = 0; c < a.microC; c += 4)
                        {
                            const int8_t* src = weight + (k * p.srcC + mac + c) * p.dstC + mad;
                            for (size_t d = 0; d < a.F; ++d)
                            {
                                for (size_t i = 0; i < 4; ++i)
                                {
                                    if (mad + d < p.dstC && mac + c + i < p.srcC)
                                        *(dst++) = src[i * p.dstC];
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

        void SynetQuantizedConvolutionNhwcSpecV0::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* bufS = a.bufS ? Allocate<uint8_t>(buf8, a.bufS) : NULL;
            int32_t* bufD = a.bufD ? Allocate<int32_t>(buf8, a.bufD) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                uint8_t* buf = bufS ? bufS : (uint8_t*)src;
                int32_t* sum = bufD ? bufD : (int32_t*)dst;
                Forward(src, buf, sum, dst);
                src += _sizeS * a.batch * _elemS;
                dst += _sizeD * a.batch * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcSpecV0::Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int32_t* sBias = _bias.data;
            const float* sNorm = _norm.data;
            const int* offs = _offset.data;
            const float* params = _params.data;
            float dNorm = 1.0f / _dstScale;
            size_t dstH = p.dstH * a.batch, dstHb = a.srcH * a.batch - a.gapV;
            size_t bufOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * a.microC;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const int8_t* weight = _weight.data + mad * a.K;
                for (size_t mac = 0, mao = 0; mac < a.srcC; mac += a.macroC, mao += a.macroO)
                {
                    size_t macroC = Simd::Min(a.srcC, mac + a.macroC) - mac;
                    size_t nK = DivHi(macroC, a.microC) * a.kA;
                    for (size_t dyBeg = 0, dyN = 0; dyBeg < dstH; dyN++)
                    {
                        size_t dyEnd = Simd::Min(dyBeg + a.macroH, dstH);
                        if (mad == 0 && mac == 0)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                size_t dB = a.srcH * a.srcW * a.microC;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _preprocess(src + b * dS, _srcZero[0], p, a, 0, p.dstH, b == a.batch - 1 ? 1 : 0, buf + b * dB);
                            }
                            else
                                _preprocess(src, _srcZero[0], p, a, dyBeg, dyEnd, dyEnd == dstH ? 1 : 0, buf);
                        }
                        if (a.batch > 1)
                        {
                            _convolution(buf + bufOffs, p, a, offs + mao, macroD, dstHb, nK, mac == 0 ? 0 : 1, weight, sum);
                        }
                        else
                        {
                            _convolution(buf + bufOffs + dyBeg * a.srcW * a.microC, p, a, offs + mao, macroD, dyEnd - dyBeg,
                                nK, mac == 0 ? 0 : 1, weight, sum + (dyBeg * a.srcW + dyN * a.F) * a.macroD);
                        }
                        if (mac + macroC == a.srcC)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = a.srcH * a.srcW * a.macroD;
                                size_t dD = p.dstH * p.dstW * p.dstC * a.elem;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _postprocess(sum + b * dS, p, a, macroD, 0, p.dstH, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero, dst + b * dD);
                            }
                            else
                                _postprocess(sum + dyN * a.F * a.macroD, p, a, macroD, dyBeg, dyEnd, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero, dst);
                        }
                        dyBeg = dyEnd;
                    }
                    weight += macroC * a.kA * a.F;
                }
                sBias += macroD;
                sNorm += macroD;
                if (p.activation == SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetQuantizedConvolutionNhwcSpecV0::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) 
                && p.dstC >= 4 && p.srcC * p.kernelX * p.kernelY >= 32;
        }
    }
#endif
}
