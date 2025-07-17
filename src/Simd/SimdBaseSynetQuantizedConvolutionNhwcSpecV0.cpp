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
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
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
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
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
            //const ConvParam& p = _param;
            //const AlgParam& a = _alg;
            //const int32_t* bias = _bias.data;
            //const float* norm = _norm.data;
            //int32_t zero = _dstZero[0];
            //size_t dstH = p.dstH * a.batch;
            //for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            //{
            //    size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
            //    const int8_t* weight = _weight.data + dc * a.bufK;
            //    for (size_t mak = 0; mak < a.K; mak += a.macroK)
            //    {
            //        size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
            //        for (size_t yBeg = 0; yBeg < dstH;)
            //        {
            //            size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
            //            size_t bufOffs = (a.macroK < a.bufK || _convert == NULL) ?
            //                yBeg * (_convert ? AlignHi(p.dstW, a.F) : p.dstW) * a.bufK + (a.reorderType ? mak * a.F : mak) : 0;
            //            size_t sumOffs = a.macroK < a.bufK ? yBeg * (a.microK > 4 ? AlignHi(p.dstW, a.F) : p.dstW) * a.dB : 0;
            //            size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
            //            if (dc == 0 && mak == 0 && _convert)
            //            {
            //                if (a.batch > 1)
            //                {
            //                    size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
            //                    size_t dB = p.dstH * p.dstW * a.bufK;
            //                    for (size_t b = 0; b < a.batch; ++b)
            //                        _convert(src + b * dS, _srcZero[0], p, a, 0, p.dstH, buf + b * dB);
            //                }
            //                else
            //                    _convert(src, _srcZero[0], p, a, yBeg, yEnd, buf + bufOffs);
            //            }
            //            if (mak + macroK == a.bufK)
            //                _convolutions[1](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 0 : 1,
            //                    weight, bias, norm, zero, sum + sumOffs, dst + dstOffs);
            //            else
            //                _convolutions[0](buf + bufOffs, p, a, macroD, yEnd - yBeg, macroK, mak == 0 ? 0 : 1,
            //                    weight, bias, norm, zero, sum + sumOffs, dst + dstOffs);
            //            yBeg = yEnd;
            //        }
            //        weight += macroK * a.F;
            //    }
            //    bias += macroD;
            //    norm += macroD;
            //    dst += macroD * _elemD;
            //    if (!a.sumBuf)
            //        sum += macroD;
            //}
        }

        bool SynetQuantizedConvolutionNhwcSpecV0::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) && p.dstC >= 4;
        }
    }
#endif
}
