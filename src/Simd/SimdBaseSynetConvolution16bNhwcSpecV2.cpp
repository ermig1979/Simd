/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
        SynetConvolution16bNhwcSpecV2::SynetConvolution16bNhwcSpecV2(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _preprocess = 0;
            _bodyConv = 0;
            _lastConv = 0;
        }

        String SynetConvolution16bNhwcSpecV2::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV2";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetConvolution16bNhwcSpecV2::SetAlgParam(size_t F, size_t microD, size_t microS, size_t microC, size_t L1, size_t L2, size_t L3)
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
            a.dstC = AlignHi(p.dstC, a.microD);
            a.kA = p.kernelX * p.kernelY;
            a.K = a.srcC * a.kA;
            a.padE = a.srcW * a.padV + a.padH * Simd::Max<size_t>(1, a.padV);

            a.macroC = Simd::RestrictRange(AlignLo(L1 / a.microD / a.kA / 2, a.microC), a.microC, a.srcC);
            a.batch = 1;
            size_t bufSize = a.srcC * a.srcH * a.srcW * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroC / a.srcW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / a.kA / 2, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));

            a.bufD = AlignHi(a.batch * a.srcH * a.srcW, a.microS) * a.macroD;

            a.elem = _elemD;
            a.bufS = (a.batch * a.srcH * a.srcW + a.padE + a.microS) * a.srcC;

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;

            int dX = (int)a.microC, dY = (int)a.srcW * dX, dC = int(a.batch * a.srcH * a.srcW + a.padE) * dX;
            _srcOffs.Resize(DivHi(a.K, a.microC));
            for (size_t c = 0, offsS = 0, i = 0; c < a.srcC; c += dX, offsS += dC)
                for (size_t y = 0, offsY = offsS; y < p.kernelY; y += 1, offsY += dY)
                    for (size_t offsX = offsY, endX = offsY + p.kernelX * dX; offsX < endX; offsX += dX, i++)
                        _srcOffs[i] = (int)offsX;

            _dstMask.Resize(AlignHi((a.srcH * a.batch - a.gapV) * a.srcW - a.padH, a.microS));
            size_t i = 0;
            for (size_t b = 0; b < a.batch; b++)
            {
                for (size_t y = 0; y < p.dstH; y++)
                {
                    for (size_t x = 0; x < p.dstW; x++, i++)
                        _dstMask[i] = -1;
                    for (size_t x = 0; x < a.gapH; x++, i++)
                        _dstMask[i] = 0;
                }
                for (size_t y = 0, gapI = a.gapV * a.srcW; y < gapI && i < _dstMask.size; y++, i++)
                    _dstMask[i] = 0;
            }
            for (; i < _dstMask.size; i++)
                _dstMask[i] = 0;

            _nK.Resize(DivHi(a.srcC, a.macroC));
            for (size_t o = 0, c = 0; o < _nK.size; o++, c += a.macroC)
            {
                size_t macroC = Simd::Min(a.srcC, c + a.macroC) - c;
                _nK[o] = int(DivHi(macroC, a.microC) * a.kA);
            }
            if (_nK.size > 1 && _nK[_nK.size - 1] < _nK[_nK.size - 2])
                Simd::Swap(_nK[_nK.size - 1], _nK[_nK.size - 2]);

            size_t n = DivHi(a.batch * p.dstH, a.macroH);
            _maBufOffs.Resize(n);
            _maSumOffs.Resize(n + 1);
            _miDstOffs.Resize(DivHi(_dstMask.size, a.microS));
            for (size_t i = 0; i <= n; ++i)
            {
                size_t dy = i * a.macroH;
                if (i == 0)
                {
                    _maBufOffs[i] = 0;
                    _maSumOffs[i] = 0;
                }
                else if (i == n)
                    _maSumOffs[i] = int((a.srcH * a.batch - a.gapV) * a.srcW - a.padH);
                else
                {
                    size_t sumOffs = dy * a.srcW - a.gapH;
                    _maSumOffs[i] = int(AlignLo(sumOffs, a.microS));
                    _maBufOffs[i] = int(dy * a.srcW + _maSumOffs[i] - sumOffs);
                }
            }
            _miDstOffs[0] = 0;
            for (size_t i = 1; i < _miDstOffs.size; ++i)
            {
                _miDstOffs[i] = _miDstOffs[i - 1];
                for (size_t j = (i - 1) * a.microS, m = i * a.microS; j < m; ++j)
                    if (_dstMask[j])
                        _miDstOffs[i]++;
            }
        }

        size_t SynetConvolution16bNhwcSpecV2::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufS * sizeof(uint16_t);
            size += a.bufD * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcSpecV2::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcSpecV2::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.K * a.dstC, true);
            uint16_t* dst = _weight.data;
            const size_t microC = a.microC, microD = a.microD;
            for (size_t mad = 0; mad < p.dstC; mad += microD)
            {
                for (size_t mac = 0; mac < p.srcC; mac += microC)
                {
                    for (size_t k = 0; k < a.kA; k++)
                    {
                        for (size_t c = 0; c < microC; c += 2)
                        {
                            const float* src = weight + (k * p.srcC + mac + c) * p.dstC + mad;
                            for (size_t d = 0; d < microD; ++d)
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

        void SynetConvolution16bNhwcSpecV2::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
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
                if(a.batch == 1)
                    ForwardSingle(src, buf, sum, dst);
                else
                    ForwardBatch(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcSpecV2::ForwardSingle(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            size_t dS = a.microC, dB = a.macroD, dD = p.dstC * _elemD;
            size_t bufOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * dS;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const uint16_t* weight = _weight.data + mad * a.K;
                const int* srcOffs = _srcOffs.data;
                for (size_t nk = 0; nk < _nK.size; ++nk)
                {
                    int zero = nk == 0 ? 1 : 0;
                    size_t nK = _nK[nk];
                    for (size_t dyBeg = 0, dyN = 0; dyBeg < p.dstH; dyN++)
                    {
                        size_t dyEnd = Simd::Min(dyBeg + a.macroH, p.dstH);
                        size_t dstS = _maSumOffs[dyN + 1] - _maSumOffs[dyN];
                        size_t miIdx = _maSumOffs[dyN] / a.microS;
                        if (mad == 0 && zero)
                            _preprocess(src, p, a, dyBeg, dyEnd, dyEnd == p.dstH ? 1 : 0, buf);
                        if (nk == _nK.size - 1)
                            _lastConv(buf + bufOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, zero, weight,
                                sum + _maSumOffs[dyN] * dB, bias, params, _dstMask.data + _maSumOffs[dyN], _miDstOffs.data + miIdx, dst + _miDstOffs[miIdx] * dD);
                        else
                            _bodyConv(buf + bufOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, zero, weight, sum + _maSumOffs[dyN] * dB);
                        dyBeg = dyEnd;
                    }
                    srcOffs += nK;
                    weight += nK * a.microC * a.microD;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        void SynetConvolution16bNhwcSpecV2::ForwardBatch(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const float* bias = _bias.data, * params = _params.data;
            const int* mask = _dstMask.data;
            size_t dstH = p.dstH * a.batch, dstS = _maSumOffs[1] - _maSumOffs[0];
            size_t bufOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * a.microC;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const uint16_t* weight = _weight.data + mad * a.K;
                const int* srcOffs = _srcOffs.data;
                for (size_t nk = 0; nk < _nK.size; ++nk)
                {
                    int zero = nk == 0 ? 1 : 0;
                    size_t nK = _nK[nk];
                    if (mad == 0 && zero)
                    {
                        size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                        size_t dB = a.srcH * a.srcW * a.microC;
                        for (size_t b = 0; b < a.batch; ++b)
                            _preprocess(src + b * dS, p, a, 0, p.dstH, b == a.batch - 1 ? 1 : 0, buf + b * dB);
                    }
                    if (nk == _nK.size - 1)
                        _lastConv(buf + bufOffs, p, a, srcOffs, macroD, dstS, nK, zero, weight, sum, bias, params, mask, _miDstOffs.data, dst);
                    else
                        _bodyConv(buf + bufOffs, p, a, srcOffs, macroD, dstS, nK, zero, weight, sum);
                    srcOffs += nK;
                    weight += nK * a.microC * a.microD;
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetConvolution16bNhwcSpecV2::Preferable(const ConvParam& p)
        {
            static int choise = 0;
            return 1 && p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) && p.dstC >= 4
                ;// && (choise++) & 0;
        }
    }
#endif
}
