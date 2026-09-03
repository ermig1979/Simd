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
        SynetQuantizedConvolutionNhwcSpecV1::SynetQuantizedConvolutionNhwcSpecV1(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _bodyConv = 0;
            _lastConv = 0;
        }

        String SynetQuantizedConvolutionNhwcSpecV1::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV1";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNhwcSpecV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 2048 * sizeof(int32_t);
            size += a.bufS * sizeof(uint8_t);
            size += a.bufD * sizeof(int32_t);
            return size;
        }

        size_t SynetQuantizedConvolutionNhwcSpecV1::InternalBufferSize() const
        {
            return SynetQuantizedConvolution::InternalBufferSize() + _srcOffs.RawSize() + _dstMask.RawSize() + 
                _nK.RawSize() + _maBufOffs.RawSize() + _maSumOffs.RawSize() + _miDstOffs.RawSize();
        }

        void SynetQuantizedConvolutionNhwcSpecV1::SetAlgParam()
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            int L1 = int(Base::AlgCacheL1() * (p.IsKernel(5) ? 1.05 : 1.00)), L2 = int(Base::AlgCacheL2() * 0.5), L3 = int(Base::AlgCacheL3());

            a.F = 16;
            a.microD = 32;
            a.microS = 32;
            a.microC = 64;
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

            a.macroC = Simd::RestrictRange(AlignLo(L1 / a.microD / a.kA, a.microC), a.microC, a.srcC);
            a.batch = 1;
            size_t bufSize = a.srcC * a.srcH * a.srcW;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)// && batch * a.srcH * a.srcW <= 32 * microS
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.macroC / a.srcW, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / a.kA, a.microD), a.microD, a.dstC);
            if(a.macroD < a.dstC)
                a.macroD = Simd::Min<size_t>(a.macroD, a.microD * 4);

            a.elem = _elemD;
            a.bufS = (a.batch * a.srcH * a.srcW + a.padE + a.microS) * a.srcC;
            a.bufD = a.macroC < a.srcC ? AlignHi(a.batch * a.srcH * a.srcW, a.microS) * a.macroD : 0;

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
                if (i == n)
                    _maSumOffs[i] = int((a.srcH * a.batch - a.gapV) * a.srcW - a.padH);
                else
                {
                    size_t dy = i * a.macroH;
                    size_t sumOffs = Simd::Max<ptrdiff_t>(dy * a.srcW - a.gapH, 0);
                    _maSumOffs[i] = int(AlignLo(sumOffs, a.microS));
                    _maBufOffs[i] = _maSumOffs[i];
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

        void SynetQuantizedConvolutionNhwcSpecV1::SetWeight(const int8_t* weight)
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

        void SynetQuantizedConvolutionNhwcSpecV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* tmp = Allocate<uint8_t>(buf8, a.bufS);
            int32_t* sum = a.bufD ? Allocate<int32_t>(buf8, a.bufD) : NULL;
            int32_t* buf = Allocate<int32_t>(buf8, 2048);
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                if (a.batch == 1)
                {
                    if(a.macroD < a.dstC)
                        ForwardSingleAny(src, tmp, sum, buf, dst);
                    else
                        ForwardSingleL3(src, tmp, sum, buf, dst);
                }
                else
                    ForwardBatch(src, tmp, sum, buf, dst);
                src += _sizeS * a.batch * _elemS;
                dst += _sizeD * a.batch * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcSpecV1::ForwardSingleAny(const uint8_t* src, uint8_t* tmp, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int32_t* sBias = _bias.data;
            const float* sNorm = _norm.data;
            const float* params = _params.data;
            float dNorm = 1.0f / _dstScale;
            size_t dS = a.microC, dB = a.macroD, dD = p.dstC * _elemD;
            size_t tmpOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * dS;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const int8_t* weight = _weight.data + mad * a.K;
                const int* srcOffs = _srcOffs.data;
                for (size_t nk = 0; nk < _nK.size; ++nk)
                {
                    int update = nk == 0 ? 0 : 1;
                    size_t nK = _nK[nk];
                    for (size_t dyBeg = 0, dyN = 0; dyBeg < p.dstH; dyN++)
                    {
                        size_t dyEnd = Simd::Min(dyBeg + a.macroH, p.dstH);
                        size_t dstS = _maSumOffs[dyN + 1] - _maSumOffs[dyN];
                        size_t miIdx = _maSumOffs[dyN] / a.microS;
                        if (mad == 0 && nk == 0)
                            _preprocess(src, _srcZero[0], p, a, dyBeg, dyEnd, dyEnd == p.dstH ? 1 : 0, tmp);
                        if (nk == _nK.size - 1)
                            _lastConv(tmp + tmpOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, update, weight,
                                sum + _maSumOffs[dyN] * dB, buf, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero, 
                                _dstMask.data + _maSumOffs[dyN], _miDstOffs.data + miIdx, dst + _miDstOffs[miIdx] * dD);
                        else
                            _bodyConv(tmp + tmpOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, update, weight, sum + _maSumOffs[dyN] * dB);
                        dyBeg = dyEnd;
                    }
                    srcOffs += nK;
                    weight += nK * a.microC * a.F;
                }
                sBias += macroD;
                sNorm += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcSpecV1::ForwardSingleL3(const uint8_t* src, uint8_t* tmp, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int32_t* sBias = _bias.data;
            const float* sNorm = _norm.data;
            const float* params = _params.data;
            float dNorm = 1.0f / _dstScale;
            size_t dS = a.microC, dB = a.macroD, dD = p.dstC * _elemD;
            size_t tmpOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * dS;
            size_t macroD = Simd::Min(p.dstC, a.macroD);
            for (size_t dyBeg = 0, dyN = 0; dyBeg < p.dstH; dyN++)
            {
                size_t dyEnd = Simd::Min(dyBeg + a.macroH, p.dstH);
                size_t dstS = _maSumOffs[dyN + 1] - _maSumOffs[dyN];
                size_t miIdx = _maSumOffs[dyN] / a.microS;
                const int8_t* weight = _weight.data;
                const int* srcOffs = _srcOffs.data;
                _preprocess(src, _srcZero[0], p, a, dyBeg, dyEnd, dyEnd == p.dstH ? 1 : 0, tmp);
                for (size_t nk = 0; nk < _nK.size; ++nk)
                {
                    int update = nk == 0 ? 0 : 1;
                    size_t nK = _nK[nk];
                    if (nk == _nK.size - 1)
                        _lastConv(tmp + tmpOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, update, weight,
                            sum + _maSumOffs[dyN] * dB, buf, sBias, sNorm, _intZero, _intScale, params, dNorm, _dstZero,
                            _dstMask.data + _maSumOffs[dyN], _miDstOffs.data + miIdx, dst + _miDstOffs[miIdx] * dD);
                    else
                        _bodyConv(tmp + tmpOffs + _maBufOffs[dyN] * dS, p, a, srcOffs, macroD, dstS, nK, update, weight, sum + _maSumOffs[dyN] * dB);
                    srcOffs += nK;
                    weight += nK * a.microC * a.F;
                }
                dyBeg = dyEnd;
            }
        }

        void SynetQuantizedConvolutionNhwcSpecV1::ForwardBatch(const uint8_t* src, uint8_t* tmp, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const int32_t* sBias = _bias.data;
            const float* sNorm = _norm.data;
            const float* params = _params.data;
            float dNorm = 1.0f / _dstScale;
            const int* mask = _dstMask.data;
            size_t dstH = p.dstH * a.batch, dstS = _maSumOffs[1] - _maSumOffs[0];
            size_t tmpOffs = ((a.padV - p.padY) * a.srcW + (a.padH - p.padX)) * a.microC;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                const int8_t* weight = _weight.data + mad * a.K;
                const int* srcOffs = _srcOffs.data;
                for (size_t nk = 0; nk < _nK.size; ++nk)
                {
                    int update = nk == 0 ? 0 : 1;
                    size_t nK = _nK[nk];
                    if (mad == 0 && nk == 0)
                    {
                        size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                        size_t dT = a.srcH * a.srcW * a.microC;
                        for (size_t b = 0; b < a.batch; ++b)
                            _preprocess(src + b * dS, _srcZero[0], p, a, 0, p.dstH, b == a.batch - 1 ? 1 : 0, tmp + b * dT);
                    }
                    if (nk == _nK.size - 1)
                        _lastConv(tmp + tmpOffs, p, a, srcOffs, macroD, dstS, nK, update, weight, sum, buf, sBias, sNorm,
                            _intZero, _intScale, params, dNorm, _dstZero, mask, _miDstOffs.data, dst);
                    else
                        _bodyConv(tmp + tmpOffs, p, a, srcOffs, macroD, dstS, nK, update, weight, sum);
                    srcOffs += nK;
                    weight += nK * a.microC * a.F;
                }
                sBias += macroD;
                sNorm += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetQuantizedConvolutionNhwcSpecV1::Preferable(const ConvParam& p)
        {
            const size_t K = p.srcC * p.kernelX * p.kernelY;
            const size_t M = p.batch * p.dstH * p.dstW;
            return p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && 
                !p.IsKernel(1) && p.dstC >= 4 && K >= 32 && M >= 16 && p.srcC >= 9 && 1;
        }
    }
#endif
}
