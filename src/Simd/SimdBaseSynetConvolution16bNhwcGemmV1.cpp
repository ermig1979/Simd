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
#include "Simd/SimdCpu.h"

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
            desc << "-" << (_alg.inv ? "i" : "d");
            desc << _alg.microM / 16 << "x" << _alg.microD / 16;
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetConvolution16bNhwcGemmV1::SetAlgParam()
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            const int L1 = (int)Base::AlgCacheL1(), L2 = int(Base::AlgCacheL2() * 0.5), L3 = (int)Base::AlgCacheL3();
            const int microK = 32, F = 16;

            a.M = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.elem = _elemD;
            if (CanDir1x4(p))
            {
                a.inv = false;
                a.microD = 64;
                a.microM = 16;
                a.miniD = 64;
                a.miniM = 256;
            }
            else if(CanDir2x2(p))
            {
                a.inv = false;
                a.microD = 32;
                a.microM = 32;
                a.miniD = 32;
                a.miniM = 256;
            }
            else
            {
                a.inv = true;
                a.microD = 32;
                a.microM = 32;
                a.miniD = 32;
                a.miniM = 256;
            }

            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, microK);
            a.batch = 1;
            size_t bufSize = a.M * a.bufK * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.macroH = Simd::RestrictRange(L2 / a.bufK / p.dstW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.bufK / 2, a.miniD), a.miniD, a.bufD);
            if (CanDir1x4(p))
            {
                if(a.macroH < p.dstH)
                    a.macroH = 16;
                a.bufM = AlignHi(a.macroH * p.dstW, F);
            }
            else
            {
                a.bufM = a.batch * p.dstH * AlignHi(p.dstW, F);
            }

            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;
        }

        size_t SynetConvolution16bNhwcGemmV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            if(_convert)
                size += a.bufM * a.bufK * sizeof(uint16_t);
            size += a.miniD * a.microM * sizeof(float) * 2;
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
            size_t F = _alg.miniD, D = DivHi(p.dstC, F);
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
                if(a.inv)
                    ForwardInv(src, buf, bufS, dst);
                else
                    ForwardDir(src, buf, bufS, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemmV1::ForwardDir(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t dstH = p.dstH * a.batch;
            for (size_t yBeg = 0; yBeg < dstH;)
            {
                size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                size_t bufOffs = _convert == NULL ? yBeg * p.dstW * p.srcC : 0;
                size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                if (_convert)
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
                _convolution(buf + bufOffs, p, a, p.dstC, yEnd - yBeg, _weight.data, _bias.data, _params.data, sum, dst + dstOffs);
                yBeg = yEnd;
            }
        }

        void SynetConvolution16bNhwcGemmV1::ForwardInv(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
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
                    size_t bufOffs = (_convert == NULL || a.macroD < p.dstC) ? yBeg * (_convert ? AlignHi(p.dstW, 16) : p.dstW) * a.bufK : 0;
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
            return 1 && p.trans != 0 && p.group == 1 && (CanDir1x4(p) || CanDir2x2(p) || CanInv2x2_old(p));
        }

        bool SynetConvolution16bNhwcGemmV1::CanDir1x4(const ConvParam& p)
        {
#if !defined(SIMD_MSVS_COMPILER_OUT_OF_HEAP_SPACE)            
            const size_t K = p.srcC * p.kernelX * p.kernelY;
            return 1 && p.dstH * p.dstW >= 16 && K >= 32 && K <= 128 && Is1x1(p);
#else
            return false;
#endif        
        }
        
        bool SynetConvolution16bNhwcGemmV1::CanDir2x2(const ConvParam& p)
        {
            const size_t K = p.srcC * p.kernelX * p.kernelY, M = p.dstH * p.dstW, N = p.dstC;
            return 1 && K >= 256 && K <= 1024 && M >= 64;
        }

        bool SynetConvolution16bNhwcGemmV1::CanInv2x2_old(const ConvParam& p)
        {
            const size_t K = p.srcC * p.kernelX * p.kernelY;
            return 1 && ((K >= 128 && p.dstT == SimdTensorData16b) || (K >= 128 && p.dstT == SimdTensorData32f)) && K <= 512;
        }
    }
#endif
}
