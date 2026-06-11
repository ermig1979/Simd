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
        SynetQuantizedConvolutionNhwcGemmV1::SynetQuantizedConvolutionNhwcGemmV1(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _convAny = 0;
            _conv1x1 = 0;
            _gemm[0] = 0;
            _gemm[1] = 0;
        }

        String SynetQuantizedConvolutionNhwcGemmV1::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemmV1";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcGemmV1::SetAlgParam()
        {
            const int L1 = int(Base::AlgCacheL1()), L2 = int(Base::AlgCacheL2() * 0.5), L3 = int(Base::AlgCacheL3());
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.M = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = 16;
            a.microD = 32;
            a.microM = 32;
            a.microK = 64;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD, a.microK), a.microK, a.bufK);
            a.batch = 1;
            size_t bufSize = a.M * a.bufK;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (batch * bufSize <= L2 && (a.bufK > 512 || batch * a.M <= 32 * a.microM))
                        a.batch = batch;
            }
            a.batch = DivHi(p.batch, DivHi(p.batch, a.batch));
            a.macroM = Simd::RestrictRange(AlignLoAny(L2 / a.macroK, a.microM), a.microM, a.batch * a.M);
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroK, a.microD), a.microD, a.bufD);
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
            a.tmpBuf = !_is1x1 || a.K != a.bufK;
            a.sumBuf = (a.bufD != p.dstC || a.bufM != a.batch * p.dstH * p.dstW) && a.macroK < a.bufK;
            a.reorderType = a.tmpBuf != 0 && (_is1x1 || (Aligned(p.srcC, a.microK) && a.isAlMaH));
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;
            a.dB = (a.sumBuf ? a.macroD : p.dstC);
        }

        size_t SynetQuantizedConvolutionNhwcGemmV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            return size;
        }

        void SynetQuantizedConvolutionNhwcGemmV1::SetWeight(const int8_t* weight)
        {
        }

        void SynetQuantizedConvolutionNhwcGemmV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
        }

        void SynetQuantizedConvolutionNhwcGemmV1::Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
        {
        }

        bool SynetQuantizedConvolutionNhwcGemmV1::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }
    }
#endif
}
