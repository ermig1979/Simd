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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Base
    {
        SynetQuantizedConvolutionNchwGemm::SynetQuantizedConvolutionNchwGemm(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _conv = 0;
            _gemm[0] = 0;
            _gemm[1] = 0;
        }

        String SynetQuantizedConvolutionNchwGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NchwGemm";
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNchwGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            return a.bufN * a.bufK * sizeof(uint8_t) + a.bufD * a.bufN * sizeof(int32_t);
        }

        void SynetQuantizedConvolutionNchwGemm::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            _weight.Assign(weight, _weight.size);
        }

        bool SynetQuantizedConvolutionNchwGemm::Preferable(const ConvParam& p)
        {
            return p.trans == 0 && p.group == 1 && Is1x1(p);
        }

        void SynetQuantizedConvolutionNchwGemm::SetAlgParam()
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            const int L1 = (int)Base::AlgCacheL1(), L2 = int(Base::AlgCacheL2() * 0.5), L3 = (int)Base::AlgCacheL3();

            a.N = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = 16;
            a.microD = 32;
            a.microN = 32;
            a.microK = 64;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD, a.microK), a.microK, a.bufK);
            a.macroH = Simd::RestrictRange(L3 / a.macroK / p.dstW, size_t(1), p.dstH);
            a.macroD = Simd::RestrictRange(AlignLoAny(L2 / a.macroK, a.microD), a.microD, a.bufD);
            a.bufN = p.dstH * AlignHi(p.dstW, a.F);
            a.elem = _elemD;
            a.reorderType = 0;
            a.sumBuf = (a.macroK < a.K) || a.microK > 2 ? 1 : 0;
            if (a.sumBuf == 0 && a.macroD > p.dstC)
                a.macroD = p.dstC;
        }

		void SynetQuantizedConvolutionNchwGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
		{
			const ConvParam& p = _param;
			const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* bufB = Allocate<uint8_t>(buf8, a.bufN * a.bufK);
            int32_t* bufS = Allocate<int32_t>(buf8, a.macroD * a.bufN);
            for (size_t b = 0; b < p.batch; b += 1)
            {
                //Forward(src, bufB, bufS, dst);
                src += _sizeS;
                dst += _sizeD;
            }
		}
    }
#endif
}
