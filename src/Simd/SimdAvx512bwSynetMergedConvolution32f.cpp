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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx512bw
    {
        SIMD_INLINE void ReorderPadF(const float* src, float* dst, size_t n)
        {
            if (n == F)
                Store<false>(dst, Load<false>(src));
            else
                Store<false>(dst, Load<false, true>(src, TailMask16(n)));
        }

        SIMD_INLINE void ReorderPadDF(const float* src, float* dst, size_t n)
        {
            if (n == DF)
            {
                Store<false>(dst + 0, Load<false>(src + 0));
                Store<false>(dst + F, Load<false>(src + F));
            }
            else if (n > F)
            {
                Store<false>(dst + 0, Load<false>(src + 0));
                Store<false>(dst + F, Load<false, true>(src + F, TailMask16(n - F)));
            }
            else
            {
                Store<false>(dst + 0, Load<false, true>(src, TailMask16(n)));
                Store<false>(dst + F, _mm512_setzero_ps());
            }
        }

        SynetMergedConvolution32fCdc::SynetMergedConvolution32fCdc(const MergConvParam& p)
            : Avx2::SynetMergedConvolution32fCdc(p)
        {
            SetSize(Base::AlgCacheL1(), size_t(Base::AlgCacheL2()*0.7), Base::AlgCacheL3(), F);
            SetInput(p.conv[0], _convolution + 0);
            SetDepthwise(p.conv[1], _convolution + 1);
            SetOutput(p.conv[2], _convolution + 2);
        }

        void SynetMergedConvolution32fCdc::ReorderFirstWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[0];
            size_t size = p.kernelY * p.kernelX * p.srcC, dstC = p.dstC;
            for (size_t c = 0; c < dstC; c += DF)
            {
                size_t n = Simd::Min(DF, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    ReorderPadDF(src + s * dstC + c, dst, n);
                    dst += DF;
                }
            }
        }

        void SynetMergedConvolution32fCdc::ReorderSecondWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[1];
            size_t dstC = p.dstC, size = p.kernelY * p.kernelX;
            for (size_t c = 0; c < dstC; c += F)
            {
                size_t n = Simd::Min(F, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    ReorderPadF(src + s * dstC + c, dst, n);
                    dst += F;
                }
            }
        }

        void SynetMergedConvolution32fCdc::ReorderThirdWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[2];
            size_t srcC = p.srcC, dstC = p.dstC;
            for (size_t m = 0; m < srcC; m += _maC)
            {
                size_t maC = Simd::Min(srcC, m + _maC) - m;
                for (size_t d = 0; d < dstC; d += DF)
                {
                    size_t n = Simd::Min(DF, dstC - d);
                    for (size_t s = 0; s < maC; s++)
                    {
                        ReorderPadDF(src + s * dstC + d, dst, n);
                        dst += DF;
                    }
                }
                src += dstC * maC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam& p)
            : Avx2::SynetMergedConvolution32fCd(p)
        {
            SetSize(Base::AlgCacheL1(), size_t(Base::AlgCacheL2() * 0.7), Base::AlgCacheL3(), F);
            SetInput(_param.conv[0], _convolution + 0);
            SetDepthwise(_param.conv[1], _convolution + 1);
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam& p)
            : Avx2::SynetMergedConvolution32fDc(p)
        {
            SetSize(Base::AlgCacheL1(), size_t(Base::AlgCacheL2() * 0.7), Base::AlgCacheL3(), F);
            SetDepthwise(p.conv[0], _convolution + 0);
            SetOutput(p.conv[1], _convolution + 1);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add, SimdSynetCompatibilityDefault);
            if (!param.Valid(SimdTensorData32f))
                return NULL;
            if (SynetMergedConvolution32fCdc::Preferable(param))
            {
                if (param.conv[1].dstC <= HF && param.conv[2].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fCdc(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fCdc(param);
            }
            else if (SynetMergedConvolution32fCd::Preferable(param))
            {
                if (param.conv[1].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fCd(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fCd(param);
            }
            else if (SynetMergedConvolution32fDc::Preferable(param))
            {
                if (param.conv[0].dstC <= HF || param.conv[1].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fDc(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fDc(param);
            }
            else
                return new Base::SynetMergedConvolution32f(param);
        }
	}
#endif
}
