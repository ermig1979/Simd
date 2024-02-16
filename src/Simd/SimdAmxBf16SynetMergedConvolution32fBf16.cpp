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
#include "Simd/SimdSynetMergedConvolution32fBf16.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)
	namespace AmxBf16
	{
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;

        //-----------------------------------------------------------------------------------------

        static void ConvertFp32ToBf16(const float* src, const ConvParam32f& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            size_t srcC = AlignHi(p.srcC, a.miK);
            size_t bufH = a.bufH[0], mask = bufH - 1;
            if (srcC == p.srcC)
            {
                size_t size = p.srcW * p.srcC;
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, bufH));
                if (yInt > yBeg)
                    Float32ToBFloat16(src + yBeg * size, (yInt - yBeg) * size, dst + (yBeg & mask) * size);
                if (yEnd > yInt)
                    Float32ToBFloat16(src + yInt * size, (yEnd - yInt) * size, dst + (yInt & mask) * size);
            }
            else
            {
                size_t srcC32 = AlignLo(p.srcC, 32);
                __mmask16 srcMask[2];
                __mmask32 dstMask[1], gapMask = TailMask32(srcC - p.srcC);
                if (srcC32 < p.srcC)
                {
                    srcMask[0] = TailMask16(p.srcC - srcC32 - F * 0);
                    srcMask[1] = TailMask16(p.srcC - srcC32 - F * 1);
                    dstMask[0] = TailMask32(p.srcC - srcC32);
                }
                for (size_t y = yBeg; y < yEnd; ++y)
                {
                    const float* ps = src + y * p.srcW * p.srcC;
                    uint16_t* pd = dst + (y & mask) * p.srcW * srcC;
                    for (size_t x = 0; x < p.srcW; ++x)
                    {
                        size_t c = 0;
                        for (; c < srcC32; c += 32)
                            Float32ToBFloat16<false, false>(ps + c, pd + c, srcMask, dstMask);
                        if (srcC32 < p.srcC)
                            Float32ToBFloat16<false, true>(ps + c, pd + c, srcMask, dstMask);
                        if (p.srcC < srcC)
                            Store<false, true>(pd + p.srcC, K_ZERO, gapMask);
                        ps += p.srcC;
                        pd += srcC;
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cdc::SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Cdc(p)
        {
            if (p.conv[2].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
#if defined(SIMD_AMX_EMULATE)
                _convert = Avx512bw::ConvertFp32ToBf16;
                if(_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                Avx512bw::SetDepthwise(_param.conv[1], _depthwise);
#else
                _convert = ConvertFp32ToBf16;
                if (_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                SetDepthwise(_param.conv[1], _depthwise);
#endif
                SetOutput(_param.conv[2], _output);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cd::SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Cd(p)
        {
            if (p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
#if defined(SIMD_AMX_EMULATE)
                _convert = Avx512bw::ConvertFp32ToBf16;
                if (_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                Avx512bw::SetDepthwise(_param.conv[1], _depthwise);
#else
                _convert = ConvertFp32ToBf16;
                if (_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                SetDepthwise(_param.conv[1], _depthwise);
#endif
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution32fBf16Dc::SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Dc(p)
        {
            if (p.conv[0].dstC > HF && p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
#if defined(SIMD_AMX_EMULATE)
                Avx512bw::SetDepthwise(_param.conv[0], _depthwise);
#else
                SetDepthwise(_param.conv[0], _depthwise);
#endif
                SetOutput(_param.conv[1], _output);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam32f param(batch, convs, count, add, compatibility);
            if (!param.Valid())
                return NULL;
            if (Base::Bf16Soft(compatibility) || Base::Bf16Hard(compatibility))
            {
                if (Base::SynetMergedConvolution32fBf16Cdc::Preferable(param))
                    return new SynetMergedConvolution32fBf16Cdc(param);
                else if (Base::SynetMergedConvolution32fBf16Cd::Preferable(param))
                    return new SynetMergedConvolution32fBf16Cd(param);
                else if (Base::SynetMergedConvolution32fBf16Dc::Preferable(param))
                    return new SynetMergedConvolution32fBf16Dc(param);
                else
                    return new Base::SynetMergedConvolution32fBf16(param);
            }
            return Avx512bw::SynetMergedConvolution32fInit(batch, convs, count, add, compatibility);
        }
	}
#endif
}
