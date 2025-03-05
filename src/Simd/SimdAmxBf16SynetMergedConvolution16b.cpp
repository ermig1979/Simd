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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)
	namespace AmxBf16
	{
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;

        //-----------------------------------------------------------------------------------------

        static void ConvertFp32ToBf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC = AlignHi(p.srcC, a.miK);
            if (srcC == p.srcC)
            {
                size_t size = p.srcW * p.srcC;
                Float32ToBFloat16(src + yBeg * size, (yEnd - yBeg) * size, dst);
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
                    uint16_t* pd = dst + (y - yBeg) * p.srcW * srcC;
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

        static void ReorderBf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC = AlignHi(p.srcC, a.miK);
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            for (size_t y = yBeg; y < yEnd; ++y)
            {
                const uint16_t* ps = src + y * p.srcW * p.srcC;
                uint16_t* pd = dst + (y - yBeg) * p.srcW * srcC;
                for (size_t x = 0; x < p.srcW; ++x)
                {
                    size_t c = 0;
                    for (; c < srcCDF; c += DF)
                        Avx512bw::Copy(ps + c, pd + c);
                    if (tailC)
                        Avx512bw::Copy(ps + c, pd + c, tailC);
                    ps += p.srcC;
                    pd += srcC;
                }
            }
        }

        static void ConvertBf16ToFp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC = p.srcC, dstC = AlignHi(srcC, F), srcW = p.srcW, srcCF = AlignLo(srcC, F);
            __mmask16 tail = TailMask16(srcC - srcCF);
            if (tail)
            {
                src += yBeg * srcW * srcC;
                dst += yBeg * srcW * dstC;
                for (size_t y = yBeg; y < yEnd; ++y)
                {
                    for (size_t x = 0; x < srcW; ++x)
                    {
                        size_t c = 0;
                        for (; c < srcCF; c += F)
                            _mm512_storeu_ps(dst + c, BFloat16ToFloat32(_mm256_loadu_epi16(src + c)));
                        if (tail)
                            _mm512_storeu_ps(dst + c, BFloat16ToFloat32(_mm256_maskz_loadu_epi16(tail, src + c)));
                        src += srcC;
                        dst += dstC;
                    }
                }
            }
            else
            {
                size_t rowSize = p.srcW * p.srcC;
                for (size_t y = yBeg; y < yEnd; ++y)
                    Avx512bw::BFloat16ToFloat32(src + y * rowSize, rowSize, dst + y * rowSize);
            }
        }

        void CopyFp32ToFp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst)
        {
            const float* src = (float*)src8;
            size_t srcC = p.srcC, dstC = AlignHi(srcC, F), srcW = p.srcW, srcCF = AlignLo(srcC, F);
            __mmask16 tail = TailMask16(srcC - srcCF);
            if (tail)
            {
                src += yBeg * srcW * srcC;
                dst += yBeg * srcW * dstC;
                for (size_t y = yBeg; y < yEnd; ++y)
                {
                    for (size_t x = 0; x < srcW; ++x)
                    {
                        size_t c = 0;
                        for (; c < srcCF; c += F)
                            _mm512_storeu_ps(dst + c, _mm512_loadu_ps(src + c));
                        if (tail)
                            _mm512_storeu_ps(dst + c, _mm512_maskz_loadu_ps(tail, src + c));
                        src += srcC;
                        dst += dstC;
                    }
                }
            }
            else
            {
                size_t rowSize = p.srcW * p.srcC;
                for (size_t y = yBeg; y < yEnd; ++y)
                    memcpy(dst + y * rowSize, src + y * rowSize, rowSize * sizeof(float));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution16bCdc::SynetMergedConvolution16bCdc(const MergConvParam& p)
            : Avx512bw::SynetMergedConvolution16bCdc(p)
        {
            if (p.conv[2].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
                if (!_src16b)
                    _toBf16 = ConvertFp32ToBf16;
                else if (!Aligned(p.conv[0].srcC, Avx512bw::DF))
                    _toBf16 = ReorderBf16;
                else
                    _toBf16 = NULL;
                if (_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                if (_src16b)
                    _toFp32 = ConvertBf16ToFp32;
                else
                    _toFp32 = CopyFp32ToFp32;
                SetDepthwise(_param.conv[1], _depthwise);
                SetOutput(_param.conv[2], _output);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution16bCd::SynetMergedConvolution16bCd(const MergConvParam& p)
            : Avx512bw::SynetMergedConvolution16bCd(p)
        {
            if (p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
                if (!_src16b)
                    _toBf16 = ConvertFp32ToBf16;
                else if (!Aligned(p.conv[0].srcC, Avx512bw::DF))
                    _toBf16 = ReorderBf16;
                else
                    _toBf16 = NULL;
                if (_param.conv[0].Is1x1())
                    SetInput(_param.conv[0], _input);
                else
                    Avx512bw::SetInput(_param.conv[0], _input);
                SetDepthwise(_param.conv[1], _depthwise);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution16bDc::SynetMergedConvolution16bDc(const MergConvParam& p)
            : Avx512bw::SynetMergedConvolution16bDc(p)
        {
            if (p.conv[0].dstC > HF && p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F, Avx512bw::DF);
                SetDepthwise(_param.conv[0], _depthwise);
                SetOutput(_param.conv[1], _output);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            if (Base::SynetMergedConvolution16bCdc::Preferable(param))
                return new SynetMergedConvolution16bCdc(param);
            else if (Base::SynetMergedConvolution16bCd::Preferable(param))
                return new SynetMergedConvolution16bCd(param);
            else if (Base::SynetMergedConvolution16bDc::Preferable(param))
                return new SynetMergedConvolution16bDc(param);
            else
                return new Base::SynetMergedConvolution16b(param);
        }
	}
#endif
}
