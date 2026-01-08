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
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
	{
		typedef Base::SynetConvolution16bNhwcGemmV1::AlgParam AlgParam;
		typedef Base::SynetConvolution16bNhwcGemmV1::ConvolutionPtr Convolution;

        //-------------------------------------------------------------------------------------------------

		static void Convert16bNhwcGemmD(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
		{
			const float* src = (float*)src8;
			size_t srcC32 = AlignLo(p.srcC, 32);
			__mmask16 srcMask0 = TailMask16(p.srcC - srcC32 - F * 0);
			__mmask16 srcMask1 = TailMask16(p.srcC - srcC32 - F * 1);
			__mmask32 dstMask = TailMask32(p.srcC - srcC32);
			__mmask32 gapMask = TailMask32(a.bufK - a.K);
			for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
			{
				for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
				{
					uint16_t* row = dst + dr * a.bufK;
					for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
					{
						size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
						if (sy < p.srcH)
						{
							for (size_t kx = 0; kx < p.kernelX; kx++)
							{
								size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
								if (sx < p.srcW)
								{
									const float* ps = src + (sy * p.srcW + sx) * p.srcC;
									size_t sc = 0;
									for (; sc < srcC32; sc += 32)
										ConvertA(ps + sc, row + sc);
									if (srcC32 < p.srcC)
										ConvertA(ps + sc, row + sc, srcMask0, srcMask1, dstMask);
									row += p.srcC;
								}
								else
								{
									SetZeros(row, srcC32, dstMask);
									row += p.srcC;
								}
							}
						}
						else
						{
							SetZeros(row, p.kernelX * p.srcC);
							row += p.kernelX * p.srcC;
						}
					}
					SetZero(row, gapMask);
				}
			}
		}

        static void Convert16bNhwcGemmD_1d32ck(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            assert(p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32);
            const float* src = (float*)src8;
            size_t K = a.bufK, C = p.srcC, C32 = AlignLo(C, 32), kY = p.kernelY, kX = p.kernelX, kcX = kX * C, sX = p.strideX, cW = p.srcW * C, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask32 gM = TailMask32(K - a.K), kcM = TailMask32(kcX);
            __mmask16 skcM0 = TailMask16(kcX - F * 0), skcM1 = TailMask16(kcX - F * 1);
            __mmask8 cM = TailMask8(C);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint16_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    ConvertAq(src + sy * cW + sxc, pd, cM);
                                else
                                    _mm_mask_storeu_epi16(pd, cM, _mm_setzero_si128());
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                }
                if (dy >= dyB && dy < dyE)
                {
                    const float* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint16_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kcX)
                            ConvertA(ps + ky * cW, pd, skcM0, skcM1, kcM);
                        _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint16_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                ConvertA(src + sy * cW + sxcB, pd, skcM0, skcM1, kcM);
                            else
                                _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                        _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint16_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    ConvertAq(src + sy * cW + sxc, pd, cM);
                                else
                                    _mm_mask_storeu_epi16(pd, cM, _mm_setzero_si128());
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                }
            }
        }

        static void Convert16bNhwcGemm1x1D(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32), n = (yEnd - yBeg) * p.dstW;
            __mmask16 srcMask0 = TailMask16(p.srcC - srcC32 - F * 0);
            __mmask16 srcMask1 = TailMask16(p.srcC - srcC32 - F * 1);
            src += yBeg * p.srcW * p.srcC;
            for (size_t i = 0; i < n; ++i)
            {
                size_t sc = 0;
                for (; sc < srcC32; sc += 32)
                    ConvertA(src + sc, dst + sc);
                if (srcC32 < p.srcC)
                    ConvertA(src + sc, dst + sc, srcMask0, srcMask1);
                src += p.srcC;
                dst += a.bufK;
            }
        }

        static void Reorder16bNhwcGemmD(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask32 gapMask = TailMask32(a.bufK - a.K), tailMask = TailMask32(p.srcC - srcC32);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    Copy(src + (sy * p.srcW + sx) * p.srcC, srcC32, tailMask, row);
                                    row += p.srcC;
                                }
                                else
                                {
                                    SetZeros(row, srcC32, tailMask);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            SetZeros(row, p.kernelX * p.srcC);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    SetZero(row, gapMask);
                }
            }
        }

        static void Reorder16bNhwcGemmD_1d32ck(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            assert(p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32);
            const uint16_t* src = (uint16_t*)src8;
            size_t K = a.bufK, C = p.srcC, C32 = AlignLo(C, 32), kY = p.kernelY, kX = p.kernelX, kcX = kX * C, sX = p.strideX, cW = p.srcW * C, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask32 gM = TailMask32(K - a.K), kcM = TailMask32(kcX);
            __mmask8 cM = TailMask8(C);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint16_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi16(pd, cM, _mm_maskz_loadu_epi16(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi16(pd, cM, _mm_setzero_si128());
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                }
                if (dy >= dyB && dy < dyE)
                {
                    const uint16_t* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint16_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kcX)
                            _mm512_mask_storeu_epi16(pd, kcM, _mm512_maskz_loadu_epi16(kcM, ps + ky * cW));
                        _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint16_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                _mm512_mask_storeu_epi16(pd, kcM, _mm512_maskz_loadu_epi16(kcM, src + sy * cW + sxcB));
                            else
                                _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                        _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint16_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi16(pd, cM, _mm_maskz_loadu_epi16(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi16(pd, cM, _mm_setzero_si128());
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi16(pd, kcM, _mm512_setzero_si512());
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi16(pd, gM, _mm512_setzero_si512());
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcGemmV1::SynetConvolution16bNhwcGemmV1(const ConvParam & p)
            : Base::SynetConvolution16bNhwcGemmV1(p)
        {
            SetAlgParam();
            AlgParam& a = _alg;            
            if (_src16b)
            {
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                {
                    if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
                        _convert = Reorder16bNhwcGemmD_1d32ck;
                    else
                        _convert = Reorder16bNhwcGemmD;
                }
            }
            else
            {
                if (_is1x1)
                {
                    _convert = Convert16bNhwcGemm1x1D;
                }
                else
                {
                    if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
                        _convert = Convert16bNhwcGemmD_1d32ck;
                    else
                        _convert = Convert16bNhwcGemmD;
                }
            }
#if !defined(SIMD_MSVS_COMPILER_OUT_OF_HEAP_SPACE)
            if (CanDir1x4(p))
                SetMacro16x64d();
            else
#endif
            if (CanDir2x2(p))
                SetMacro32x32d();
            else if (CanInv4x1(p))
                SetMacro64x16i();
            else
                SetMacro32x32i_old();
        }
    }
#endif
}
