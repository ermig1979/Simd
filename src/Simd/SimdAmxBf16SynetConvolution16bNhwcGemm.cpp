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

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
	{
		typedef Base::SynetConvolution16bNhwcGemm::AlgParam AlgParam;
		typedef Base::SynetConvolution16bNhwcGemm::ConvolutionPtr Convolution;

#define SIMD_CONV_REORDER_TYPE 1

		//-----------------------------------------------------------------------------------------

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

		static void Convert16bNhwcGemmR(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
		{
			const float* src = (float*)src8;
			size_t srcC32 = AlignLo(p.srcC, 32);
			assert(p.srcC == srcC32);
			for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
			{
				for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
				{
                    size_t drB = dr & (~15), drO = dr & 15;
					uint16_t* row = dst + drB * a.bufK + drO * 32;
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
									for (size_t sc = 0; sc < srcC32; sc += 32, row += 512)
										ConvertA(ps + sc, row);
								}
								else
								{
                                    for (size_t sc = 0; sc < srcC32; sc += 32, row += 512)
                                        SetZero(row);
								}
							}
						}
						else
						{
                            for (size_t sc = 0, scN = p.kernelX * srcC32; sc < scN; sc += 32, row += 512)
                                SetZero(row);
						}
					}
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

        static void Convert16bNhwcGemm1x1R(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32), n = (yEnd - yBeg) * p.dstW;
            __mmask16 srcMask0 = TailMask16(p.srcC - srcC32 - F * 0);
            __mmask16 srcMask1 = TailMask16(p.srcC - srcC32 - F * 1);
            src += yBeg * p.srcW * p.srcC;
            for (size_t i = 0; i < n; i += 16)
            {
                size_t m = Min(i + 16, n) - i;
                size_t sc = 0;
                for (; sc < srcC32; sc += 32)
                {
                    size_t j = 0;
                    for(; j < m; ++j)
                        ConvertA(src + sc + j * p.srcC, dst + j * 32 + sc * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 32 + sc * 16);
                }
                if (srcC32 < p.srcC)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        ConvertA(src + sc + j * p.srcC, dst + j * 32 + sc * 16, srcMask0, srcMask1);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 32 + sc * 16);
                }
                src += p.srcC * 16;
                dst += a.bufK * 16;
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

        static void Reorder16bNhwcGemmR(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            assert(p.srcC == srcC32);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    size_t drB = dr & (~15), drO = dr & 15;
                    uint16_t* row = dst + drB * a.bufK + drO * 32;
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
                                    const uint16_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    for (size_t sc = 0; sc < srcC32; sc += 32, row += 512)
                                        Avx512bw::Copy(ps + sc, row);
                                }
                                else
                                {
                                    for (size_t sc = 0; sc < srcC32; sc += 32, row += 512)
                                        SetZero(row);
                                }
                            }
                        }
                        else
                        {
                            for (size_t sc = 0, scN = p.kernelX * srcC32; sc < scN; sc += 32, row += 512)
                                SetZero(row);
                        }
                    }
                }
            }
        }

        static void Reorder16bNhwcGemm1x1R(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32), n = (yEnd - yBeg) * p.dstW;
            __mmask32 srcMask = TailMask32(p.srcC - srcC32);
            src += yBeg * p.srcW * p.srcC;
            for (size_t i = 0; i < n; i += 16)
            {
                size_t m = Min(i + 16, n) - i;
                size_t sc = 0;
                for (; sc < srcC32; sc += 32)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 32 + sc * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 32 + sc * 16);
                }
                if (srcC32 < p.srcC)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 32 + sc * 16, srcMask);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 32 + sc * 16);
                }
                src += p.srcC * 16;
                dst += a.bufK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNhwcGemm_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if (cfg)
                SetTileConf2x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
                _tile_stream_loadd(3, buf + 16 * dB + F, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (term == Term16bLast16b)
            {
                __mmask32 tailD = TailMask32(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply16b2x8<type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply16b2<type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else if (term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply2x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply2<term, type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
                TileMoveToMemory(buf + 16 * dB + F, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNhwcGemm_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC32; sc += 32, src1 += stepS)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            if (term == Term16bLast16b || term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply1x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term, type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNhwcGemm_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if (cfg)
                SetTileConf1x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32; src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            if (term == Term16bLast16b || term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply2x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply2<term, type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNhwcGemm_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
            }

            for (size_t sc = 0; sc < srcC; sc += 32, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            _tile_stored(0, buf + 0, strideB);
            if (term == Term16bLast16b || term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply1x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term, type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
            }
        }

        typedef void (*Convolution16bNhwcGemmPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.macroK < a.bufK ? a.dB : 0, dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            if (nn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 1 && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), nn = n1 - m;
                Convolution16bNhwcGemmPtr body_2 = Convolution16bNhwcGemm_32x32<term, type, 0>;
                Convolution16bNhwcGemmPtr tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, 0> : Convolution16bNhwcGemm_16x32<term, type, 0>;
                Convolution16bNhwcGemmPtr body_1 = Convolution16bNhwcGemm_32x16<term, type, 0>;
                Convolution16bNhwcGemmPtr tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, 0> : Convolution16bNhwcGemm_16x16<term, type, 0>;
                SetTileConfFull();
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    float* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, srcC, n, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                        if (m)
                            tail_2(s + nn * dS, p, a, srcC, m, dC, zero, weight, _bias, _params, b + i * dB, d + nn * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, srcC, n, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                        if (m)
                            tail_1(s + nn * dS, p, a, srcC, m, dC, zero, weight, _bias, _params, b + i * dB, d + nn * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                Convolution16bNhwcGemmPtr tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, 0> : Convolution16bNhwcGemm_16x32<term, type, 0>;
                Convolution16bNhwcGemmPtr tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, 0> : Convolution16bNhwcGemm_16x16<term, type, 0>;
                if(m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    float* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                        tail_2(s, p, a, srcC, m, dC, zero, weight, _bias, _params, b, d);
                    else
                        tail_1(s, p, a, srcC, m, dC, zero, weight, _bias, _params, b, d);
                    weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply16b1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0 * F), bias[0]), params, 0);
            if(flush == 2)
                _mm256_stream_si256((__m256i*)ptr, (__m256i)_mm512_cvtneps_pbh(f0));
            else
            {
                _mm256_mask_storeu_epi16((uint16_t*)ptr, tail, (__m256i)_mm512_cvtneps_pbh(f0));
                if (flush)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply16b2(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0 * F), bias[0]), params, 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 1 * F), bias[1]), params, 1);
            if (flush == 2)
                _mm512_stream_si512((__m512i*)ptr, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
            else
            {
                _mm512_mask_storeu_epi16((uint16_t*)ptr, tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                if (flush)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply16b2x2(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            Apply16b2<type, flush>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            Apply16b2<type, flush>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply16b2x4(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            Apply16b2x2<type, flush>(ptr + 0 * dP, dP, buf + 0 * dB, dB, bias, params, tail);
            Apply16b2x2<type, flush>(ptr + 2 * dP, dP, buf + 2 * dB, dB, bias, params, tail);
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply16b2x8(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            Apply16b2x4<type, flush>(ptr + 0 * dP, dP, buf + 0 * dB, dB, bias, params, tail);
            Apply16b2x4<type, flush>(ptr + 4 * dP, dP, buf + 4 * dB, dB, bias, params, tail);
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply32f1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0 * F), bias[0]), params, 0);
            if (flush == 2)
                _mm512_stream_ps((float*)ptr, f0);
            else
            {
                _mm512_mask_storeu_ps((float*)ptr, tail, f0);
                if (flush)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply32f2(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0 * F), bias[0]), params, 0);
            if (flush == 2)
                _mm512_stream_ps((float*)ptr + 0, f0);
            else
            {
                _mm512_storeu_ps((float*)ptr + 0, f0);
                if (flush)
                    _mm_prefetch((const char*)(ptr + 0), _MM_HINT_NTA);
            }
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 1 * F), bias[1]), params, 1);
            if (flush == 2)
                _mm512_stream_ps((float*)ptr + F, f1);
            else
            {
                _mm512_mask_storeu_ps((float*)ptr + F, tail, f1);
                if (flush)
                    _mm_prefetch((const char*)(ptr + A), _MM_HINT_NTA);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_TinyC_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = a.sumBuf ? 32 : (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            int srcC32 = (int)srcC - 32, sc = 0;
            if(stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                if (stream)
                    _tile_stream_loadd(5, src1, strideS);
                else
                    _tile_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            if (stream)
                _tile_stream_loadd(5, src1, strideS);
            else
                _tile_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf + F, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, buf + 16 * dB + F, strideB);

            if (term == Term16bLast16b)
            {
                __mmask32 tailD = TailMask32(dstC);
                size_t ds = 0, dstS8 = dstS&(~7);
                for (; ds < dstS8; ds += 8)
                    Apply16b2x8<type, flush>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply16b2<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            if (term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply32f2<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_TinyC_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = a.sumBuf ? 32 : (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            _tile_zero(0);
            _tile_zero(2);

            int srcC32 = (int)srcC - 32, sc = 0;
            if (stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            for (; sc < srcC32; sc += 32, src1 += stepS)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                if (stream)
                    _tile_stream_loadd(5, src1, strideS);
                else
                    _tile_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                src0 += stepS;
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            if (stream)
                _tile_stream_loadd(5, src1, strideS);
            else
                _tile_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf + 16 * dB + 0, strideB);

            if (term == Term16bLast16b)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply16b1<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            if (term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply32f1<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_TinyC_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = a.sumBuf ? 32 : (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            _tile_zero(0);
            _tile_zero(1);

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32; src0 += stepS)
            {
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf + F, strideB);

            if (term == Term16bLast16b)
            {
                __mmask32 tailD = TailMask32(dstC);
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    Apply16b2x8<type, flush>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply16b2<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            if (term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply32f2<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_TinyC_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = a.sumBuf ? 32 : (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;

            _tile_zero(0);

            for (size_t sc = 0; sc < srcC; sc += 32, src0 += stepS)
            {
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            _tile_stored(0, buf + 0, strideB);
            if (term == Term16bLast16b)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply16b1<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            if (term == Term16bLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t ds = 0;
                for (; ds < dstS; ++ds)
                    Apply32f1<type, flush>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
        }

        typedef void (*Convolution16bNhwcGemmTinyCPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_TinyC_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dD = p.dstC * a.elem, dS = a.bufK;
            bool bigAlignedDst = Aligned(dst, A) && Aligned(dD, A) && dD * p.dstW * p.dstH > 4 * Base::AlgCacheL1();

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            Convolution16bNhwcGemmTinyCPtr body_2, tail_2, body_1, tail_1;

            if (nn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 1 && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), nn = n1 - m;
                tail_2 = m > 16 ? Convolution16bNhwcGemm_TinyC_32x32<term, type, 1, 1> : Convolution16bNhwcGemm_TinyC_16x32<term, type, 1, 1>;
                body_1 = Convolution16bNhwcGemm_TinyC_32x16<term, type, 1, 1>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_TinyC_32x16<term, type, 1, 1> : Convolution16bNhwcGemm_TinyC_16x16<term, type, 1, 1>;
                SetTileConfFull();
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    if(bigAlignedDst)
                        body_2 = dc == 0 ? Convolution16bNhwcGemm_TinyC_32x32<term, type, 0, 2> : Convolution16bNhwcGemm_TinyC_32x32<term, type, 1, 2>;
                    else
                        body_2 = dc == 0 ? Convolution16bNhwcGemm_TinyC_32x32<term, type, 0, 1> : Convolution16bNhwcGemm_TinyC_32x32<term, type, 1, 1>;
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, srcC, n, dC, weight, _bias, _params, buf, d + i * dD);
                        if (m)
                            tail_2(s + nn * dS, p, a, srcC, m, dC, weight, _bias, _params, buf, d + nn * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, srcC, n, dC, weight, _bias, _params, buf, d + i * dD);
                        if (m)
                            tail_1(s + nn * dS, p, a, srcC, m, dC, weight, _bias, _params, buf, d + nn * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                tail_2 = m > 16 ? Convolution16bNhwcGemm_TinyC_32x32<term, type, 1, 1> : Convolution16bNhwcGemm_TinyC_16x32<term, type, 1, 1>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_TinyC_32x16<term, type, 1, 1> : Convolution16bNhwcGemm_TinyC_16x16<term, type, 1, 1>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                        tail_2(s, p, a, srcC, m, dC, weight, _bias, _params, buf, d);
                    else
                        tail_1(s, p, a, srcC, m, dC, weight, _bias, _params, buf, d);
                    weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution* convolutions)
        {
            if (a.macroK < a.K)
            {
                convolutions[0] = Convolution16bNhwcGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
                if (p.dstT == SimdTensorData16b)
                    convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast16b, type>;
                else
                    convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast32f, type>;
            }
            else
            {
                convolutions[0] = NULL;
                if (p.dstT == SimdTensorData16b)
                    convolutions[1] = Convolution16bNhwcGemm_TinyC_2<Term16bLast16b, type>;
                else
                    convolutions[1] = Convolution16bNhwcGemm_TinyC_2<Term16bLast32f, type>;
            }
        }

        SynetConvolution16bNhwcGemm::SynetConvolution16bNhwcGemm(const ConvParam & p)
            : Avx512bw::SynetConvolution16bNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, F * 2, 32, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            AlgParam& a = _alg;            
            if (_src16b)
            {
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                {
                    if (_is1x1 && a.batch == 1)
                    {
                        _convert = Reorder16bNhwcGemm1x1R;
                        a.reorderType = 1;
                    }
                    else
                    {
                        if (Aligned(p.srcC, 32) && a.batch == 1 && Aligned(p.dstW, a.F))
                        {
                            _convert = Reorder16bNhwcGemmR;
                            a.reorderType = 1;
                        }
                        else
                        {
                            if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
                                _convert = Reorder16bNhwcGemmD_1d32ck;
                            else
                                _convert = Reorder16bNhwcGemmD;
                        }
                    }
                }
            }
            else
            {
                if (_is1x1)
                {
                    if (a.batch == 1/* && a.macroK < a.bufK*/)
                    {
                        _convert = Convert16bNhwcGemm1x1R;
                        a.reorderType = 1;
                    }
                    else
                    {
                        _convert = Convert16bNhwcGemm1x1D;
                        a.reorderType = 0;
                    }
                }
                else
                {
                    if (p.srcC == AlignLo(p.srcC, 32) && a.batch == 1 && Aligned(p.dstW, a.F))
                    {
                        _convert = Convert16bNhwcGemmR;
                        a.reorderType = 1;
                    }
                    else
                    {
                        if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
                            _convert = Convert16bNhwcGemmD_1d32ck;
                        else
                            _convert = Convert16bNhwcGemmD;
                        a.reorderType = 0;
                    }
                }
            }
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, _alg, _convolutions); break;
            default: assert(0);
            }
        }
    }
#endif
}
