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
#include "Simd/SimdSynetApply16b.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
	{
		typedef Base::SynetConvolution16bNhwcGemmV2::AlgParam AlgParam;
		typedef Base::SynetConvolution16bNhwcGemmV2::GemmPtr GemmPtr;

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

        static void Convolution16bNhwcGemmV2_Convert1x1D(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t M, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask16 srcMask0 = TailMask16(p.srcC - srcC32 - F * 0);
            __mmask16 srcMask1 = TailMask16(p.srcC - srcC32 - F * 1);
            for (size_t i = 0; i < M; ++i)
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

        static void Convolution16bNhwcGemmV2_Convert1x1R(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t M, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask16 srcMask0 = TailMask16(p.srcC - srcC32 - F * 0);
            __mmask16 srcMask1 = TailMask16(p.srcC - srcC32 - F * 1);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
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

        static void Convolution16bNhwcGemmV2_Reorder1x1R(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t M, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask32 srcMask = TailMask32(p.srcC - srcC32);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
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

        //--------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void Convolution16bNhwcGemmV2_Gemm1xMx2(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, const uint16_t* weight0,
            const __m512* bias, const __m512* params, float* buf0, float* buf1, float* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem), dS = (int)a.bufK, dW = 16, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            int dB = term == Term16bInterim ? (int)a.dB : DF, strideB = dB * 4;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if (zero)
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
                if (M > 0) _tile_zero(2);
                if (M > 1) _tile_zero(3);
            }
            else
            {
                int dB = (int)a.dB, strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
                buf0 += 16 * dB;
                if (M > 0) _tile_stream_loadd(2, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(3, buf0 + F, strideB);
            }

            int sC32 = (int)srcC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            }
            for (; sc < sC32; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            _tile_stream_loadd(5, src1, strideS);
            if (M > 0) _tile_dpbf16ps(0, 4, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
            buf2 += 16 * dB;
            if (M > 0) _tile_dpbf16ps(2, 5, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(2, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(3, 5, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(3, buf2 + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void Convolution16bNhwcGemmV2_Gemm1xMx1(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, const uint16_t* weight0,
            const __m512* bias, const __m512* params, float* buf0, float* buf1, float* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem), dS = (int)a.bufK, dW = 16, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            int dB = term == Term16bInterim ? (int)a.dB : DF, strideB = dB * 4;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if (zero)
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
            }
            else
            {
                int dB = (int)a.dB, strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
            }

            int sC32 = (int)srcC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            for (; sc < sC32;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            if (M > 0) _tile_dpbf16ps(0, 4, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> void Convolution16bNhwcGemmV2_GemmNxMx2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf1, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK;
            float* buf2 = buf1 + 1024;

            if (term == Term16bInterim)
            {
                for (size_t cds = 0; cds < dstS; cds += 32)
                {
                    size_t bds = cds;
                    if (cds + 16 >= dstS)
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 16, cds);
                        Convolution16bNhwcGemmV2_Gemm1xMx1<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, sum0 + bds * dB, NULL, sum0 + bds * dB, NULL, tailD);
                    }
                    else
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 32, cds);
                        Convolution16bNhwcGemmV2_Gemm1xMx2<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, sum0 + bds * dB, NULL, sum0 + bds * dB, NULL, tailD);
                    }
                }
            }
            else
            {
                size_t cds = 0, pds = 0;
                Convolution16bNhwcGemmV2_Gemm1xMx2<term, type, flush, M, 0>(src0, p, a, srcC, zero, weight0, bias, params, sum0, buf1, buf2, dst, tailD), cds += 32;
                for (; cds < dstS; pds = cds, cds += 32)
                {
                    Swap(buf1, buf2);
                    size_t bds = cds;
                    if (cds + 16 >= dstS)
                    {
                        if(a.reorderType == 0)
                            cds = Simd::Min(dstS - 16, cds);
                        Convolution16bNhwcGemmV2_Gemm1xMx1<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, sum0 + bds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                    else
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 32, cds);
                        Convolution16bNhwcGemmV2_Gemm1xMx2<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, sum0 + bds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                }
                uint8_t* dst1 = dst + pds * dD;
                dstS -= pds;
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, M, 8>(dst1 + ds * dD, dD, buf2 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, M, 1>(dst1 + ds * dD, dD, buf2 + ds * DF, bias, params, tailD);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int cfg> void Convolution16bNhwcGemmV2_Gemm2x2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf0, size_t dB, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)dB * 4;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + dS * 16, * weight1 = weight0 + a.bufK * F;
            float* buf1 = buf0 + 16 * dB;

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
                int strideS = (int)a.dB * 4;
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(1, sum0 + F, strideS);
                sum0 += 16 * a.dB;
                _tile_stream_loadd(2, sum0 + 0, strideS);
                _tile_stream_loadd(3, sum0 + F, strideS);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                sc += 32;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            _tile_stored(3, buf1 + F, strideB);
            if (term != Term16bInterim)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 2, 8>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 2, 1>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int cfg> void Convolution16bNhwcGemmV2_Gemm2x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf0, size_t dB, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)dB * 4;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + dS * 16;
            float* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                int strideS = (int)a.dB * 4;
                _tile_stream_loadd(0, sum0 + 0, strideS);
                sum0 += 16 * a.dB;
                _tile_stream_loadd(2, sum0 + 0, strideS);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                src0 += stepS;
                sc += 32;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            if (term != Term16bInterim)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 1, 8>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 1, 1>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int cfg> void Convolution16bNhwcGemmV2_Gemm1x2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf0, size_t dB, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)dB * 4;
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
                int strideS = (int)a.dB * 4;
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(1, sum0 + F, strideS);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0, strideW);
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            if (term != Term16bInterim)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 2, 8>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 2, 1>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int cfg> void Convolution16bNhwcGemmV2_Gemm1x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf0, size_t dB, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)dB * 4;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                int strideS = (int)a.dB * 4;
                _tile_stream_loadd(0, sum0 + 0, strideS);
            }

            size_t sc = 0;
            for (; sc < srcC; sc += 32, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            _tile_stored(0, buf0 + 0, strideB);
            if (term != Term16bInterim)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 1, 8>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 1, 1>(dst + ds * dD, dD, buf0 + ds * DF, bias, params, tailD);
            }
        }

        typedef void (*Convolution16bNhwcGemmV2_GemmPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int zero, 
            const uint16_t* weight0, const __m512* bias, const __m512* params, float* sum0, float* buf0, size_t dB, uint8_t* dst, __mmask32 tailD);

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int apply> void Convolution16bNhwcGemmV2_Gemm(const uint16_t* src, const ConvParam& p, 
            const AlgParam& a, size_t N, size_t M, size_t K, int zero, const uint16_t* weight, const float* bias, const float* params, float* sum, float* buf, uint8_t* dst)
        {
            size_t n = 32, Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dW = a.bufK * DF, dS = a.bufK, dB = (term == Term16bInterim ? a.dB : DF), dD = p.dstC * a.elem;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (Mn)
            {
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_ps(bias + j + 0);
                    _bias[1] = _mm512_loadu_ps(bias + j + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16)); 
                    buf = (term == Term16bInterim ? sum + j : buf);
                    if (dN > F)
                        Convolution16bNhwcGemmV2_GemmNxMx2<term, type, flush, 2, apply>(src, p, a, K, M, zero, weight, _bias, _params, sum + j, buf, dst + j * a.elem, tailD);
                    else
                        Convolution16bNhwcGemmV2_GemmNxMx2<term, type, flush, 1, apply>(src, p, a, K, M, zero, weight, _bias, _params, sum + j, buf, dst + j * a.elem, tailD);
                    weight += dW;
                }
            }
            else
            {
                Convolution16bNhwcGemmV2_GemmPtr tail_2 = m > 16 ? Convolution16bNhwcGemmV2_Gemm2x2<term, type, flush, 0> : Convolution16bNhwcGemmV2_Gemm1x2<term, type, flush, 0>;
                Convolution16bNhwcGemmV2_GemmPtr tail_1 = m > 16 ? Convolution16bNhwcGemmV2_Gemm2x1<term, type, flush, 0> : Convolution16bNhwcGemmV2_Gemm1x1<term, type, flush, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_ps(bias + j + 0);
                    _bias[1] = _mm512_loadu_ps(bias + j + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16));
                    buf = (term == Term16bInterim ? sum + j : buf);
                    if (dN > F)
                        tail_2(src, p, a, K, m, dN, zero, weight, _bias, _params, sum + j, buf, dB, dst + j * a.elem, tailD);
                    else
                        tail_1(src, p, a, K, m, dN, zero, weight, _bias, _params, sum + j, buf, dB, dst + j * a.elem, tailD);
                    weight += dW;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type, int apply> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = Convolution16bNhwcGemmV2_Gemm<Term16bInterim, SimdConvolutionActivationIdentity, 0, 0>;
            if (p.dstT == SimdTensorData16b)
                gemm[1] = Convolution16bNhwcGemmV2_Gemm<Term16bLast16b, type, 0, apply>;
            else
                gemm[1] = Convolution16bNhwcGemmV2_Gemm<Term16bLast32f, type, 0, apply>;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam & a, GemmPtr* gemm)
        {
            size_t lastMacroK = a.bufK - AlignLoAny(a.bufK - 1, a.macroK);
            if (lastMacroK > 224)
                SetGemm<type, 1>(p, a, gemm);
            else if (lastMacroK > 96)
                SetGemm<type, 2>(p, a, gemm);
            else if (lastMacroK > 32)
                SetGemm<type, 4>(p, a, gemm);
            else
                SetGemm<type, 8>(p, a, gemm);
        }

        SynetConvolution16bNhwcGemmV2::SynetConvolution16bNhwcGemmV2(const ConvParam & p)
            : Base::SynetConvolution16bNhwcGemmV2(p)
        {
            SetAlgParam();
            AlgParam& a = _alg; 
            if (a.tmpBuf)
            {
                if (_is1x1)
                {
                    if (_src16b)
                    {
                        _conv1x1 = Convolution16bNhwcGemmV2_Reorder1x1R;
                    }
                    else
                    {
                        _conv1x1 = a.reorderType ? Convolution16bNhwcGemmV2_Convert1x1R : Convolution16bNhwcGemmV2_Convert1x1D;
                    }
                }
                else
                {
                    assert(0);
                }
            }
            //if (_src16b)
            //{
            //    if (_is1x1 && a.K == a.bufK)
            //        _convert = NULL;
            //    else
            //    {
            //        if (_is1x1 && a.batch == 1)
            //        {
            //            _convert = Reorder16bNhwcGemm1x1R;
            //            a.reorderType = 1;
            //        }
            //        else
            //        {
            //            if (Aligned(p.srcC, 32) && a.batch == 1 && Aligned(p.dstW, a.F))
            //            {
            //                _convert = Reorder16bNhwcGemmR;
            //                a.reorderType = 1;
            //            }
            //            else
            //            {
            //                if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
            //                    _convert = Reorder16bNhwcGemmD_1d32ck;
            //                else
            //                    _convert = Reorder16bNhwcGemmD;
            //            }
            //        }
            //    }
            //}
            //else
            //{
            //    if (_is1x1)
            //    {
            //        if (a.batch == 1/* && a.macroK < a.bufK*/)
            //        {
            //            _convert = Convert16bNhwcGemm1x1R;
            //            a.reorderType = 1;
            //        }
            //        else
            //        {
            //            _convert = Convert16bNhwcGemm1x1D;
            //            a.reorderType = 0;
            //        }
            //    }
            //    else
            //    {
            //        if (p.srcC == AlignLo(p.srcC, 32) && a.batch == 1 && Aligned(p.dstW, a.F))
            //        {
            //            _convert = Convert16bNhwcGemmR;
            //            a.reorderType = 1;
            //        }
            //        else
            //        {
            //            if (p.IsDilation(1) && p.srcC <= 8 && p.srcC * p.kernelX <= 32)
            //                _convert = Convert16bNhwcGemmD_1d32ck;
            //            else
            //                _convert = Convert16bNhwcGemmD;
            //            a.reorderType = 0;
            //        }
            //    }
            //}
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetGemm<SimdConvolutionActivationIdentity>(p, _alg, _gemm); break;
            case SimdConvolutionActivationRelu: SetGemm<SimdConvolutionActivationRelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationLeakyRelu: SetGemm<SimdConvolutionActivationLeakyRelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationRestrictRange: SetGemm<SimdConvolutionActivationRestrictRange>(p, _alg, _gemm); break;
            case SimdConvolutionActivationPrelu: SetGemm<SimdConvolutionActivationPrelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationElu: SetGemm<SimdConvolutionActivationElu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationHswish: SetGemm<SimdConvolutionActivationHswish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationMish: SetGemm<SimdConvolutionActivationMish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationHardSigmoid: SetGemm<SimdConvolutionActivationHardSigmoid>(p, _alg, _gemm); break;
            case SimdConvolutionActivationSwish: SetGemm<SimdConvolutionActivationSwish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationGelu: SetGemm<SimdConvolutionActivationGelu>(p, _alg, _gemm); break;
            default: assert(0);
            }
        }
    }
#endif
}
