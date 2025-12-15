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

        //------------------------------------------------------------------------------------------------

        template<int tile, int order> static SIMD_INLINE void LoadS(const void* ptr, int stride)
        {
            switch (tile)
            {
            case 4: if (order) _tile_loadd(4, ptr, stride); else _tile_stream_loadd(4, ptr, stride); break;
            case 5: if (order) _tile_loadd(5, ptr, stride); else _tile_stream_loadd(5, ptr, stride); break;
            case 6: if (order) _tile_loadd(6, ptr, stride); else _tile_stream_loadd(6, ptr, stride); break;
            case 7: if (order) _tile_loadd(7, ptr, stride); else _tile_stream_loadd(7, ptr, stride); break;
            }
        }

        template<int tile, int order> static SIMD_INLINE void LoadW(const void* ptr, int stride)
        {
            switch (tile)
            {
            case 4: if (order) _tile_stream_loadd(4, ptr, stride); else _tile_loadd(4, ptr, stride); break;
            case 5: if (order) _tile_stream_loadd(5, ptr, stride); else _tile_loadd(5, ptr, stride); break;
            case 6: if (order) _tile_stream_loadd(6, ptr, stride); else _tile_loadd(6, ptr, stride); break;
            case 7: if (order) _tile_stream_loadd(7, ptr, stride); else _tile_loadd(7, ptr, stride); break;
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int start> static SIMD_INLINE void Apply2x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[start + 0]), params, start + 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 1) * F), bias[start + 1]), params, start + 1);
            std::cout << " buf " << buf[ + (start + 0) * F] << " ";
            SIMD_LOG(f0);
            //SIMD_LOG(bias[start]);
            if (term == Term16bLast16b)
            {
                _mm512_mask_storeu_epi16((uint16_t*)(ptr + start * DF), tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + start * DF), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + start * DF));
            }
            else
            {
                _mm512_storeu_ps((float*)(ptr + (start + 0) * A), f0);
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 0) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 0) * A));
                _mm512_mask_storeu_ps((float*)(ptr + (start + 1) * A), (__mmask16)tail, f1);
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 1) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 1) * A));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int flush, int start> static SIMD_INLINE void Apply2xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) Apply2x1<term, type, flush, start>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply2x1<term, type, flush, start>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply2x1<term, type, flush, start>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply2x1<term, type, flush, start>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply2x1<term, type, flush, start>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply2x1<term, type, flush, start>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply2x1<term, type, flush, start>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply2x1<term, type, flush, start>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int start> static SIMD_INLINE void Apply1x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[0]), params, 0);
            if (term == Term16bLast16b)
            {
                _mm256_mask_storeu_epi16((uint16_t*)(ptr + start * DF), (__mmask16)tail, (__m256i)_mm512_cvtneps_pbh(f0));
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + start * DF), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + start * DF));
            }
            else
            {
                _mm512_storeu_ps((float*)(ptr + (start + 0) * A), f0);
                if (flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 0) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 0) * A));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int flush, int start> static SIMD_INLINE void Apply1xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) Apply1x1<term, type, flush, start>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply1x1<term, type, flush, start>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply1x1<term, type, flush, start>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply1x1<term, type, flush, start>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply1x1<term, type, flush, start>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply1x1<term, type, flush, start>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply1x1<term, type, flush, start>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply1x1<term, type, flush, start>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_1x32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst0, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + 2 * F;

            int srcC32 = (int)a.bufK - 32, applyC = apply ? (32 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            LoadS<4, order>(src0, strideS);
            LoadW<6, order>(weight0 + sc * dW, strideW);
            Apply2xN<term, type, apply, flush, 0>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
            for (; sc < applyC; src1 += stepS)
            {
                LoadW<7, order>(weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                LoadS<5, order>(src1, strideS);
                Apply2xN<term, type, apply, flush, 0>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                LoadW<6, order>(weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            for (; sc < srcC32; src1 += stepS)
            {
                LoadW<7, order>(weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                LoadS<5, order>(src1, strideS);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                LoadW<6, order>(weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            LoadW<7, order>(weight1 + sc * dW, strideW);
            LoadS<5, order>(src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf1 + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf1 + F, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf1 + 16 * dB + 0, strideB);
            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, buf1 + 16 * dB + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Nx32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dW = (int)a.microD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 32 * dB;

            size_t ds = 0;
            Convolution16bNhwcGemm_1x32x32<term, type, 0, 0, order>(src0, p, a, dstS, dstC, weight0, bias, params, buf0, buf1, dst, tailD), ds += 32;
            for (; ds < dstS; ds += 32)
            {
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_1x32x32<term, type, apply, flush, order>(src0 + ds * dS, p, a, dstS - ds, dstC, weight0, bias, params, buf0, buf1, dst + (ds - 32) * dD, tailD);
            }
            uint8_t* dst1 = dst + (ds - 32) * dD;
            dstS -= ds - 32;
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    Apply2xN<term, type, 8, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply2xN<term, type, 1, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
            }
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_1x16x64(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst0, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            int srcC32 = (int)a.bufK - 32, applyC = apply ? (16 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            LoadS<4, order>(src0, strideS);
            Apply2xN<term, type, apply, flush, 0>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
            LoadW<5, order>(weight0 + 0 * DF, strideW);
            Apply2xN<term, type, apply, flush, 2>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
            for (; sc < applyC; sc += 32)
            {
                LoadW<6, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                Apply2xN<term, type, apply, flush, 0>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                LoadW<7, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 6);
                Apply2xN<term, type, apply, flush, 2>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                LoadW<5, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 7);
                _tile_dpbf16ps(3, 4, 5);
                src0 += stepS;
                weight0 += 32 * dW;
                LoadS<4, order>(src0, strideS);
                LoadW<5, order>(weight0 + 0 * DF, strideW);
            }
            for (; sc < srcC32; sc += 32)
            {
                LoadW<6, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                LoadW<7, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 6);
                LoadW<5, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 7);
                _tile_dpbf16ps(3, 4, 5);
                src0 += stepS;
                weight0 += 32 * dW;
                LoadS<4, order>(src0, strideS);
                LoadW<5, order>(weight0 + 0 * DF, strideW);
            }
            LoadW<6, order>(weight0 + 1 * DF, strideW);
            _tile_dpbf16ps(0, 4, 5);
            _tile_stored(0, buf1 + 0 * F, strideB);
            LoadW<7, order>(weight0 + 2 * DF, strideW);
            _tile_dpbf16ps(1, 4, 6);
            _tile_stored(1, buf1 + 1 * F, strideB);
             LoadW<5, order>(weight0 + 3 * DF, strideW);
            _tile_dpbf16ps(2, 4, 7);
            _tile_stored(2, buf1 + 2 * F, strideB);
            _tile_dpbf16ps(3, 4, 5);
            _tile_stored(3, buf1 + 3 * F, strideB);
            std::cout << " strideB " << strideB << std::endl;
            Simd::Log(buf1 + 0, 32, " buf1 + 0", 1);
            Simd::Log(buf1 + 32, 32, " buf1 + 32", 1);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Nx16x64(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dW = (int)a.microD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 16 * dB;
            std::cout << " buf1 - buf0 " << buf1 - buf0 << std::endl;
            size_t ds = 0;
            Convolution16bNhwcGemm_1x16x64<term, type, 0, 0, order>(src0, p, a, dstS, dstC, weight0, bias, params, buf0, buf1, dst, tailD), ds += 16;
            for (; ds < dstS; ds += 16)
            {
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_1x16x64<term, type, apply, flush, order>(src0 + ds * dS, p, a, dstS - ds, dstC, weight0, bias, params, buf0, buf1, dst + (ds - 16) * dD, tailD);
            }
            uint8_t* dst1 = dst + (ds - 16) * dD;
            dstS -= ds - 16;
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                {
                    Apply2xN<term, type, 8, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params);
                    Apply2xN<term, type, 8, flush, 2>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                }
                for (; ds < dstS; ++ds)
                {
                    Apply2xN<term, type, 1, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params);
                    Apply2xN<term, type, 1, flush, 2>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                }
            }
        }

        //--------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int order> void Convolution16bNhwcGemm_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + 2 * F;

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            int srcC32 = (int)a.bufK - 32, sc = 0;
            LoadS<4, order>(src0, strideS);
            LoadW<6, order>(weight0 + sc * dW, strideW);
            for (; sc < srcC32; src1 += stepS)
            {
                LoadW<7, order>(weight1 + sc * dW, strideW);
                LoadS<5, order>(src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                LoadW<6, order>(weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            LoadW<7, order>(weight1 + sc * dW, strideW);
            LoadS<5, order>(src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf + F, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, buf + 16 * dB + F, strideB);

            size_t ds = 0, dstS8 = dstS & (~7);
            for (; ds < dstS8; ds += 8)
                Apply2xN<term, type, 8, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
            for (; ds < dstS; ++ds)
                Apply2xN<term, type, 1, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int order> void Convolution16bNhwcGemm_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            _tile_zero(0);
            _tile_zero(2);

            int srcC32 = (int)a.bufK - 32, sc = 0;
            LoadS<4, order>(src0, strideS);
            for (; sc < srcC32; sc += 32, src1 += stepS)
            {
                LoadW<6, order>(weight0 + sc * dW, strideW);
                LoadS<5, order>(src1, strideS);
                _tile_dpbf16ps(0, 4, 6);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            LoadW<6, order>(weight0 + sc * dW, strideW);
            LoadS<5, order>(src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf + 16 * dB + 0, strideB);

            size_t ds = 0, dstS8 = dstS & (~7);
            for (; ds < dstS8; ds += 8)
                Apply1xN<term, type, 8, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
            for (; ds < dstS; ++ds)
                Apply1xN<term, type, 1, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int order> void Convolution16bNhwcGemm_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* weight1 = weight0 + 2 * F;

            _tile_zero(0);
            _tile_zero(1);

            int srcC32 = (int)a.bufK - 32, sc = 0;
            LoadW<6, order>(weight0 + sc * dW, strideW);
            for (; sc < srcC32; src0 += stepS)
            {
                LoadS<4, order>(src0, strideS);
                LoadW<7, order>(weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                LoadW<6, order>(weight0 + sc * dW, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            LoadS<4, order>(src0, strideS);
            LoadW<7, order>(weight1 + sc * dW, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf + F, strideB);

            size_t ds = 0, dstS8 = dstS & (~7);
            for (; ds < dstS8; ds += 8)
                Apply2xN<term, type, 8, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
            for (; ds < dstS; ++ds)
                Apply2xN<term, type, 1, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int order> void Convolution16bNhwcGemm_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            _tile_zero(0);

            for (size_t sc = 0, srcC = a.bufK; sc < srcC; sc += 32, src0 += stepS)
            {
                LoadS<4, order>(src0, strideS);
                LoadW<6, order>(weight0 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            _tile_stored(0, buf + 0, strideB);

            size_t ds = 0, dstS8 = dstS & (~7);
            for (; ds < dstS8; ds += 8)
                Apply1xN<term, type, 8, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
            for (; ds < dstS; ++ds)
                Apply1xN<term, type, 1, flush, 0>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
        }

        typedef void (*Convolution16bNhwcGemmPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD);

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Macro32x32(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * a.microD;
            size_t dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            Convolution16bNhwcGemmPtr tail_2, body_1, tail_1;
            if (nn)
            {
                bool avoidSrcOverflow = !(a.bufK == p.srcC && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16); 
                size_t nm = n1 - m;
                tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, flush, order> : Convolution16bNhwcGemm_16x32<term, type, flush, order>;
                body_1 = Convolution16bNhwcGemm_32x16<term, type, flush, order>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, flush, order> : Convolution16bNhwcGemm_16x16<term, type, flush, order>;
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
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                    {
                        __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC) : (__mmask32)TailMask16(dC - F);
                        for (; i < nn;)
                        {
                            size_t dn = Simd::Min(n * 8, nn - i);
                            Convolution16bNhwcGemm_Nx32x32<term, type, apply, flush, order>(s + i * dS, p, a, dn, dC, weight, _bias, _params, buf, d + i * dD, tailD);
                            i += dn;
                        }
                        if (m)
                            tail_2(s + nm * dS, p, a, m, weight, _bias, _params, buf, d + nm * dD, tailD);
                    }
                    else
                    {
                        __mmask32 tailD = TailMask32(dC);
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, n, weight, _bias, _params, buf, d + i * dD, tailD);
                        if (m)
                            tail_1(s + nm * dS, p, a, m, weight, _bias, _params, buf, d + nm * dD, tailD);
                    }
                    weight += dW;
                }
            }
            else
            {
                tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, flush, order> : Convolution16bNhwcGemm_16x32<term, type, flush, order>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, flush, order> : Convolution16bNhwcGemm_16x16<term, type, flush, order>;
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
                    {
                        __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC) : (__mmask32)TailMask16(dC - F);
                        tail_2(s, p, a, m, weight, _bias, _params, buf, d, tailD);
                    }
                    else
                    {
                        __mmask32 tailD = TailMask32(dC);
                        tail_1(s, p, a, m, weight, _bias, _params, buf, d, tailD);
                    }
                    weight += dW;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Macro16x64(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 16, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * a.microD;
            size_t dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _params[4], _bias[4];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            SetTileConfFull();
            size_t i = 0;
            for (; i < nn;)
            {
                size_t dn = Simd::Min(n * 8, nn - i);
                const uint16_t* s = src + i * dS;
                const uint16_t* w = weight;
                uint8_t* d = dst + i * dD;
                for (size_t dc = 0; dc < dstC; dc += 4 * F)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC - 2 * F) : (__mmask32)TailMask16(dC - 3 * F);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                    _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                    _bias[2] = _mm512_loadu_ps(bias + dc + 2 * F);
                    _bias[3] = _mm512_loadu_ps(bias + dc + 3 * F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                        _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                        _params[2] = _mm512_loadu_ps(params + dc + 2 * F);
                        _params[3] = _mm512_loadu_ps(params + dc + 3 * F);
                    }
                    Convolution16bNhwcGemm_Nx16x64<term, type, apply, flush, order>(s, p, a, dn, dC, w, _bias, _params, buf, d + dc * a.elem, tailD);
                    w += dW;
                }
                i += dn;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush, int order> SIMD_INLINE void SetMacro32x32(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 1024)
                convolution = Convolution16bNhwcGemm_Macro32x32<term, type, 1, flush, order>;
            else if (a.bufK >= 512)
                convolution = Convolution16bNhwcGemm_Macro32x32<term, type, 2, flush, order>;
            else if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_Macro32x32<term, type, 4, flush, order>;
            else if (a.bufK >= 128)
                convolution = Convolution16bNhwcGemm_Macro32x32<term, type, 8, flush, order>;
            else
                convolution = NULL;
        }

        template <Term16bType term, SimdConvolutionActivationType type, int flush, int order> SIMD_INLINE void SetMacro16x64(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 512)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 1, flush, order>;
            else if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 2, flush, order>;
            else if (a.bufK >= 128)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 4, flush, order>;
            else if (a.bufK >= 64)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 8, flush, order>;
            else
                convolution = NULL;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution& convolution)
        {
            bool is16x64 = a.microD == 64 && Aligned(p.dstC, 64) && Aligned(p.dstH * p.dstW, 16) && a.bufK <= 512;
            if (p.dstT == SimdTensorData16b)
            {
                if(is16x64)
                    SetMacro16x64<Term16bLast16b, type, 0, 0>(p, a, convolution);
                else
                    SetMacro32x32<Term16bLast16b, type, 0, 0>(p, a, convolution);
            }
            else
            {
                SetMacro32x32<Term16bLast32f, type, 0, 0>(p, a, convolution);
            }
        }

        SynetConvolution16bNhwcGemmV1::SynetConvolution16bNhwcGemmV1(const ConvParam & p)
            : Base::SynetConvolution16bNhwcGemmV1(p)
        {
            if(Aligned(p.dstC, 64) && Aligned(p.dstH * p.dstW, 16) && p.srcC <= 512)
                SetAlgParam(F, F * 4, F * 1, 32, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            else
                SetAlgParam(F, F * 2, F * 2, 32, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
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
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
            default: assert(0);
            }
        }
    }
#endif
}
