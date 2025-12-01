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
                {
                    _m_prefetchw((char*)ptr);
                    //_mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
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

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> static SIMD_INLINE void Apply2x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0 * F), bias[0]), params, 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 1 * F), bias[1]), params, 1);
            if (term == Term16bLast16b)
            {
                _mm512_mask_storeu_epi16((uint16_t*)ptr, tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                //_m_prefetchw((char*)ptr);
                _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
            }
            else
            {
                _mm512_storeu_ps((float*)ptr + 0, f0);
                //_m_prefetchw((char*)(ptr + 0));
                _mm_prefetch((const char*)(ptr + 0), _MM_HINT_NTA);
                _mm512_mask_storeu_ps((float*)ptr + F, (__mmask16)tail, f1);
                //_m_prefetchw((char*)(ptr + A));
                _mm_prefetch((const char*)(ptr + A), _MM_HINT_NTA);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N> static SIMD_INLINE void Apply2xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) Apply2x1<term, type>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply2x1<term, type>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply2x1<term, type>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply2x1<term, type>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply2x1<term, type>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply2x1<term, type>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply2x1<term, type>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply2x1<term, type>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply> void Convolution16bNhwcGemm_1x32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
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

            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * dW, strideW);
            Apply2xN<term, type, apply>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
            for (; sc < applyC; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                Apply2xN<term, type, apply>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * dW, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf1 + 0, strideB);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf1 + F, strideB);
            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf1 + 16 * dB + 0, strideB);
            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, buf1 + 16 * dB + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply> void Convolution16bNhwcGemm_Nx32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dW = (int)a.microD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 32 * dB;

            size_t ds = 0;
            Convolution16bNhwcGemm_1x32x32<term, type, 0>(src0, p, a, dstS, dstC, weight0, bias, params, buf0, buf1, dst, tailD), ds += 32;
            for (; ds < dstS; ds += 32)
            {
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_1x32x32<term, type, apply>(src0 + ds * dS, p, a, dstS - ds, dstC, weight0, bias, params, buf0, buf1, dst + (ds - 32) * dD, tailD);
            }
            uint8_t* dst1 = dst + (ds - 32) * dD;
            dstS -= ds - 32;
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    Apply2xN<term, type, 8>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    Apply2xN<term, type, 1>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
            }
        }

        //--------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
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
            if (stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < srcC32; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * dW, strideW);
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
                _tile_loadd(6, weight0 + sc * dW, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * dW, strideW);
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

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            _tile_zero(0);
            _tile_zero(2);

            int srcC32 = (int)a.bufK - 32, sc = 0;
            if (stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            for (; sc < srcC32; sc += 32, src1 += stepS)
            {
                _tile_loadd(6, weight0 + sc * dW, strideW);
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
            _tile_loadd(6, weight0 + sc * dW, strideW);
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

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* weight1 = weight0 + 2 * F;

            _tile_zero(0);
            _tile_zero(1);

            int srcC32 = (int)a.bufK - 32, sc = 0;
            _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < srcC32; src0 += stepS)
            {
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_loadd(7, weight1 + sc * dW, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * dW, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (stream)
                _tile_stream_loadd(4, src0, strideS);
            else
                _tile_loadd(4, src0, strideS);
            _tile_loadd(7, weight1 + sc * dW, strideW);
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

        template<Term16bType term, SimdConvolutionActivationType type, int stream, int flush> void Convolution16bNhwcGemm_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            _tile_zero(0);

            for (size_t sc = 0, srcC = a.bufK; sc < srcC; sc += 32, src0 += stepS)
            {
                if (stream)
                    _tile_stream_loadd(4, src0, strideS);
                else
                    _tile_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * dW, strideW);
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

        typedef void (*Convolution16bNhwcGemmPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * a.microD;
            size_t dD = p.dstC * a.elem, dS = a.bufK;
            bool bigAlignedDst = Aligned(dst, A) && Aligned(dD, A) && dD * p.dstW * p.dstH > 4 * Base::AlgCacheL1() && 0;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            Convolution16bNhwcGemmPtr body_2, tail_2, body_1, tail_1;
            if (nn)
            {
                bool avoidSrcOverflow = !(a.bufK == p.srcC && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16); //, nn = n1 - m;
                int nm = n1 - m;
                tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, 1, 1> : Convolution16bNhwcGemm_16x32<term, type, 1, 1>;
                body_1 = Convolution16bNhwcGemm_32x16<term, type, 1, 1>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, 1, 1> : Convolution16bNhwcGemm_16x16<term, type, 1, 1>;
                SetTileConfFull();
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    if(bigAlignedDst)
                        body_2 = dc == 0 ? Convolution16bNhwcGemm_32x32<term, type, 0, 2> : Convolution16bNhwcGemm_32x32<term, type, 1, 2>;
                    else
                        body_2 = (dc == 0 && 0) ? Convolution16bNhwcGemm_32x32<term, type, 0, 1> : Convolution16bNhwcGemm_32x32<term, type, 1, 1>;
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
                            int dn = Simd::Min(n * 8, nn - i);
                            if (dS >= 1024)
                                Convolution16bNhwcGemm_Nx32x32<term, type, 1>(s + i * dS, p, a, dn, dC, weight, _bias, _params, buf, d + i * dD, tailD);
                            else if (dS >= 512)
                                Convolution16bNhwcGemm_Nx32x32<term, type, 2>(s + i * dS, p, a, dn, dC, weight, _bias, _params, buf, d + i * dD, tailD);
                            else if (dS >= 256)
                                Convolution16bNhwcGemm_Nx32x32<term, type, 4>(s + i * dS, p, a, dn, dC, weight, _bias, _params, buf, d + i * dD, tailD);
                            else
                                Convolution16bNhwcGemm_Nx32x32<term, type, 8>(s + i * dS, p, a, dn, dC, weight, _bias, _params, buf, d + i * dD, tailD);
                            i += dn;
                        }
                        if (m)
                            tail_2(s + nm * dS, p, a, m, dC, weight, _bias, _params, buf, d + nm * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, n, dC, weight, _bias, _params, buf, d + i * dD);
                        if (m)
                            tail_1(s + nm * dS, p, a, m, dC, weight, _bias, _params, buf, d + nm * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type, 1, 1> : Convolution16bNhwcGemm_16x32<term, type, 1, 1>;
                tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type, 1, 1> : Convolution16bNhwcGemm_16x16<term, type, 1, 1>;
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
                        tail_2(s, p, a, m, dC, weight, _bias, _params, buf, d);
                    else
                        tail_1(s, p, a, m, dC, weight, _bias, _params, buf, d);
                    weight += dW;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution& convolution)
        {
            if (p.dstT == SimdTensorData16b)
                convolution = Convolution16bNhwcGemm_2<Term16bLast16b, type>;
            else
                convolution = Convolution16bNhwcGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNhwcGemmV1::SynetConvolution16bNhwcGemmV1(const ConvParam & p)
            : Base::SynetConvolution16bNhwcGemmV1(p)
        {
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
