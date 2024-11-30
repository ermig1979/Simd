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

        static void Reorder16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
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

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[1] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[3] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((dstC - 16) * 4);
            conf.colsb[2] = 64;
            conf.colsb[3] = uint16_t((dstC - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((dstC - 16) * 4);
            _tile_loadconfig(&conf);

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
            for (size_t sc = 0; sc < srcC; sc += 32, src0 += stepS, src1 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (type)
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

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[2] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

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
            for (size_t sc = 0; sc < srcC; sc += 32, src0 += stepS, src1 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);

            if (type)
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

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[1] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t(dstC - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t(dstC - 16) * 4;
            _tile_loadconfig(&conf);

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
            for (size_t sc = 0; sc < srcC; sc += 32, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);

            if (type)
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

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 512 : 32, strideS = a.reorderType ? 64 : dS * 2;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

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

            if (type)
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
            Convolution16bNhwcGemmPtr body_2 = Convolution16bNhwcGemm_32x32<term, type>;
            Convolution16bNhwcGemmPtr tail_2 = m > 16 ? Convolution16bNhwcGemm_32x32<term, type> : Convolution16bNhwcGemm_16x32<term, type>;
            Convolution16bNhwcGemmPtr body_1 = Convolution16bNhwcGemm_32x16<term, type>;
            Convolution16bNhwcGemmPtr tail_1 = m > 16 ? Convolution16bNhwcGemm_32x16<term, type> : Convolution16bNhwcGemm_16x16<term, type>;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

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
                    for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                        body_2(s, p, a, srcC, n, dC, zero, weight, _bias, _params, b, d);
                    if (m)
                        tail_2(s, p, a, srcC, m, dC, zero, weight, _bias, _params, b, d);
                }
                else
                {
                    for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                        body_1(s, p, a, srcC, n, dC, zero, weight, _bias, _params, b, d);
                    if (m)
                        tail_1(s, p, a, srcC, m, dC, zero, weight, _bias, _params, b, d);
                }
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNhwcGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if(p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast32f, type>;
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
                    _convert = Reorder16bNhwcGemm;
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
                    if (p.srcC == AlignLo(p.srcC, 32) && a.batch == 1)
                    {
                        _convert = Convert16bNhwcGemmR;
                        a.reorderType = 1;
                    }
                    else
                    {
                        _convert = Convert16bNhwcGemmD;
                        a.reorderType = 0;
                    }
                }
            }
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
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
