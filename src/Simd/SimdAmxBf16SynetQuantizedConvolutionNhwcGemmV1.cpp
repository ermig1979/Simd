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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcGemmV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemmV1::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmV1_ReorderD(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t C = p.srcC, C64 = AlignLo(C, 64), K = a.bufK, kcX = p.kernelX * C;
            __mmask64 gM = TailMask64(K - a.K), cM = TailMask64(C - C64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                    Copy(src + (sy * p.srcW + sx) * C, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                                pd += C;
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kcX);
                            pd += kcX;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderD1d(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(p.IsDilation(1));
            size_t C = p.srcC, C64 = AlignLo(C, 64), K = a.bufK, kC = p.kernelX * C, kC64 = AlignLo(kC, 64), sX = p.strideX, cW = p.srcW * C, kY = p.kernelY, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask64 gM = TailMask64(K - a.K), cM = TailMask64(C - C64), kcM = TailMask64(kC - kC64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kC;
                    for (size_t ky = 0, k = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    Copy(src + sy * cW + sxc, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
                if (dy >= dyB && dy < dyE)
                {
                    const uint8_t* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint8_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kC)
                            Copy(ps + ky * cW, kC64, kcM, pd);
                        SetZero(pd, _mm512_setzero_si512(), gM);
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint8_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                Copy(src + sy * cW + sxcB, kC64, kcM, pd);
                            else
                                SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                        SetZero(pd, _mm512_setzero_si512(), gM);
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kC;
                    for (size_t ky = 0, k = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    Copy(src + sy * cW + sxc, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderD1d16c(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(p.IsDilation(1) && p.srcC <= 16 && p.srcC * p.kernelX <= 64);
            size_t K = a.bufK, C = p.srcC, kcX = p.kernelX * C, sX = p.strideX, cW = p.srcW * C, cwH = cW * p.srcH, kY = p.kernelY, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask64 gM = TailMask64(K - a.K), kcM = TailMask64(kcX);
            __mmask16 cM = TailMask16(C);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi8(pd, cM, _mm_maskz_loadu_epi8(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi8(pd, cM, _mm512_castsi512_si128(_zero));
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                }
                if (dy >= dyB && dy < dyE)
                {
                    const uint8_t* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint8_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kcX)
                            _mm512_mask_storeu_epi8(pd, kcM, _mm512_maskz_loadu_epi8(kcM, ps + ky * cW));
                        _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint8_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                _mm512_mask_storeu_epi8(pd, kcM, _mm512_maskz_loadu_epi8(kcM, src + sy * cW + sxcB));
                            else
                                _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                        _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi8(pd, cM, _mm_maskz_loadu_epi8(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi8(pd, cM, _mm512_castsi512_si128(_zero));
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderR(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t K = a.bufK, C = p.srcC;
            size_t C64 = AlignLo(C, 64), kcX = p.kernelX * C64;
            assert(C64 == C);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    size_t drB = dr & (~15), drO = dr & 15;
                    uint8_t* row = dst + drB * K + drO * 64;
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    for (size_t sc = 0; sc < C64; sc += 64, row += 1024)
                                        Avx512bw::Copy(ps + sc, row);
                                }
                                else
                                {
                                    for (size_t sc = 0; sc < C64; sc += 64, row += 1024)
                                        SetZero(row, _zero);
                                }
                            }
                        }
                        else
                        {
                            for (size_t sc = 0; sc < kcX; sc += 64, row += 1024)
                                SetZero(row, _zero);
                        }
                    }
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_Reorder1x1D(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t M, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            for (size_t i = 0; i < M; ++i)
            {
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                    Avx512bw::Copy(src + sc, dst + sc);
                if(srcMask)
                    Avx512bw::Copy(src + sc, dst + sc, srcMask);
                src += p.srcC;
                dst += a.bufK;
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_Reorder1x1R(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t M, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                if (srcC64 < p.srcC)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16, srcMask);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                src += p.srcC * 16;
                dst += a.bufK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void QuantizedConvolutionNhwcGemmV1_Gemm1xMx2(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo,
            const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf0, int32_t* buf1, int32_t* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem), dS = (int)a.bufK, dW = 16, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            int dB = term == Term8iInterim ? (int)a.dB : DF, strideB = dB * 4;
            const uint8_t* src1 = src0 + 16 * dS;
            const int8_t* weight1 = weight0 + a.bufK * F;

            if (update)
            {
                int dB = (int)a.dB, strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
                buf0 += 16 * a.dB;
                if (M > 0) _tile_stream_loadd(2, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(3, buf0 + F, strideB);
            }
            else
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
                if (M > 0) _tile_zero(2);
                if (M > 1) _tile_zero(3);
            }

            int sC64 = (int)srcC - 64, aC64 = apply ? (8 * 64 / apply - 64) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC64; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbusd(2, 5, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                sc += 64;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbusd(3, 5, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            }
            for (; sc < sC64; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbusd(2, 5, 6);
                sc += 64;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbusd(3, 5, 7);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            _tile_stream_loadd(5, src1, strideS);
            if (M > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
            buf2 += 16 * dB;
            if (M > 0) _tile_dpbusd(2, 5, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 0) _tile_stored(2, buf2 + 0, strideB);
            if (M > 1) _tile_dpbusd(3, 5, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 1) _tile_stored(3, buf2 + F, strideB);
        }

        template<Term8iType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void QuantizedConvolutionNhwcGemmV1_Gemm1xMx1(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo,
            const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf0, int32_t* buf1, int32_t* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem), dS = (int)a.bufK, dW = 16, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            int dB = term == Term8iInterim ? (int)a.dB : DF, strideB = dB * 4;
            const int8_t* weight1 = weight0 + a.bufK * F;

            if (update)
            {
                int dB = (int)a.dB, strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
            }
            else
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
            }

            int sC64 = (int)srcC - 64, aC64 = apply ? (8 * 64 / apply - 64) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC64;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                sc += 64;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            for (; sc < sC64;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                sc += 64;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            if (M > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
        }

        template<Term8iType term, SimdConvolutionActivationType type, int flush, int M, int apply> void QuantizedConvolutionNhwcGemmV1_GemmNxMx2(const uint8_t* src0, 
            const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, 
            const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf1, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK;
            int32_t* buf2 = buf1 + 1024;
            if (term == Term8iInterim)
            {
                for (size_t cds = 0; cds < dstS; cds += 32)
                {
                    size_t bds = cds;
                    if (cds + 16 >= dstS)
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 16, cds);
                        QuantizedConvolutionNhwcGemmV1_Gemm1xMx1<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, update, weight0, 
                            sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, sum0 + bds * dB, NULL, sum0 + bds * dB, NULL, tailD);
                    }
                    else
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 32, cds);
                        QuantizedConvolutionNhwcGemmV1_Gemm1xMx2<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, update, weight0, 
                            sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, sum0 + bds * dB, NULL, sum0 + bds * dB, NULL, tailD);
                    }
                }
            }
            else
            {
                size_t cds = 0, pds = 0;
                QuantizedConvolutionNhwcGemmV1_Gemm1xMx2<term, type, flush, M, 0>(src0, p, a, srcC, update, weight0, 
                    sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, sum0, NULL, buf2, dst, tailD), cds += 32;
                for (; cds < dstS; pds = cds, cds += 32)
                {
                    Swap(buf1, buf2);
                    size_t bds = cds;
                    if (cds + 16 >= dstS)
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 16, cds);
                        QuantizedConvolutionNhwcGemmV1_Gemm1xMx1<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, update, weight0, 
                            sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, sum0 + bds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                    else
                    {
                        if (a.reorderType == 0)
                            cds = Simd::Min(dstS - 32, cds);
                        QuantizedConvolutionNhwcGemmV1_Gemm1xMx2<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, update, weight0, 
                            sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, sum0 + bds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                }
                uint8_t* dst1 = dst + pds * dD;
                dstS -= pds;
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, M, 8>(dst1 + ds * dD, dD, buf2 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, M, 1>(dst1 + ds * dD, dD, buf2 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int flush> void QuantizedConvolutionNhwcGemmV1_Gemm2x2(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, 
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, 
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)a.dB * 4;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const uint8_t* src1 = src0 + dS * 16;
            const int8_t* weight1 = weight0 + a.bufK * F;
            int32_t* buf1 = buf0 + 16 * DF, * sum1 = sum0 + 16 * a.dB;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideB);
                _tile_stream_loadd(1, sum0 + F, strideB);
                _tile_stream_loadd(2, sum1 + 0, strideB);
                _tile_stream_loadd(3, sum1 + F, strideB);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC64; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbusd(0, 4, 6);
                _tile_dpbusd(1, 4, 7);
                src0 += stepS;
                sc += 64;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbusd(2, 5, 6);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);

            if (term == Term8iInterim)
            {
                _tile_stored(0, sum0 + 0, strideB);
                _tile_stored(1, sum0 + F, strideB);
                _tile_stored(2, sum1 + 0, strideB);
                _tile_stored(3, sum1 + F, strideB);
            }
            else
            {
                _tile_stored(0, buf0 + 0, DA);
                _tile_stored(1, buf0 + F, DA);
                _tile_stored(2, buf1 + 0, DA);
                _tile_stored(3, buf1 + F, DA);
            }
            if (term == Term8iLast8u)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 2, 8>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 2, 1>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int flush> void QuantizedConvolutionNhwcGemmV1_Gemm2x1(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params,
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)a.dB * 4;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const uint8_t* src1 = src0 + dS * 16;
            int32_t* buf1 = buf0 + 16 * DF, * sum1 = sum0 + 16 * a.dB;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(2, sum1 + 0, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(2);
            }

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC64; src1 += stepS)
            {
                _tile_loadd(6, weight0, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbusd(0, 4, 6);
                src0 += stepS;
                sc += 64;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(2, 5, 6);

            if (term == Term8iInterim)
            {
                _tile_stored(0, sum0 + 0, strideB);
                _tile_stored(2, sum1 + 0, strideB);
            }
            else
            {
                _tile_stored(0, buf0 + 0, DA);
                _tile_stored(2, buf1 + 0, DA);
            }
            if (term == Term8iLast8u)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 1, 8>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 1, 1>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int flush> void QuantizedConvolutionNhwcGemmV1_Gemm1x2(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params,
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)a.dB * 4;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const int8_t* weight1 = weight0 + a.bufK * F;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideB);
                _tile_stream_loadd(1, sum0 + F, strideB);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC64; src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                sc += 64;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);

            if (term == Term8iInterim)
            {
                _tile_stored(0, sum0 + 0, strideB);
                _tile_stored(1, sum0 + F, strideB);
            }
            else
            {
                _tile_stored(0, buf0 + 0, DA);
                _tile_stored(1, buf0 + F, DA);
            }
            if (term == Term8iLast8u)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 2, 8>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 2, 1>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int flush> void QuantizedConvolutionNhwcGemmV1_Gemm1x1(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params,
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst, __mmask32 tailD)
        {
            int dS = (int)a.bufK, dD = int(p.dstC * a.elem), strideW = 64, strideB = (int)a.dB * 4;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideB);
            }
            else
            {
                _tile_zero(0);
            }

            size_t sc = 0;
            for (; sc < srcC; sc += 64, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
            }

            if (term == Term8iInterim)
            {
                _tile_stored(0, sum0 + 0, strideB);
            }
            else
            {
                _tile_stored(0, buf0 + 0, DA);
            }
            if (term == Term8iLast8u)
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, 1, 8>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, 1, 1>(dst + ds * dD, dD, buf0 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
        }

        typedef void (*QuantizedConvolutionNhwcGemmV1_GemmPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, 
            size_t dstC, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, 
            const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst, __mmask32 tailD);

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int flush, int apply> void QuantizedConvolutionNhwcGemmV1_Gemm(
            const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t N, size_t M, size_t K, int update, const int8_t* weight, const int32_t* sBias, 
            const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            size_t n = 32, Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dW = a.bufK * DF, dS = a.bufK, dD = p.dstC * a.elem;

            __m512 _sNorm[2], _iScale, _params[2], _dNorm;
            __m512i _sBias[2], _dZero = _mm512_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            if (Mn)
            {
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + j) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + j) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + j + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + j + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term8iLast8u ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16));
                    buf = (term == Term8iInterim ? sum + j : buf);
                    if (dN > F)
                        QuantizedConvolutionNhwcGemmV1_GemmNxMx2<term, type, flush, 2, apply>(src, p, a, K, M, update, weight, 
                            _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, sum + j, buf, dst + j * a.elem, tailD);
                    else
                        QuantizedConvolutionNhwcGemmV1_GemmNxMx2<term, type, flush, 1, apply>(src, p, a, K, M, update, weight,
                            _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, sum + j, buf, dst + j * a.elem, tailD);
                    weight += dW;
                }
            }
            else
            {
                QuantizedConvolutionNhwcGemmV1_GemmPtr tail_2 = m > 16 ? QuantizedConvolutionNhwcGemmV1_Gemm2x2<term, type, flush> : QuantizedConvolutionNhwcGemmV1_Gemm1x2<term, type, flush>;
                QuantizedConvolutionNhwcGemmV1_GemmPtr tail_1 = m > 16 ? QuantizedConvolutionNhwcGemmV1_Gemm2x1<term, type, flush> : QuantizedConvolutionNhwcGemmV1_Gemm1x1<term, type, flush>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + j) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + j) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + j + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + j + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term8iLast8u ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16));
                    buf = (term == Term8iInterim ? sum + j : buf);
                    if (dN > F)
                        tail_2(src, p, a, K, m, dN, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, 
                            _params, _dNorm, _dZero, sum + j, buf, dst + j * a.elem, tailD);
                    else
                        tail_1(src, p, a, K, m, dN, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, 
                            _params, _dNorm, _dZero, sum + j, buf, dst + j * a.elem, tailD);
                    weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type, int apply> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = QuantizedConvolutionNhwcGemmV1_Gemm<Term8iInterim, SimdConvolutionActivationIdentity, 0, 0>;
            if (p.dstT == SimdTensorData8u)
                gemm[1] = QuantizedConvolutionNhwcGemmV1_Gemm<Term8iLast8u, type, 0, apply>;
            else
            {
                gemm[1] = NULL;
                assert(0);
            }
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            size_t lastMacroK = a.bufK - AlignLoAny(a.bufK - 1, a.macroK);
            if (lastMacroK > 448)
                SetGemm<type, 1>(p, a, gemm);
            else if (lastMacroK > 192)
                SetGemm<type, 2>(p, a, gemm);
            else if (lastMacroK > 64)
                SetGemm<type, 4>(p, a, gemm);
            else
                SetGemm<type, 8>(p, a, gemm);
        }

        SynetQuantizedConvolutionNhwcGemmV1::SynetQuantizedConvolutionNhwcGemmV1(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcGemmV1(p)
        {
            SetAlgParam();
            AlgParam& a = _alg;
            if (_is1x1)
            {
                _convAny = NULL;
                if(a.K == a.bufK)
                    _conv1x1 = NULL;
                else if (a.batch == 1)
                {
                    _conv1x1 = QuantizedConvolutionNhwcGemmV1_Reorder1x1R;
                    a.reorderType = 1;
                }
                else
                {
                    _conv1x1 = QuantizedConvolutionNhwcGemmV1_Reorder1x1D;
                    a.reorderType = 0;
                }
            }
            else
            {
                _conv1x1 = NULL;
                if (Aligned(p.srcC, 64) && a.batch == 1 && a.isAlMaH)
                {
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderR;
                    a.reorderType = 1;
                }
                else if (p.IsDilation(1) && p.srcC <= 16 && p.srcC * p.kernelX <= 64)
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD1d16c;
                else if (p.IsDilation(1))
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD1d;
                else
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD;
            }
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
