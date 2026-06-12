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
        typedef Base::SynetQuantizedConvolutionNhwcGemmV1::GemmPtr Gemm;

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
            assert(Aligned(p.srcC, 64));
            size_t K = a.bufK, C = p.srcC, kcX = p.kernelX * C;
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
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
                                        Avx512bw::Copy(ps + sc, row);
                                }
                                else
                                {
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
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
                if (Aligned(p.srcC, 64) && a.batch == 1 && Aligned(p.dstW, a.F))
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
        }
    }
#endif
}
