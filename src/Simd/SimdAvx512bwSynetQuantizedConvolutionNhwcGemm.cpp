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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        typedef Base::SynetQuantizedConvolutionNhwcGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmReorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t C = p.srcC, C64 = AlignLo(C, 64), K = a.bufK, kcX = p.kernelX * C;
            __mmask64 gM = TailMask64(K - a.K), cM= TailMask64(C - C64);
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

        static void QuantizedConvolutionNhwcGemmReorder1d(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            //SIMD_PERF_BEG(ToStr(p.srcC));
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

        static void QuantizedConvolutionNhwcGemmReorder1d16c(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
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

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void QuantizedConvolutionNhwcGemm_i2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, 
            const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const int8_t* weight1 = weight0 + a.bufK * F;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm512_loadu_si512(buf + 0x0 * dB + 0), d01 = _mm512_loadu_si512(buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = _mm512_loadu_si512(buf + 0x1 * dB + 0), d11 = _mm512_loadu_si512(buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = _mm512_loadu_si512(buf + 0x2 * dB + 0), d21 = _mm512_loadu_si512(buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = _mm512_loadu_si512(buf + 0x3 * dB + 0), d31 = _mm512_loadu_si512(buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = _mm512_loadu_si512(buf + 0x4 * dB + 0), d41 = _mm512_loadu_si512(buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = _mm512_loadu_si512(buf + 0x5 * dB + 0), d51 = _mm512_loadu_si512(buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = _mm512_loadu_si512(buf + 0x6 * dB + 0), d61 = _mm512_loadu_si512(buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = _mm512_loadu_si512(buf + 0x7 * dB + 0), d71 = _mm512_loadu_si512(buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = _mm512_loadu_si512(buf + 0x8 * dB + 0), d81 = _mm512_loadu_si512(buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = _mm512_loadu_si512(buf + 0x9 * dB + 0), d91 = _mm512_loadu_si512(buf + 0x9 * dB + F);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(buf + 0xA * dB + 0), dA1 = _mm512_loadu_si512(buf + 0xA * dB + F);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(buf + 0xB * dB + 0), dB1 = _mm512_loadu_si512(buf + 0xB * dB + F);
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                    if (M > 0x1) d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                    if (M > 0x2) d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                    if (M > 0x3) d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                    if (M > 0x4) d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                    if (M > 0x5) d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                    if (M > 0x6) d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                    if (M > 0x7) d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                    if (M > 0x8) d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                    if (M > 0x9) d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                    if (M > 0xA) dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                    if (M > 0xB) dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    w1 = _mm512_loadu_si512((__m512i*)weight1);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0), Madd4<true>(d51, s0, w1);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0), Madd4<true>(d61, s0, w1);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0), Madd4<true>(d71, s0, w1);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0), Madd4<true>(d81, s0, w1);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0), Madd4<true>(d91, s0, w1);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s0, w1);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s0, w1);
                    weight0 += A, weight1 += A;
                }
                __mmask16 tail = TailMask16(dstC - F);
                if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm512_loadu_si512(buf + 0x0 * dB + 0);
                    if (M > 0x1) d10 = _mm512_loadu_si512(buf + 0x1 * dB + 0);
                    if (M > 0x2) d20 = _mm512_loadu_si512(buf + 0x2 * dB + 0);
                    if (M > 0x3) d30 = _mm512_loadu_si512(buf + 0x3 * dB + 0);
                    if (M > 0x4) d40 = _mm512_loadu_si512(buf + 0x4 * dB + 0);
                    if (M > 0x5) d50 = _mm512_loadu_si512(buf + 0x5 * dB + 0);
                    if (M > 0x6) d60 = _mm512_loadu_si512(buf + 0x6 * dB + 0);
                    if (M > 0x7) d70 = _mm512_loadu_si512(buf + 0x7 * dB + 0);
                    if (M > 0x8) d80 = _mm512_loadu_si512(buf + 0x8 * dB + 0);
                    if (M > 0x9) d90 = _mm512_loadu_si512(buf + 0x9 * dB + 0);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(buf + 0xA * dB + 0);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(buf + 0xB * dB + 0);
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_setzero_si512();
                    if (M > 0x1) d10 = _mm512_setzero_si512();
                    if (M > 0x2) d20 = _mm512_setzero_si512();
                    if (M > 0x3) d30 = _mm512_setzero_si512();
                    if (M > 0x4) d40 = _mm512_setzero_si512();
                    if (M > 0x5) d50 = _mm512_setzero_si512();
                    if (M > 0x6) d60 = _mm512_setzero_si512();
                    if (M > 0x7) d70 = _mm512_setzero_si512();
                    if (M > 0x8) d80 = _mm512_setzero_si512();
                    if (M > 0x9) d90 = _mm512_setzero_si512();
                    if (M > 0xA) dA0 = _mm512_setzero_si512();
                    if (M > 0xB) dB0 = _mm512_setzero_si512();
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0);
                    weight0 += A;
                }
                __mmask16 tail = TailMask16(dstC);
                if (M > 0x0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x5) Save1<term, type>(dst, buf, d50, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x6) Save1<term, type>(dst, buf, d60, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x7) Save1<term, type>(dst, buf, d70, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x8) Save1<term, type>(dst, buf, d80, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0x9) Save1<term, type>(dst, buf, d90, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0xA) Save1<term, type>(dst, buf, dA0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
                if (M > 0xB) Save1<term, type>(dst, buf, dB0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail), dst += dD, buf += dB;
            }
        }

        typedef void(*QuantizedConvolutionNhwcGemm_i2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, int update, const int8_t* weight,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> QuantizedConvolutionNhwcGemm_i2xM_Ptr GetQuantizedConvolutionNhwcGemm_i2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x1>;
            case 0x2: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x2>;
            case 0x3: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x3>;
            case 0x4: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x4>;
            case 0x5: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x5>;
            case 0x6: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x6>;
            case 0x7: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x7>;
            case 0x8: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x8>;
            case 0x9: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0x9>;
            case 0xA: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0xA>;
            case 0xB: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0xB>;
            case 0xC: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcGemm_i2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, 
            int update, const int8_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            QuantizedConvolutionNhwcGemm_i2xM_Ptr convolution_i2xN = GetQuantizedConvolutionNhwcGemm_i2xM<term, type>(n);
            QuantizedConvolutionNhwcGemm_i2xM_Ptr convolution_i2xM = GetQuantizedConvolutionNhwcGemm_i2xM<term, type>(m);

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
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 0);
                _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 1);
                _sNorm[0] = _mm512_loadu_ps(sNorm + dc + 0);
                _sNorm[1] = _mm512_loadu_ps(sNorm + dc + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                const uint8_t* s = src;
                int32_t* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_i2xN(s, p, a, srcC, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_i2xM(s, p, a, srcC, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_i2<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                convolutions[1] = NULL;
            }
        }

        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : Avx2::SynetQuantizedConvolutionNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src8u)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                {
                    if (p.IsDilation(1) && p.srcC <= 16 && p.srcC*p.kernelX <= 64)
                        _convert = QuantizedConvolutionNhwcGemmReorder1d16c;
                    else if (p.IsDilation(1))
                        _convert = QuantizedConvolutionNhwcGemmReorder1d;
                    else
                        _convert = QuantizedConvolutionNhwcGemmReorder;
                }
            }
            else
                assert(0);
            Set(p, _alg, _convolutions);
        }
    }
#endif
}
