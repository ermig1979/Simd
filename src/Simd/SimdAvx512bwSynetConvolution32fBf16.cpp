/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        typedef Base::SynetConvolution32fBf16Nhwc::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvertPtr Convert;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        void ConvolutionBf16NhwcConvertConv(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, uint16_t* dst)
        {
            ptrdiff_t beg = yBeg * p.strideY - p.padY;
            ptrdiff_t end = (yEnd - 1) * p.strideY - p.padY + p.kernelY * p.dilationY;
            src += Max<ptrdiff_t>(0, beg) * p.srcW * p.srcC;
            size_t srcC32 = AlignLo(srcC, 32);
            size_t srcW = p.srcW + p.padX + p.padW;
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (srcC32 < srcC)
            {
                srcMask[0] = TailMask16(srcC - srcC32 - F * 0);
                srcMask[1] = TailMask16(srcC - srcC32 - F * 1);
                dstMask[0] = TailMask32(srcC - srcC32);
            }
            for (ptrdiff_t sy = beg; sy < end; ++sy)
            {
                if ((size_t)sy >= p.srcH)
                {
                    memset(dst, 0, srcW * srcC * 2);
                    dst += srcW * srcC;
                }
                else
                {
                    if (p.padX)
                    {
                        memset(dst, 0, p.padX * srcC * 2);
                        dst += p.padX * srcC;
                    }
                    if (p.srcC == srcC)
                    {
                        Float32ToBFloat16(src, srcC * p.srcW, dst);
                        src += srcC * p.srcW;
                        dst += srcC * p.srcW;
                    }
                    else
                    {
                        for (size_t sx = 0; sx < p.srcW; ++sx)
                        {
                            size_t sc = 0;
                            for (; sc < srcC32; sc += 32)
                                Float32ToBFloat16<false, false>(src + sc, dst + sc, srcMask, dstMask);
                            if (srcC32 < srcC)
                                Float32ToBFloat16<false, true>(src + sc, dst + sc, srcMask, dstMask);
                            src += p.srcC;
                            dst += srcC;
                        }
                    }
                    if (p.padW)
                    {
                        memset(dst, 0, p.padW * srcC * 2);
                        dst += p.padW * srcC;
                    }
                }
            }
        }

        void ConvolutionBf16NhwcConvertGemm(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, uint16_t* dst)
        {
            size_t srcCh2 = AlignHi(srcC, 2);
            size_t srcC32 = AlignLo(srcC, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (srcC32 < srcC)
            {
                srcMask[0] = TailMask16(srcC - srcC32 - F * 0);
                srcMask[1] = TailMask16(srcC - srcC32 - F * 1);
                dstMask[0] = TailMask32(srcCh2 - srcC32);
            }
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
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
                                    const float* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    size_t sc = 0;
                                    for (; sc < srcC32; sc += 32)
                                        Float32ToBFloat16<false, false>(ps + sc, dst + sc, srcMask, dstMask);
                                    if (srcC32 < srcC)
                                        Float32ToBFloat16<false, true>(ps + sc, dst + sc, srcMask, dstMask);
                                    dst += srcCh2;
                                }
                                else
                                {
                                    memset(dst, 0, srcCh2 * 2);
                                    dst += srcCh2;
                                }
                            }
                        }
                        else
                        {
                            memset(dst, 0, p.kernelX * srcCh2 * 2);
                            dst += p.kernelX * srcCh2;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcConv_2xM(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, int zero, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY, dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            if (tails[1])
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_maskz_loadu_ps(tails[1], dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_maskz_loadu_ps(tails[1], dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xb * dD + F);
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs0 = ky * dY + kx * dX, offs6 = offs0 + 6 * dS, end = offs0 + srcC; offs0 < end; offs0 += 2, offs6 += 2)
                        {
                            w01 = _mm512_loadu_ps((float*)weight + 0);
                            w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                            w01 = _mm512_and_ps(w01, m);
                            w11 = _mm512_loadu_ps((float*)weight + F);
                            w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                            w11 = _mm512_and_ps(w11, m);
                            if (M > 0x0)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                                d00 = _mm512_fmadd_ps(s0, w00, d00); d01 = _mm512_fmadd_ps(s0, w10, d01);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                                d00 = _mm512_fmadd_ps(s0, w01, d00); d01 = _mm512_fmadd_ps(s0, w11, d01);
                            }
                            if (M > 0x1)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                                d10 = _mm512_fmadd_ps(s0, w00, d10); d11 = _mm512_fmadd_ps(s0, w10, d11);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                                d10 = _mm512_fmadd_ps(s0, w01, d10); d11 = _mm512_fmadd_ps(s0, w11, d11);
                            }
                            if (M > 0x2)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                                d20 = _mm512_fmadd_ps(s0, w00, d20); d21 = _mm512_fmadd_ps(s0, w10, d21);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                                d20 = _mm512_fmadd_ps(s0, w01, d20); d21 = _mm512_fmadd_ps(s0, w11, d21);
                            }
                            if (M > 0x3)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                                d30 = _mm512_fmadd_ps(s0, w00, d30); d31 = _mm512_fmadd_ps(s0, w10, d31);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                                d30 = _mm512_fmadd_ps(s0, w01, d30); d31 = _mm512_fmadd_ps(s0, w11, d31);
                            }
                            if (M > 0x4)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                                d40 = _mm512_fmadd_ps(s0, w00, d40); d41 = _mm512_fmadd_ps(s0, w10, d41);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                                d40 = _mm512_fmadd_ps(s0, w01, d40); d41 = _mm512_fmadd_ps(s0, w11, d41);
                            }
                            if (M > 0x5)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                                d50 = _mm512_fmadd_ps(s0, w00, d50); d51 = _mm512_fmadd_ps(s0, w10, d51);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                                d50 = _mm512_fmadd_ps(s0, w01, d50); d51 = _mm512_fmadd_ps(s0, w11, d51);
                            }
                            if (M > 0x6)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                                d60 = _mm512_fmadd_ps(s0, w00, d60); d61 = _mm512_fmadd_ps(s0, w10, d61);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                                d60 = _mm512_fmadd_ps(s0, w01, d60); d61 = _mm512_fmadd_ps(s0, w11, d61);
                            }
                            if (M > 0x7)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                                d70 = _mm512_fmadd_ps(s0, w00, d70); d71 = _mm512_fmadd_ps(s0, w10, d71);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                                d70 = _mm512_fmadd_ps(s0, w01, d70); d71 = _mm512_fmadd_ps(s0, w11, d71);
                            }
                            if (M > 0x8)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                                d80 = _mm512_fmadd_ps(s0, w00, d80); d81 = _mm512_fmadd_ps(s0, w10, d81);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                                d80 = _mm512_fmadd_ps(s0, w01, d80); d81 = _mm512_fmadd_ps(s0, w11, d81);
                            }
                            if (M > 0x9)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                                d90 = _mm512_fmadd_ps(s0, w00, d90); d91 = _mm512_fmadd_ps(s0, w10, d91);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                                d90 = _mm512_fmadd_ps(s0, w01, d90); d91 = _mm512_fmadd_ps(s0, w11, d91);
                            }
                            if (M > 0xa)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                                da0 = _mm512_fmadd_ps(s0, w00, da0); da1 = _mm512_fmadd_ps(s0, w10, da1);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                                da0 = _mm512_fmadd_ps(s0, w01, da0); da1 = _mm512_fmadd_ps(s0, w11, da1);
                            }
                            if (M > 0xb)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                                db0 = _mm512_fmadd_ps(s0, w00, db0); db1 = _mm512_fmadd_ps(s0, w10, db1);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                                db0 = _mm512_fmadd_ps(s0, w01, db0); db1 = _mm512_fmadd_ps(s0, w11, db1);
                            }
                            weight += QF;
                        }
                    }
                }
                if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
                if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, tails), dst += dD;
                if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, tails), dst += dD;
                if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, tails), dst += dD;
            }
            else
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tails[0], dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xb * dD + 0);
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs0 = ky * dY + kx * dX, offs6 = offs0 + 6 * dS, end = offs0 + srcC; offs0 < end; offs0 += 2, offs6 += 2)
                        {
                            w01 = _mm512_loadu_ps((float*)weight + 0);
                            w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                            w01 = _mm512_and_ps(w01, m);
                            if (M > 0x0)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                                d00 = _mm512_fmadd_ps(s0, w00, d00);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                                d00 = _mm512_fmadd_ps(s0, w01, d00);
                            }
                            if (M > 0x1)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                                d10 = _mm512_fmadd_ps(s0, w00, d10);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                                d10 = _mm512_fmadd_ps(s0, w01, d10);
                            }
                            if (M > 0x2)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                                d20 = _mm512_fmadd_ps(s0, w00, d20);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                                d20 = _mm512_fmadd_ps(s0, w01, d20);
                            }
                            if (M > 0x3)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                                d30 = _mm512_fmadd_ps(s0, w00, d30);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                                d30 = _mm512_fmadd_ps(s0, w01, d30);
                            }
                            if (M > 0x4)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                                d40 = _mm512_fmadd_ps(s0, w00, d40);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                                d40 = _mm512_fmadd_ps(s0, w01, d40);
                            }
                            if (M > 0x5)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                                d50 = _mm512_fmadd_ps(s0, w00, d50);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                                d50 = _mm512_fmadd_ps(s0, w01, d50);
                            } 
                            if (M > 0x6)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                                d60 = _mm512_fmadd_ps(s0, w00, d60); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                                d60 = _mm512_fmadd_ps(s0, w01, d60); 
                            }
                            if (M > 0x7)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                                d70 = _mm512_fmadd_ps(s0, w00, d70); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                                d70 = _mm512_fmadd_ps(s0, w01, d70); 
                            }
                            if (M > 0x8)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                                d80 = _mm512_fmadd_ps(s0, w00, d80); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                                d80 = _mm512_fmadd_ps(s0, w01, d80); 
                            }
                            if (M > 0x9)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                                d90 = _mm512_fmadd_ps(s0, w00, d90); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                                d90 = _mm512_fmadd_ps(s0, w01, d90); 
                            }
                            if (M > 0xa)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                                da0 = _mm512_fmadd_ps(s0, w00, da0); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                                da0 = _mm512_fmadd_ps(s0, w01, da0); 
                            }
                            if (M > 0xb)
                            {
                                s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                                db0 = _mm512_fmadd_ps(s0, w00, db0); 
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                                db0 = _mm512_fmadd_ps(s0, w01, db0); 
                            }
                            weight += QF;
                        }
                    }
                }
                if (M > 0x0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 0x8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
                if (M > 0x9) Save1<term, type>(dst, d90, bias, params, tails), dst += dD;
                if (M > 0xa) Save1<term, type>(dst, da0, bias, params, tails), dst += dD;
                if (M > 0xb) Save1<term, type>(dst, db0, bias, params, tails), dst += dD;
            }
        }

        typedef void(*ConvolutionBf16NhwcConv_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, 
            int zero, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2]);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcConv_2xM_Ptr GetConvolutionBf16NhwcConv_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionBf16NhwcConv_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionBf16NhwcConv_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionBf16NhwcConv_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionBf16NhwcConv_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionBf16NhwcConv_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionBf16NhwcConv_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionBf16NhwcConv_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionBf16NhwcConv_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionBf16NhwcConv_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionBf16NhwcConv_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionBf16NhwcConv_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionBf16NhwcConv_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 12, dstWn = AlignLoAny(p.dstW, n), m = p.dstW - dstWn;
            size_t dW = p.kernelY * p.kernelX * AlignHi(srcC, 2) * DF, dD = p.dstW * p.dstC;
            size_t dS = p.strideY * (p.srcW + p.padX + p.padW) * srcC;
            ConvolutionBf16NhwcConv_2xM_Ptr convolution_2xN = GetConvolutionBf16NhwcConv_2xM<term, type>(n);
            ConvolutionBf16NhwcConv_2xM_Ptr convolution_2xM = GetConvolutionBf16NhwcConv_2xM<term, type>(m);

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(dC), TailMask16(dC - F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                for (size_t dy = 0; dy < dstH; dy++)
                {
                    float* d = dst + dy * dD;
                    const uint16_t* s = src + dy * dS;
                    size_t dx = 0;
                    for (; dx < dstWn; dx += n, d += n * p.dstC, s += n * p.strideX * srcC)
                        convolution_2xN(s, p, srcC, zero, weight, _bias, _params, d, tails);
                    for (; dx < p.dstW; dx += m, d += m * p.dstC, s += m * p.strideX * srcC)
                        convolution_2xM(s, p, srcC, zero, weight, _bias, _params, d, tails);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcGemm_2xM(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, int zero, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dD = p.dstC;
            const uint16_t* src1 = src0 + 1 * srcC;
            const uint16_t* src2 = src0 + 2 * srcC;
            const uint16_t* src3 = src0 + 3 * srcC;
            const uint16_t* src4 = src0 + 4 * srcC;
            const uint16_t* src5 = src0 + 5 * srcC;
            if (tails[1])
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_maskz_loadu_ps(tails[1], dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_maskz_loadu_ps(tails[1], dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_maskz_loadu_ps(tails[1], dst + 0xb * dD + F);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight + 0);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    w11 = _mm512_loadu_ps((float*)weight + F);
                    w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                    w11 = _mm512_and_ps(w11, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00); d01 = _mm512_fmadd_ps(s0, w10, d01);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00); d01 = _mm512_fmadd_ps(s0, w11, d01);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10); d11 = _mm512_fmadd_ps(s0, w10, d11);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10); d11 = _mm512_fmadd_ps(s0, w11, d11);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20); d21 = _mm512_fmadd_ps(s0, w10, d21);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20); d21 = _mm512_fmadd_ps(s0, w11, d21);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30); d31 = _mm512_fmadd_ps(s0, w10, d31);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30); d31 = _mm512_fmadd_ps(s0, w11, d31);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40); d41 = _mm512_fmadd_ps(s0, w10, d41);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40); d41 = _mm512_fmadd_ps(s0, w11, d41);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50); d51 = _mm512_fmadd_ps(s0, w10, d51);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50); d51 = _mm512_fmadd_ps(s0, w11, d51);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60); d61 = _mm512_fmadd_ps(s0, w10, d61);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60); d61 = _mm512_fmadd_ps(s0, w11, d61);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70); d71 = _mm512_fmadd_ps(s0, w10, d71);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70); d71 = _mm512_fmadd_ps(s0, w11, d71);
                    }
                    if (M > 0x8)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                        d80 = _mm512_fmadd_ps(s0, w00, d80); d81 = _mm512_fmadd_ps(s0, w10, d81);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                        d80 = _mm512_fmadd_ps(s0, w01, d80); d81 = _mm512_fmadd_ps(s0, w11, d81);
                    }
                    if (M > 0x9)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                        d90 = _mm512_fmadd_ps(s0, w00, d90); d91 = _mm512_fmadd_ps(s0, w10, d91);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                        d90 = _mm512_fmadd_ps(s0, w01, d90); d91 = _mm512_fmadd_ps(s0, w11, d91);
                    }
                    if (M > 0xa)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                        da0 = _mm512_fmadd_ps(s0, w00, da0); da1 = _mm512_fmadd_ps(s0, w10, da1);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                        da0 = _mm512_fmadd_ps(s0, w01, da0); da1 = _mm512_fmadd_ps(s0, w11, da1);
                    }
                    if (M > 0xb)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                        db0 = _mm512_fmadd_ps(s0, w00, db0); db1 = _mm512_fmadd_ps(s0, w10, db1);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                        db0 = _mm512_fmadd_ps(s0, w01, db0); db1 = _mm512_fmadd_ps(s0, w11, db1);
                    }
                    weight += QF;
                }
                if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
                if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, tails), dst += dD;
                if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, tails), dst += dD;
                if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, tails), dst += dD;
            }
            else
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tails[0], dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tails[0], dst + 0xb * dD + 0);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight + 0);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70);
                    }
                    if (M > 0x8)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                        d80 = _mm512_fmadd_ps(s0, w00, d80);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                        d80 = _mm512_fmadd_ps(s0, w01, d80);
                    }
                    if (M > 0x9)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                        d90 = _mm512_fmadd_ps(s0, w00, d90);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                        d90 = _mm512_fmadd_ps(s0, w01, d90);
                    }
                    if (M > 0xa)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                        da0 = _mm512_fmadd_ps(s0, w00, da0);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                        da0 = _mm512_fmadd_ps(s0, w01, da0);
                    }
                    if (M > 0xb)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                        db0 = _mm512_fmadd_ps(s0, w00, db0);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                        db0 = _mm512_fmadd_ps(s0, w01, db0);
                    }
                    weight += QF;
                }
                if (M > 0x0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 0x8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
                if (M > 0x9) Save1<term, type>(dst, d90, bias, params, tails), dst += dD;
                if (M > 0xa) Save1<term, type>(dst, da0, bias, params, tails), dst += dD;
                if (M > 0xb) Save1<term, type>(dst, db0, bias, params, tails), dst += dD;
            }
        }

        typedef void(*ConvolutionBf16NhwcGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, int zero, 
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2]);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcGemm_2xM_Ptr GetConvolutionBf16NhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionBf16NhwcGemm_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionBf16NhwcGemm_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionBf16NhwcGemm_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionBf16NhwcGemm_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n1 = dstH * p.dstW, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 2) * DF;
            ConvolutionBf16NhwcGemm_2xM_Ptr convolution_2xN = GetConvolutionBf16NhwcGemm_2xM<term, type>(n);
            ConvolutionBf16NhwcGemm_2xM_Ptr convolution_2xM = GetConvolutionBf16NhwcGemm_2xM<term, type>(m);

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(dC), TailMask16(dC - F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                    convolution_2xN(s, p, srcC, zero, weight, _bias, _params, d, tails);
                for (; i < n1; i += m, s += m * srcC, d += m * p.dstC)
                    convolution_2xM(s, p, srcC, zero, weight, _bias, _params, d, tails);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcGemm_3xM(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, int zero, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[3])
        {
            __m512 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, 
                d40, d41, d42, d50, d51, d52, d60, d61, d62, d70, d71, d72,
                s0, s1, w00, w01, w10, w11, w20, w21, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dD = p.dstC;
            const uint16_t* src1 = src0 + 1 * srcC;
            const uint16_t* src2 = src0 + 2 * srcC;
            const uint16_t* src3 = src0 + 3 * srcC;
            const uint16_t* src4 = src0 + 4 * srcC;
            const uint16_t* src5 = src0 + 5 * srcC;
            const uint16_t* src6 = src0 + 6 * srcC;
            const uint16_t* src7 = src0 + 7 * srcC;
            if (tails[2])
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps(), d02 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps(), d12 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps(), d22 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps(), d32 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps(), d42 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps(), d52 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps(), d62 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps(), d72 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0 * F), d01 = _mm512_loadu_ps(dst + 0x0 * dD + 1 * F), d02 = _mm512_maskz_loadu_ps(tails[2], dst + 0x0 * dD + 2 * F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0 * F), d11 = _mm512_loadu_ps(dst + 0x1 * dD + 1 * F), d12 = _mm512_maskz_loadu_ps(tails[2], dst + 0x1 * dD + 2 * F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0 * F), d21 = _mm512_loadu_ps(dst + 0x2 * dD + 1 * F), d22 = _mm512_maskz_loadu_ps(tails[2], dst + 0x2 * dD + 2 * F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0 * F), d31 = _mm512_loadu_ps(dst + 0x3 * dD + 1 * F), d32 = _mm512_maskz_loadu_ps(tails[2], dst + 0x3 * dD + 2 * F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0 * F), d41 = _mm512_loadu_ps(dst + 0x4 * dD + 1 * F), d42 = _mm512_maskz_loadu_ps(tails[2], dst + 0x4 * dD + 2 * F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0 * F), d51 = _mm512_loadu_ps(dst + 0x5 * dD + 1 * F), d52 = _mm512_maskz_loadu_ps(tails[2], dst + 0x5 * dD + 2 * F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0 * F), d61 = _mm512_loadu_ps(dst + 0x6 * dD + 1 * F), d62 = _mm512_maskz_loadu_ps(tails[2], dst + 0x6 * dD + 2 * F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0 * F), d71 = _mm512_loadu_ps(dst + 0x7 * dD + 1 * F), d72 = _mm512_maskz_loadu_ps(tails[2], dst + 0x7 * dD + 2 * F);
                }
                for (size_t offs0 = 0; offs0 < srcC; offs0 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight + 0 * F);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(w01)));

                    w11 = _mm512_loadu_ps((float*)weight + 1 * F);
                    w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                    w11 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(w11)));

                    w21 = _mm512_loadu_ps((float*)weight + 2 * F);
                    w20 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w21), Base::Bf16::SHIFT));
                    w21 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(w21)));

                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00); d01 = _mm512_fmadd_ps(s0, w10, d01); d02 = _mm512_fmadd_ps(s0, w20, d02);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src0 + offs0)))));
                        d00 = _mm512_fmadd_ps(s1, w01, d00); d01 = _mm512_fmadd_ps(s1, w11, d01); d02 = _mm512_fmadd_ps(s1, w21, d02);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10); d11 = _mm512_fmadd_ps(s0, w10, d11); d12 = _mm512_fmadd_ps(s0, w20, d12);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src1 + offs0)))));
                        d10 = _mm512_fmadd_ps(s1, w01, d10); d11 = _mm512_fmadd_ps(s1, w11, d11); d12 = _mm512_fmadd_ps(s1, w21, d12);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20); d21 = _mm512_fmadd_ps(s0, w10, d21); d22 = _mm512_fmadd_ps(s0, w20, d22);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src2 + offs0)))));
                        d20 = _mm512_fmadd_ps(s1, w01, d20); d21 = _mm512_fmadd_ps(s1, w11, d21); d22 = _mm512_fmadd_ps(s1, w21, d22);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30); d31 = _mm512_fmadd_ps(s0, w10, d31); d32 = _mm512_fmadd_ps(s0, w20, d32);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src3 + offs0)))));
                        d30 = _mm512_fmadd_ps(s1, w01, d30); d31 = _mm512_fmadd_ps(s1, w11, d31); d32 = _mm512_fmadd_ps(s1, w21, d32);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40); d41 = _mm512_fmadd_ps(s0, w10, d41); d42 = _mm512_fmadd_ps(s0, w20, d42);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src4 + offs0)))));
                        d40 = _mm512_fmadd_ps(s1, w01, d40); d41 = _mm512_fmadd_ps(s1, w11, d41); d42 = _mm512_fmadd_ps(s1, w21, d42);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50); d51 = _mm512_fmadd_ps(s0, w10, d51); d52 = _mm512_fmadd_ps(s0, w20, d52);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src5 + offs0)))));
                        d50 = _mm512_fmadd_ps(s1, w01, d50); d51 = _mm512_fmadd_ps(s1, w11, d51); d52 = _mm512_fmadd_ps(s1, w21, d52);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src6[offs0]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60); d61 = _mm512_fmadd_ps(s0, w10, d61); d62 = _mm512_fmadd_ps(s0, w20, d62);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src6 + offs0)))));
                        d60 = _mm512_fmadd_ps(s1, w01, d60); d61 = _mm512_fmadd_ps(s1, w11, d61); d62 = _mm512_fmadd_ps(s1, w21, d62);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src7[offs0]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70); d71 = _mm512_fmadd_ps(s0, w10, d71); d72 = _mm512_fmadd_ps(s0, w20, d72);
                        s1 = _mm512_castsi512_ps(_mm512_maskz_mov_epi16(0xAAAAAAAA, _mm512_castps_si512(_mm512_set1_ps(*(float*)(src7 + offs0)))));
                        d70 = _mm512_fmadd_ps(s1, w01, d70); d71 = _mm512_fmadd_ps(s1, w11, d71); d72 = _mm512_fmadd_ps(s1, w21, d72);
                    }
                    weight += 6 * F;
                }
                if (M > 0x0) Save3<term, type>(dst, d00, d01, d02, bias, params, tails), dst += dD;
                if (M > 0x1) Save3<term, type>(dst, d10, d11, d12, bias, params, tails), dst += dD;
                if (M > 0x2) Save3<term, type>(dst, d20, d21, d22, bias, params, tails), dst += dD;
                if (M > 0x3) Save3<term, type>(dst, d30, d31, d32, bias, params, tails), dst += dD;
                if (M > 0x4) Save3<term, type>(dst, d40, d41, d42, bias, params, tails), dst += dD;
                if (M > 0x5) Save3<term, type>(dst, d50, d51, d52, bias, params, tails), dst += dD;
                if (M > 0x6) Save3<term, type>(dst, d60, d61, d62, bias, params, tails), dst += dD;
                if (M > 0x7) Save3<term, type>(dst, d70, d71, d72, bias, params, tails), dst += dD;
            }
            else if (tails[1])
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0 * F), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0x0 * dD + 1 * F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0 * F), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 0x1 * dD + 1 * F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0 * F), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 0x2 * dD + 1 * F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0 * F), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 0x3 * dD + 1 * F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0 * F), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 0x4 * dD + 1 * F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0 * F), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 0x5 * dD + 1 * F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0 * F), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 0x6 * dD + 1 * F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0 * F), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 0x7 * dD + 1 * F);
                }
                for (size_t offs0 = 0; offs0 < srcC; offs0 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight + 0 * F);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    w11 = _mm512_loadu_ps((float*)weight + 1 * F);
                    w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                    w11 = _mm512_and_ps(w11, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00); d01 = _mm512_fmadd_ps(s0, w10, d01);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00); d01 = _mm512_fmadd_ps(s0, w11, d01);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10); d11 = _mm512_fmadd_ps(s0, w10, d11);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10); d11 = _mm512_fmadd_ps(s0, w11, d11);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20); d21 = _mm512_fmadd_ps(s0, w10, d21);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20); d21 = _mm512_fmadd_ps(s0, w11, d21);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30); d31 = _mm512_fmadd_ps(s0, w10, d31);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30); d31 = _mm512_fmadd_ps(s0, w11, d31);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40); d41 = _mm512_fmadd_ps(s0, w10, d41);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40); d41 = _mm512_fmadd_ps(s0, w11, d41);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50); d51 = _mm512_fmadd_ps(s0, w10, d51);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50); d51 = _mm512_fmadd_ps(s0, w11, d51); 
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src6[offs0]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60); d61 = _mm512_fmadd_ps(s0, w10, d61);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src6 + offs0)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60); d61 = _mm512_fmadd_ps(s0, w11, d61);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src7[offs0]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70); d71 = _mm512_fmadd_ps(s0, w10, d71); 
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src7 + offs0)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70); d71 = _mm512_fmadd_ps(s0, w11, d71);
                    }
                    weight += 6 * F;
                }
                if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
            }
            else
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 0x7 * dD + 0);
                }
                for (size_t offs0 = 0; offs0 < srcC; offs0 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight + 0 * F);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src6[offs0]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src6 + offs0)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src7[offs0]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src7 + offs0)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70);
                    }
                    weight += 6 * F;
                }
                if (M > 0x0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
            }
        }

        typedef void(*ConvolutionBf16NhwcGemm_3xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, int zero,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[3]);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcGemm_3xM_Ptr GetConvolutionBf16NhwcGemm_3xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x1>;
            case 0x2: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x2>;
            case 0x3: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x3>;
            case 0x4: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x4>;
            case 0x5: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x5>;
            case 0x6: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x6>;
            case 0x7: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x7>;
            case 0x8: return ConvolutionBf16NhwcGemm_3xM<term, type, 0x8>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_3(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n1 = dstH * p.dstW, n = 8;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 2) * 3 * F;
            ConvolutionBf16NhwcGemm_3xM_Ptr convolution_3xN = GetConvolutionBf16NhwcGemm_3xM<term, type>(n);
            ConvolutionBf16NhwcGemm_3xM_Ptr convolution_3xM = GetConvolutionBf16NhwcGemm_3xM<term, type>(m);

            __m512 _params[3], _bias[3];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += F * 3)
            {
                size_t dC = Simd::Min(F * 3, dstC - dc);
                __mmask16 tails[3] = { TailMask16(dC - 0 * F), TailMask16(dC - 1 * F), TailMask16(dC - 2 * F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                _bias[2] = _mm512_loadu_ps(bias + dc + 2 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                    _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                    _params[2] = _mm512_loadu_ps(params + dc + 2 * F);
                }
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                    convolution_3xN(s, p, srcC, zero, weight, _bias, _params, d, tails);
                for (; i < n1; i += m, s += m * srcC, d += m * p.dstC)
                    convolution_3xM(s, p, srcC, zero, weight, _bias, _params, d, tails);
                weight += dW;
                dst += 3 * F;
            }
        }

        static SIMD_INLINE bool UseF3(const ConvParam32f& p)
        {
            if (p.Is1x1())
            {
                if (p.dstC <= 2 * F || (p.dstC > 3 * F && p.dstC <= 4 * F))
                    return false;
                return true;
            }
            else
                return false;
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convolution* convolutions)
        {
            if (p.Is1x1() || a.mode)
            {
                if (UseF3(p))
                {
                    convolutions[TermLast] = ConvolutionBf16NhwcGemm_3<TermLast, type>;
                    convolutions[TermInterim] = ConvolutionBf16NhwcGemm_3<TermInterim, type>;
                }
                else
                {
                    convolutions[TermLast] = ConvolutionBf16NhwcGemm_2<TermLast, type>;
                    convolutions[TermInterim] = ConvolutionBf16NhwcGemm_2<TermInterim, type>;
                }
            }
            else
            {
                convolutions[TermLast] = ConvolutionBf16NhwcConv_2<TermLast, type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcConv_2<TermInterim, type>;
            }
        }

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f & p)
            : Avx2::SynetConvolution32fBf16Nhwc(p)
        {
            size_t microD = F * (UseF3(p) ? 3 : 2);
            size_t microHW = UseF3(p) ? 8 : 12;
            SetAlgParam(microD, microHW, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_alg.mode)
                _convert = ConvolutionBf16NhwcConvertGemm;
            else
                _convert = ConvolutionBf16NhwcConvertConv;
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
            default: assert(0);
            }
        }
    }
#endif
}
