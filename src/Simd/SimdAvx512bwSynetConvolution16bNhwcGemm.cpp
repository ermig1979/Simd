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

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        typedef Base::SynetConvolution16bNhwcGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (srcC32 < p.srcC)
            {
                srcMask[0] = TailMask16(p.srcC - srcC32 - F * 0);
                srcMask[1] = TailMask16(p.srcC - srcC32 - F * 1);
                dstMask[0] = TailMask32(p.srcC - srcC32);
            }
            size_t gap = a.bufK - a.K;
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
                                        Avx512bw::Float32ToBFloat16<false, false>(ps + sc, row + sc, srcMask, dstMask);
                                    if (srcC32 < p.srcC)
                                        Avx512bw::Float32ToBFloat16<false, true>(ps + sc, row + sc, srcMask, dstMask);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        static void Reorder16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t gap = a.bufK - a.K;
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
                                    const uint16_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    memcpy(row, ps, p.srcC * 2);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNhwcGemm_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a, 
            size_t srcC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const uint16_t* weight1 = weight0 + a.bufK * F;
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
                    if (M > 0x0) d00 = _mm512_loadu_ps(buf + 0x0 * dB + 0), d01 = _mm512_maskz_loadu_ps(tails[1], buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(buf + 0x1 * dB + 0), d11 = _mm512_maskz_loadu_ps(tails[1], buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(buf + 0x2 * dB + 0), d21 = _mm512_maskz_loadu_ps(tails[1], buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(buf + 0x3 * dB + 0), d31 = _mm512_maskz_loadu_ps(tails[1], buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(buf + 0x4 * dB + 0), d41 = _mm512_maskz_loadu_ps(tails[1], buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(buf + 0x5 * dB + 0), d51 = _mm512_maskz_loadu_ps(tails[1], buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(buf + 0x6 * dB + 0), d61 = _mm512_maskz_loadu_ps(tails[1], buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(buf + 0x7 * dB + 0), d71 = _mm512_maskz_loadu_ps(tails[1], buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(buf + 0x8 * dB + 0), d81 = _mm512_maskz_loadu_ps(tails[1], buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(buf + 0x9 * dB + 0), d91 = _mm512_maskz_loadu_ps(tails[1], buf + 0x9 * dB + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(buf + 0xa * dB + 0), da1 = _mm512_maskz_loadu_ps(tails[1], buf + 0xa * dB + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(buf + 0xb * dB + 0), db1 = _mm512_maskz_loadu_ps(tails[1], buf + 0xb * dB + F);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight0);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    w11 = _mm512_loadu_ps((float*)weight1);
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
                    weight0 += DF;
                    weight1 += DF;
                }
                if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, bias, params, tails), dst += dD, buf += dB;
                if (M > 0xa) Save2<term, type>(dst, buf, da0, da1, bias, params, tails), dst += dD, buf += dB;
                if (M > 0xb) Save2<term, type>(dst, buf, db0, db1, bias, params, tails), dst += dD, buf += dB;
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
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tails[0], buf + 0x0 * dB + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tails[0], buf + 0x1 * dB + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tails[0], buf + 0x2 * dB + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tails[0], buf + 0x3 * dB + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tails[0], buf + 0x4 * dB + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tails[0], buf + 0x5 * dB + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tails[0], buf + 0x6 * dB + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tails[0], buf + 0x7 * dB + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tails[0], buf + 0x8 * dB + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tails[0], buf + 0x9 * dB + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tails[0], buf + 0xa * dB + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tails[0], buf + 0xb * dB + 0);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight0);
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
                    weight0 += DF;
                }
                if (M > 0x0) Save1<term, type>(dst, buf, d00, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x1) Save1<term, type>(dst, buf, d10, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x2) Save1<term, type>(dst, buf, d20, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x3) Save1<term, type>(dst, buf, d30, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x4) Save1<term, type>(dst, buf, d40, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x5) Save1<term, type>(dst, buf, d50, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x6) Save1<term, type>(dst, buf, d60, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x7) Save1<term, type>(dst, buf, d70, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x8) Save1<term, type>(dst, buf, d80, bias, params, tails), dst += dD, buf += dB;
                if (M > 0x9) Save1<term, type>(dst, buf, d90, bias, params, tails), dst += dD, buf += dB;
                if (M > 0xa) Save1<term, type>(dst, buf, da0, bias, params, tails), dst += dD, buf += dB;
                if (M > 0xb) Save1<term, type>(dst, buf, db0, bias, params, tails), dst += dD, buf += dB;
            }
        }

        typedef void(*Convolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, 
            const uint16_t* weight, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, const __mmask16 tails[2]);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNhwcGemm_2xM_Ptr GetConvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Convolution16bNhwcGemm_2xM<term, type, 0x1>;
            case 0x2: return Convolution16bNhwcGemm_2xM<term, type, 0x2>;
            case 0x3: return Convolution16bNhwcGemm_2xM<term, type, 0x3>;
            case 0x4: return Convolution16bNhwcGemm_2xM<term, type, 0x4>;
            case 0x5: return Convolution16bNhwcGemm_2xM<term, type, 0x5>;
            case 0x6: return Convolution16bNhwcGemm_2xM<term, type, 0x6>;
            case 0x7: return Convolution16bNhwcGemm_2xM<term, type, 0x7>;
            case 0x8: return Convolution16bNhwcGemm_2xM<term, type, 0x8>;
            case 0x9: return Convolution16bNhwcGemm_2xM<term, type, 0x9>;
            case 0xa: return Convolution16bNhwcGemm_2xM<term, type, 0xa>;
            case 0xb: return Convolution16bNhwcGemm_2xM<term, type, 0xb>;
            case 0xc: return Convolution16bNhwcGemm_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xN = GetConvolution16bNhwcGemm_2xM<term, type>(n);
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xM = GetConvolution16bNhwcGemm_2xM<term, type>(m);

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
                const uint16_t* s = src;
                float* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_2xN(s, p, a, srcC, zero, weight, _bias, _params, b, d, tails);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_2xM(s, p, a, srcC, zero, weight, _bias, _params, b, d, tails);
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
            : Avx2::SynetConvolution16bNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = Reorder16bNhwcGemm;
            }
            else
                _convert = Convert16bNhwcGemm;
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
