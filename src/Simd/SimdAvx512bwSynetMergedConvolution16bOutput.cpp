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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution16b::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, const __m512* bias, const __m512* params, float* buf, uint8_t* dst)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dS = a.maC * p.strideX, dB = p.dstC, dD = p.dstC * a.elem[1];
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                __mmask16 tail = TailMask16(dstC - F);
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
                    if (M > 0x0) d00 = _mm512_loadu_ps(buf + 0x0 * dB + 0), d01 = _mm512_maskz_loadu_ps(tail, buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(buf + 0x1 * dB + 0), d11 = _mm512_maskz_loadu_ps(tail, buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(buf + 0x2 * dB + 0), d21 = _mm512_maskz_loadu_ps(tail, buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(buf + 0x3 * dB + 0), d31 = _mm512_maskz_loadu_ps(tail, buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(buf + 0x4 * dB + 0), d41 = _mm512_maskz_loadu_ps(tail, buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(buf + 0x5 * dB + 0), d51 = _mm512_maskz_loadu_ps(tail, buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(buf + 0x6 * dB + 0), d61 = _mm512_maskz_loadu_ps(tail, buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(buf + 0x7 * dB + 0), d71 = _mm512_maskz_loadu_ps(tail, buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(buf + 0x8 * dB + 0), d81 = _mm512_maskz_loadu_ps(tail, buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(buf + 0x9 * dB + 0), d91 = _mm512_maskz_loadu_ps(tail, buf + 0x9 * dB + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(buf + 0xa * dB + 0), da1 = _mm512_maskz_loadu_ps(tail, buf + 0xa * dB + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(buf + 0xb * dB + 0), db1 = _mm512_maskz_loadu_ps(tail, buf + 0xb * dB + F);
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
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
                            d00 = Fmadd<true>(s0, w00, d00); d01 = Fmadd<true>(s0, w10, d01);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                            d00 = Fmadd<true>(s0, w01, d00); d01 = Fmadd<true>(s0, w11, d01);
                        }
                        if (M > 0x1)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                            d10 = Fmadd<true>(s0, w00, d10); d11 = Fmadd<true>(s0, w10, d11);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                            d10 = Fmadd<true>(s0, w01, d10); d11 = Fmadd<true>(s0, w11, d11);
                        }
                        if (M > 0x2)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                            d20 = Fmadd<true>(s0, w00, d20); d21 = Fmadd<true>(s0, w10, d21);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                            d20 = Fmadd<true>(s0, w01, d20); d21 = Fmadd<true>(s0, w11, d21);
                        }
                        if (M > 0x3)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                            d30 = Fmadd<true>(s0, w00, d30); d31 = Fmadd<true>(s0, w10, d31);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                            d30 = Fmadd<true>(s0, w01, d30); d31 = Fmadd<true>(s0, w11, d31);
                        }
                        if (M > 0x4)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                            d40 = Fmadd<true>(s0, w00, d40); d41 = Fmadd<true>(s0, w10, d41);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                            d40 = Fmadd<true>(s0, w01, d40); d41 = Fmadd<true>(s0, w11, d41);
                        }
                        if (M > 0x5)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                            d50 = Fmadd<true>(s0, w00, d50); d51 = Fmadd<true>(s0, w10, d51);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                            d50 = Fmadd<true>(s0, w01, d50); d51 = Fmadd<true>(s0, w11, d51);
                        }
                        if (M > 0x6)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                            d60 = Fmadd<true>(s0, w00, d60); d61 = Fmadd<true>(s0, w10, d61);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                            d60 = Fmadd<true>(s0, w01, d60); d61 = Fmadd<true>(s0, w11, d61);
                        }
                        if (M > 0x7)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                            d70 = Fmadd<true>(s0, w00, d70); d71 = Fmadd<true>(s0, w10, d71);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                            d70 = Fmadd<true>(s0, w01, d70); d71 = Fmadd<true>(s0, w11, d71);
                        }
                        if (M > 0x8)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                            d80 = Fmadd<true>(s0, w00, d80); d81 = Fmadd<true>(s0, w10, d81);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                            d80 = Fmadd<true>(s0, w01, d80); d81 = Fmadd<true>(s0, w11, d81);
                        }
                        if (M > 0x9)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                            d90 = Fmadd<true>(s0, w00, d90); d91 = Fmadd<true>(s0, w10, d91);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                            d90 = Fmadd<true>(s0, w01, d90); d91 = Fmadd<true>(s0, w11, d91);
                        }
                        if (M > 0xa)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                            da0 = Fmadd<true>(s0, w00, da0); da1 = Fmadd<true>(s0, w10, da1);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                            da0 = Fmadd<true>(s0, w01, da0); da1 = Fmadd<true>(s0, w11, da1);
                        }
                        if (M > 0xb)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                            db0 = Fmadd<true>(s0, w00, db0); db1 = Fmadd<true>(s0, w10, db1);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                            db0 = Fmadd<true>(s0, w01, db0); db1 = Fmadd<true>(s0, w11, db1);
                        }
                        weight += QF;
                    }
                }
                else
                {
                    for (size_t offs0 = 0, offs6 = 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
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
                            d00 = Fmadd<false>(s0, w00, d00); d01 = Fmadd<false>(s0, w10, d01);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                            d00 = Fmadd<false>(s0, w01, d00); d01 = Fmadd<false>(s0, w11, d01);
                        }
                        if (M > 0x1)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                            d10 = Fmadd<false>(s0, w00, d10); d11 = Fmadd<false>(s0, w10, d11);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                            d10 = Fmadd<false>(s0, w01, d10); d11 = Fmadd<false>(s0, w11, d11);
                        }
                        if (M > 0x2)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                            d20 = Fmadd<false>(s0, w00, d20); d21 = Fmadd<false>(s0, w10, d21);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                            d20 = Fmadd<false>(s0, w01, d20); d21 = Fmadd<false>(s0, w11, d21);
                        }
                        if (M > 0x3)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                            d30 = Fmadd<false>(s0, w00, d30); d31 = Fmadd<false>(s0, w10, d31);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                            d30 = Fmadd<false>(s0, w01, d30); d31 = Fmadd<false>(s0, w11, d31);
                        }
                        if (M > 0x4)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                            d40 = Fmadd<false>(s0, w00, d40); d41 = Fmadd<false>(s0, w10, d41);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                            d40 = Fmadd<false>(s0, w01, d40); d41 = Fmadd<false>(s0, w11, d41);
                        }
                        if (M > 0x5)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                            d50 = Fmadd<false>(s0, w00, d50); d51 = Fmadd<false>(s0, w10, d51);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                            d50 = Fmadd<false>(s0, w01, d50); d51 = Fmadd<false>(s0, w11, d51);
                        }
                        if (M > 0x6)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                            d60 = Fmadd<false>(s0, w00, d60); d61 = Fmadd<false>(s0, w10, d61);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                            d60 = Fmadd<false>(s0, w01, d60); d61 = Fmadd<false>(s0, w11, d61);
                        }
                        if (M > 0x7)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                            d70 = Fmadd<false>(s0, w00, d70); d71 = Fmadd<false>(s0, w10, d71);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                            d70 = Fmadd<false>(s0, w01, d70); d71 = Fmadd<false>(s0, w11, d71);
                        }
                        if (M > 0x8)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                            d80 = Fmadd<false>(s0, w00, d80); d81 = Fmadd<false>(s0, w10, d81);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                            d80 = Fmadd<false>(s0, w01, d80); d81 = Fmadd<false>(s0, w11, d81);
                        }
                        if (M > 0x9)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                            d90 = Fmadd<false>(s0, w00, d90); d91 = Fmadd<false>(s0, w10, d91);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                            d90 = Fmadd<false>(s0, w01, d90); d91 = Fmadd<false>(s0, w11, d91);
                        }
                        if (M > 0xa)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                            da0 = Fmadd<false>(s0, w00, da0); da1 = Fmadd<false>(s0, w10, da1);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                            da0 = Fmadd<false>(s0, w01, da0); da1 = Fmadd<false>(s0, w11, da1);
                        }
                        if (M > 0xb)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                            db0 = Fmadd<false>(s0, w00, db0); db1 = Fmadd<false>(s0, w10, db1);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                            db0 = Fmadd<false>(s0, w01, db0); db1 = Fmadd<false>(s0, w11, db1);
                        }
                        weight += QF;
                    }
                }
                if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, bias, params, tail), buf += dB, dst += dD;
                if (M > 0xa) Save2<term, type>(dst, buf, da0, da1, bias, params, tail), buf += dB, dst += dD;
                if (M > 0xb) Save2<term, type>(dst, buf, db0, db1, bias, params, tail), buf += dB, dst += dD;
            }
            else
            {
                __mmask16 tail = TailMask16(dstC);
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
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tail, buf + 0x0 * dB + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tail, buf + 0x1 * dB + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tail, buf + 0x2 * dB + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tail, buf + 0x3 * dB + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tail, buf + 0x4 * dB + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tail, buf + 0x5 * dB + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tail, buf + 0x6 * dB + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tail, buf + 0x7 * dB + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tail, buf + 0x8 * dB + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tail, buf + 0x9 * dB + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tail, buf + 0xa * dB + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tail, buf + 0xb * dB + 0);
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                    {
                        w01 = _mm512_loadu_ps((float*)weight + 0);
                        w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                        w01 = _mm512_and_ps(w01, m);
                        if (M > 0x0)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                            d00 = Fmadd<true>(s0, w00, d00);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                            d00 = Fmadd<true>(s0, w01, d00);
                        }
                        if (M > 0x1)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                            d10 = Fmadd<true>(s0, w00, d10);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                            d10 = Fmadd<true>(s0, w01, d10);
                        }
                        if (M > 0x2)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                            d20 = Fmadd<true>(s0, w00, d20);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                            d20 = Fmadd<true>(s0, w01, d20);
                        }
                        if (M > 0x3)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                            d30 = Fmadd<true>(s0, w00, d30);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                            d30 = Fmadd<true>(s0, w01, d30);
                        }
                        if (M > 0x4)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                            d40 = Fmadd<true>(s0, w00, d40);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                            d40 = Fmadd<true>(s0, w01, d40);
                        }
                        if (M > 0x5)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                            d50 = Fmadd<true>(s0, w00, d50);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                            d50 = Fmadd<true>(s0, w01, d50);
                        }
                        if (M > 0x6)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                            d60 = Fmadd<true>(s0, w00, d60);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                            d60 = Fmadd<true>(s0, w01, d60);
                        }
                        if (M > 0x7)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                            d70 = Fmadd<true>(s0, w00, d70);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                            d70 = Fmadd<true>(s0, w01, d70);
                        }
                        if (M > 0x8)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                            d80 = Fmadd<true>(s0, w00, d80);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                            d80 = Fmadd<true>(s0, w01, d80);
                        }
                        if (M > 0x9)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                            d90 = Fmadd<true>(s0, w00, d90);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                            d90 = Fmadd<true>(s0, w01, d90);
                        }
                        if (M > 0xa)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                            da0 = Fmadd<true>(s0, w00, da0);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                            da0 = Fmadd<true>(s0, w01, da0);
                        }
                        if (M > 0xb)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                            db0 = Fmadd<true>(s0, w00, db0);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                            db0 = Fmadd<true>(s0, w01, db0);
                        }
                        weight += QF;
                    }
                }
                else
                {
                    for (size_t offs0 = 0, offs6 = 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                    {
                        w01 = _mm512_loadu_ps((float*)weight + 0);
                        w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                        w01 = _mm512_and_ps(w01, m);
                        if (M > 0x0)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                            d00 = Fmadd<false>(s0, w00, d00);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                            d00 = Fmadd<false>(s0, w01, d00);
                        }
                        if (M > 0x1)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                            d10 = Fmadd<false>(s0, w00, d10);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                            d10 = Fmadd<false>(s0, w01, d10);
                        }
                        if (M > 0x2)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                            d20 = Fmadd<false>(s0, w00, d20);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                            d20 = Fmadd<false>(s0, w01, d20);
                        }
                        if (M > 0x3)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                            d30 = Fmadd<false>(s0, w00, d30);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                            d30 = Fmadd<false>(s0, w01, d30);
                        }
                        if (M > 0x4)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                            d40 = Fmadd<false>(s0, w00, d40);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                            d40 = Fmadd<false>(s0, w01, d40);
                        }
                        if (M > 0x5)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                            d50 = Fmadd<false>(s0, w00, d50);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                            d50 = Fmadd<false>(s0, w01, d50);
                        }
                        if (M > 0x6)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                            d60 = Fmadd<false>(s0, w00, d60);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                            d60 = Fmadd<false>(s0, w01, d60);
                        }
                        if (M > 0x7)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                            d70 = Fmadd<false>(s0, w00, d70);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                            d70 = Fmadd<false>(s0, w01, d70);
                        }
                        if (M > 0x8)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                            d80 = Fmadd<false>(s0, w00, d80);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                            d80 = Fmadd<false>(s0, w01, d80);
                        }
                        if (M > 0x9)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                            d90 = Fmadd<false>(s0, w00, d90);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                            d90 = Fmadd<false>(s0, w01, d90);
                        }
                        if (M > 0xa)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                            da0 = Fmadd<false>(s0, w00, da0);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                            da0 = Fmadd<false>(s0, w01, da0);
                        }
                        if (M > 0xb)
                        {
                            s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                            db0 = Fmadd<false>(s0, w00, db0);
                            s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                            db0 = Fmadd<false>(s0, w01, db0);
                        }
                        weight += QF;
                    }
                }
                if (M > 0x0) Save1<term, type>(dst, buf, d00, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x1) Save1<term, type>(dst, buf, d10, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x2) Save1<term, type>(dst, buf, d20, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x3) Save1<term, type>(dst, buf, d30, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x4) Save1<term, type>(dst, buf, d40, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x5) Save1<term, type>(dst, buf, d50, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x6) Save1<term, type>(dst, buf, d60, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x7) Save1<term, type>(dst, buf, d70, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x8) Save1<term, type>(dst, buf, d80, bias, params, tail), buf += dB, dst += dD;
                if (M > 0x9) Save1<term, type>(dst, buf, d90, bias, params, tail), buf += dB, dst += dD;
                if (M > 0xa) Save1<term, type>(dst, buf, da0, bias, params, tail), buf += dB, dst += dD;
                if (M > 0xb) Save1<term, type>(dst, buf, db0, bias, params, tail), buf += dB, dst += dD;
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return OutputConvolution1x1_2xM<term, type, 0x1>;
            case 0x2: return OutputConvolution1x1_2xM<term, type, 0x2>;
            case 0x3: return OutputConvolution1x1_2xM<term, type, 0x3>;
            case 0x4: return OutputConvolution1x1_2xM<term, type, 0x4>;
            case 0x5: return OutputConvolution1x1_2xM<term, type, 0x5>;
            case 0x6: return OutputConvolution1x1_2xM<term, type, 0x6>;
            case 0x7: return OutputConvolution1x1_2xM<term, type, 0x7>;
            case 0x8: return OutputConvolution1x1_2xM<term, type, 0x8>;
            case 0x9: return OutputConvolution1x1_2xM<term, type, 0x9>;
            case 0xa: return OutputConvolution1x1_2xM<term, type, 0xa>;
            case 0xb: return OutputConvolution1x1_2xM<term, type, 0xb>;
            case 0xc: return OutputConvolution1x1_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src, const ConvParam& p, const AlgParam& a, 
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                const uint16_t* s = src;
                float * b = buf + dc + yBeg * p.dstW * p.dstC;
                uint8_t * d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += p.dstC * n, d += p.dstC * a.elem[1] * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, zero, weight, _bias, _params, b, d);
                for (; i < n1; i += m, s += a.maC * m, b += p.dstC * m, d += p.dstC * a.elem[1] * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, zero, weight, _bias, _params, b, d);
                weight += DivHi(maC, 2) * QF;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            if (p.dstT == SimdTensorData16b)
                output[0] = OutputConvolution1x1_2<Term16bLast16b, type>;
            else
                output[0] = OutputConvolution1x1_2<Term16bLast32f, type>;
            output[1] = OutputConvolution1x1_2<Term16bInterim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, output); break;
            case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, output); break;
            case SimdConvolutionActivationMish: SetOutput<SimdConvolutionActivationMish>(p, output); break;
            case SimdConvolutionActivationHardSigmoid: SetOutput<SimdConvolutionActivationHardSigmoid>(p, output); break;
            case SimdConvolutionActivationSwish: SetOutput<SimdConvolutionActivationSwish>(p, output); break;
            case SimdConvolutionActivationGelu: SetOutput<SimdConvolutionActivationGelu>(p, output); break;
            }
        }
    }
#endif
}
