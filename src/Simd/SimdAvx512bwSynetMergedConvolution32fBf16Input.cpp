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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using InputPtr = Base::SynetMergedConvolution32fBf16::InputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, bool nofma> void InputConvolution_2x1(const uint16_t* src0,
            const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const uint16_t* weight,
            const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01, s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 2) * QF, sM = a.bufH[0] - 1;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm512_loadu_ps((float*)weight + 0);
                                w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                                w01 = _mm512_and_ps(w01, m);
                                w11 = _mm512_loadu_ps((float*)weight + F);
                                w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                                w11 = _mm512_and_ps(w11, m);

                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs - 1)), m);
                                d00 = Fmadd<nofma>(s0, w00, d00); d01 = Fmadd<nofma>(s0, w10, d01);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = Fmadd<nofma>(s0, w01, d00); d01 = Fmadd<nofma>(s0, w11, d01);

                                weight += QF;
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                SaveInput2<type>(dst0, dst1, d00, d01, bias, params);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm512_loadu_ps((float*)weight + 0);
                                w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                                w01 = _mm512_and_ps(w01, m);

                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs - 1)), m);
                                d00 = Fmadd<nofma>(s0, w00, d00);
                                s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = Fmadd<nofma>(s0, w01, d00);

                                weight += QF;
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                SaveInput1<type>(dst0, d00, bias, params);
            }
        }

        template<SimdConvolutionActivationType type, int M> void InputConvolution_2xM(const uint16_t* src0, 
            const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC, dWz = DivHi(p.srcC, 2) * QF * p.kernelX, sM = a.bufH[0] - 1;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
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
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 2, offs6 += 2)
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
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 2, offs6 += 2)
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
                        }
                        else
                            weight += dWz;
                    }
                }
                if (M > 0x0) SaveInput2<type>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, bias, params);
                if (M > 0x1) SaveInput2<type>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, bias, params);
                if (M > 0x2) SaveInput2<type>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, bias, params);
                if (M > 0x3) SaveInput2<type>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, bias, params);
                if (M > 0x4) SaveInput2<type>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, bias, params);
                if (M > 0x5) SaveInput2<type>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, bias, params);
                if (M > 0x6) SaveInput2<type>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, bias, params);
                if (M > 0x7) SaveInput2<type>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, bias, params);
                if (M > 0x8) SaveInput2<type>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, bias, params);
                if (M > 0x9) SaveInput2<type>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, bias, params);
                if (M > 0xa) SaveInput2<type>(dst0 + 0xa * F, dst1 + 0xa * F, da0, da1, bias, params);
                if (M > 0xb) SaveInput2<type>(dst0 + 0xb * F, dst1 + 0xb * F, db0, db1, bias, params);
            }
            else
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
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 2, offs6 += 2)
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
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 2, offs6 += 2)
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
                        }
                        else
                            weight += dWz;
                    }
                }
                if (M > 0x0) SaveInput1<type>(dst0 + 0x0 * F, d00, bias, params);
                if (M > 0x1) SaveInput1<type>(dst0 + 0x1 * F, d10, bias, params);
                if (M > 0x2) SaveInput1<type>(dst0 + 0x2 * F, d20, bias, params);
                if (M > 0x3) SaveInput1<type>(dst0 + 0x3 * F, d30, bias, params);
                if (M > 0x4) SaveInput1<type>(dst0 + 0x4 * F, d40, bias, params);
                if (M > 0x5) SaveInput1<type>(dst0 + 0x5 * F, d50, bias, params);
                if (M > 0x6) SaveInput1<type>(dst0 + 0x6 * F, d60, bias, params);
                if (M > 0x7) SaveInput1<type>(dst0 + 0x7 * F, d70, bias, params);
                if (M > 0x8) SaveInput1<type>(dst0 + 0x8 * F, d80, bias, params);
                if (M > 0x9) SaveInput1<type>(dst0 + 0x9 * F, d90, bias, params);
                if (M > 0xa) SaveInput1<type>(dst0 + 0xa * F, da0, bias, params);
                if (M > 0xb) SaveInput1<type>(dst0 + 0xb * F, db0, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx,
            size_t dstC, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return InputConvolution_2xM<type, 0x1>;
            case 0x2: return InputConvolution_2xM<type, 0x2>;
            case 0x3: return InputConvolution_2xM<type, 0x3>;
            case 0x4: return InputConvolution_2xM<type, 0x4>;
            case 0x5: return InputConvolution_2xM<type, 0x5>;
            case 0x6: return InputConvolution_2xM<type, 0x6>;
            case 0x7: return InputConvolution_2xM<type, 0x7>;
            case 0x8: return InputConvolution_2xM<type, 0x8>;
            case 0x9: return InputConvolution_2xM<type, 0x9>;
            case 0xa: return InputConvolution_2xM<type, 0xa>;
            case 0xb: return InputConvolution_2xM<type, 0xb>;
            case 0xc: return InputConvolution_2xM<type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseW = NoseW(p), bodyW = BodyW(p), tailW = p.dstW;
            size_t n = 12, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;
            InputConvolution_2xM_Ptr inputConvolution_2x1 = Base::FmaAvoid(p.compatibility) ? 
                InputConvolution_2x1<type, true> : InputConvolution_2x1<type, false>;
            InputConvolution_2xM_Ptr inputConvolution_2xN = GetInputConvolution_2xM<type>(n);
            InputConvolution_2xM_Ptr inputConvolution_2xM = GetInputConvolution_2xM<type>(m);
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < bodyWn; dx += n, dst0 += F * n, dst1 += F * n)
                        inputConvolution_2xN(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < bodyW; dx += m, dst0 += F * m, dst1 += F * m)
                        inputConvolution_2xM(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 2) * QF;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint16_t* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dstC, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            const uint16_t* src1 = src0 + 1 * p.srcC;
            const uint16_t* src2 = src0 + 2 * p.srcC;
            const uint16_t* src3 = src0 + 3 * p.srcC;
            const uint16_t* src4 = src0 + 4 * p.srcC;
            const uint16_t* src5 = src0 + 5 * p.srcC;
            if (dstC > F)
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
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs6 += 2)
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
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs6 += 2)
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
                if (M > 0x0) SaveInput2<type>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, bias, params);
                if (M > 0x1) SaveInput2<type>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, bias, params);
                if (M > 0x2) SaveInput2<type>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, bias, params);
                if (M > 0x3) SaveInput2<type>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, bias, params);
                if (M > 0x4) SaveInput2<type>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, bias, params);
                if (M > 0x5) SaveInput2<type>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, bias, params);
                if (M > 0x6) SaveInput2<type>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, bias, params);
                if (M > 0x7) SaveInput2<type>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, bias, params);
                if (M > 0x8) SaveInput2<type>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, bias, params);
                if (M > 0x9) SaveInput2<type>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, bias, params);
                if (M > 0xa) SaveInput2<type>(dst0 + 0xa * F, dst1 + 0xa * F, da0, da1, bias, params);
                if (M > 0xb) SaveInput2<type>(dst0 + 0xb * F, dst1 + 0xb * F, db0, db1, bias, params);
            }
            else
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
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs6 += 2)
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
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs6 += 2)
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
                if (M > 0x0) SaveInput1<type>(dst0 + 0x0 * F, d00, bias, params);
                if (M > 0x1) SaveInput1<type>(dst0 + 0x1 * F, d10, bias, params);
                if (M > 0x2) SaveInput1<type>(dst0 + 0x2 * F, d20, bias, params);
                if (M > 0x3) SaveInput1<type>(dst0 + 0x3 * F, d30, bias, params);
                if (M > 0x4) SaveInput1<type>(dst0 + 0x4 * F, d40, bias, params);
                if (M > 0x5) SaveInput1<type>(dst0 + 0x5 * F, d50, bias, params);
                if (M > 0x6) SaveInput1<type>(dst0 + 0x6 * F, d60, bias, params);
                if (M > 0x7) SaveInput1<type>(dst0 + 0x7 * F, d70, bias, params);
                if (M > 0x8) SaveInput1<type>(dst0 + 0x8 * F, d80, bias, params);
                if (M > 0x9) SaveInput1<type>(dst0 + 0x9 * F, d90, bias, params);
                if (M > 0xa) SaveInput1<type>(dst0 + 0xa * F, da0, bias, params);
                if (M > 0xb) SaveInput1<type>(dst0 + 0xb * F, db0, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dstC,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 0x1: return InputConvolution1x1_2xM<type, 0x1>;
            case 0x2: return InputConvolution1x1_2xM<type, 0x2>;
            case 0x3: return InputConvolution1x1_2xM<type, 0x3>;
            case 0x4: return InputConvolution1x1_2xM<type, 0x4>;
            case 0x5: return InputConvolution1x1_2xM<type, 0x5>;
            case 0x6: return InputConvolution1x1_2xM<type, 0x6>;
            case 0x7: return InputConvolution1x1_2xM<type, 0x7>;
            case 0x8: return InputConvolution1x1_2xM<type, 0x8>;
            case 0x9: return InputConvolution1x1_2xM<type, 0x9>;
            case 0xa: return InputConvolution1x1_2xM<type, 0xa>;
            case 0xb: return InputConvolution1x1_2xM<type, 0xb>;
            case 0xc: return InputConvolution1x1_2xM<type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (a.bufH[0] == 0 || a.bufH[0] >= p.srcH)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 12;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    if (yInt > yBeg)
                    {
                        const uint16_t* src0 = src + yBeg * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (in < i1)
                            inputConvolution1x1_2xI(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    if (yEnd > yInt)
                    {
                        const uint16_t* src0 = src + yInt * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (en < e1)
                            inputConvolution1x1_2xE(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 2) * QF;
                }
            }
            else
            {
                size_t n = 12, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), m = bodyW - bodyWn;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xM = GetInputConvolution1x1_2xM<type>(m);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    for (size_t dy = yBeg; dy < yEnd; dy++)
                    {
                        const uint16_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        size_t dx = 0;
                        for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (dx < bodyW)
                            inputConvolution1x1_2xM(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 2) * QF;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            if (Is1x1(p))
                input = InputConvolution1x1_2<type>;
            else
                input = InputConvolution_2<type>;
        }

        void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInput<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            }
        }
    }
#endif
}
