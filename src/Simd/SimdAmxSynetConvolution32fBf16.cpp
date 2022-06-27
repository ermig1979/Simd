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
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdAmx.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMX_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Amx
    {
        typedef Base::SynetConvolution32fBf16Nhwc::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvertPtr Convert;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvolutionPtr Convolution;

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
                if (M > 0x0) Avx512f::Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 0x1) Avx512f::Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 0x2) Avx512f::Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 0x3) Avx512f::Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 0x4) Avx512f::Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 0x5) Avx512f::Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 0x6) Avx512f::Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 0x7) Avx512f::Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 0x8) Avx512f::Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
                if (M > 0x9) Avx512f::Save2<term, type>(dst, d90, d91, bias, params, tails), dst += dD;
                if (M > 0xa) Avx512f::Save2<term, type>(dst, da0, da1, bias, params, tails), dst += dD;
                if (M > 0xb) Avx512f::Save2<term, type>(dst, db0, db1, bias, params, tails), dst += dD;
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
                if (M > 0x0) Avx512f::Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 0x1) Avx512f::Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 0x2) Avx512f::Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 0x3) Avx512f::Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 0x4) Avx512f::Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 0x5) Avx512f::Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 0x6) Avx512f::Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 0x7) Avx512f::Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 0x8) Avx512f::Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
                if (M > 0x9) Avx512f::Save1<term, type>(dst, d90, bias, params, tails), dst += dD;
                if (M > 0xa) Avx512f::Save1<term, type>(dst, da0, bias, params, tails), dst += dD;
                if (M > 0xb) Avx512f::Save1<term, type>(dst, db0, bias, params, tails), dst += dD;
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

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2x2(const uint16_t* src0, const ConvParam32f& p,
            size_t dstC, size_t dstW, size_t srcC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY, 
                dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX;
            size_t srcC32 = AlignLo(srcC, 32), strideS = srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* src1 = src0 + srcC * 16 * dS, * weight1 = weight0 + 32;

            TileConf body, tail;
            body.rows[0] = 16;
            body.rows[1] = 16;
            body.rows[2] = uint8_t(dstW - 16);
            body.rows[3] = uint8_t(dstW - 16);
            body.rows[4] = 16;
            body.rows[5] = uint8_t(dstW - 16);
            body.rows[6] = 16;
            body.rows[7] = 16;
            body.colsb[0] = 64;
            body.colsb[1] = uint16_t((dstC - 16) * 4);
            body.colsb[2] = 64;
            body.colsb[3] = uint16_t((dstC - 16) * 4);
            body.colsb[4] = 64;
            body.colsb[5] = 64;
            body.colsb[6] = 64;
            body.colsb[7] = uint16_t((dstC - 16) * 4);
            if (srcC32 < srcC)
            {
                size_t tailC = srcC - srcC32;
                tail = body;
                tail.rows[6] = tailC / 2;
                tail.rows[7] = tailC / 2;
                tail.colsb[4] = tailC * 2;
                tail.colsb[5] = tailC * 2;
            }
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_loadd(3, dst + 16 * dD + F, strideD);
            }
            for (size_t ky = 0; ky < kY; ++ky)
            {
                for (size_t kx = 0; kx < kX; ++kx)
                {
                    size_t sc = 0, offs = ky * dY + kx * dX;
                    _tile_loadconfig(&body);
                    for (;sc < srcC32; sc += 32)
                    {
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    if (sc < srcC)
                    {
                        _tile_loadconfig(&tail);
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    weight0 += srcC * 32;
                    weight1 += srcC * 32;
                }
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                for (size_t w = 0; w < dstW; ++w, dst += dD)
                    Apply2<type>(dst, dst, bias, params, tailD);
            }
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

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32), strideS = srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* src1 = src0 + srcC * 16, *weight1 = weight0 + 32;

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
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_loadd(3, dst + 16 * dD + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            if(sc < srcC)
            {
                size_t tailC = srcC - sc;
                conf.rows[6] = tailC / 2;
                conf.rows[7] = tailC / 2;
                conf.colsb[4] = tailC * 2;
                conf.colsb[5] = tailC * 2;
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                for(size_t s = 0; s < dstS; ++s, dst += dD) 
                    Apply2<type>(dst, dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32), strideS = srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* src1 = src0 + srcC * 16;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = dstS - 16;
            conf.rows[4] = 16;
            conf.rows[5] = dstS - 16;
            conf.rows[6] = 16;
            conf.colsb[0] = dstC * 4;
            conf.colsb[2] = dstC * 4;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = dstC * 4;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = srcC - sc;
                conf.rows[6] = tailC / 2;
                conf.colsb[4] = tailC * 2;
                conf.colsb[5] = tailC * 2;
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                for (size_t s = 0; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32), strideS = srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + 32;

            TileConf conf;
            conf.rows[0] = dstS;
            conf.rows[1] = dstS;
            conf.rows[4] = dstS;
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = (dstC - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = (dstC - 16) * 4;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = srcC - sc;
                conf.rows[6] = tailC / 2;
                conf.rows[7] = tailC / 2;
                conf.colsb[4] = tailC * 2;
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                for (size_t s = 0; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32), strideS = srcC * 2, strideW = 128, strideD = dD * 4;

            TileConf conf;
            conf.rows[0] = dstS;
            conf.rows[4] = dstS;
            conf.rows[6] = 16;
            conf.colsb[0] = dstC * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = dstC * 4;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = srcC - sc;
                conf.rows[6] = tailC / 2;
                conf.colsb[4] = tailC * 2;
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, dst + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                for (size_t s = 0; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, dst, bias, params, tailD);
            }
        }

        typedef void (*ConvolutionBf16NhwcGemmPtr)(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst);

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = dstH * p.dstW, n32 = AlignLoAny(n, 32), m = n - n32, dW = AlignHi(srcC, 2) * DF;
            ConvolutionBf16NhwcGemmPtr body_2 = ConvolutionBf16NhwcGemm_2x2<type>;
            ConvolutionBf16NhwcGemmPtr tail_2 = m > 16 ? ConvolutionBf16NhwcGemm_2x2<type> : ConvolutionBf16NhwcGemm_1x2<type>;
            ConvolutionBf16NhwcGemmPtr body_1 = ConvolutionBf16NhwcGemm_2x1<type>;
            ConvolutionBf16NhwcGemmPtr tail_1 = m > 16 ? ConvolutionBf16NhwcGemm_2x1<type> : ConvolutionBf16NhwcGemm_1x1<type>;

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
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < n32; i += 32, s += 32 * srcC, d += 32 * p.dstC)
                        body_2(s, p, srcC, 32, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_2(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                else
                {
                    for (; i < n32; i += 32, s += 32 * srcC, d += 32 * p.dstC)
                        body_1(s, p, srcC, 32, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_1(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convolution* convolutions)
        {
            if (p.Is1x1() || a.mode)
            {
                convolutions[TermLast] = ConvolutionBf16NhwcGemm_2<type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcGemm_2<SimdConvolutionActivationIdentity>;
            }
            else
            {
                convolutions[TermLast] = ConvolutionBf16NhwcConv_2<TermLast, type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcConv_2<TermInterim, type>;
            }
        }

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f & p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetConvolution32fBf16Nhwc(p)
#else
            : Avx512bf16::SynetConvolution32fBf16Nhwc(p)
#endif
        {
#if defined(SIMD_AMX_EMULATE)
            size_t microD, microHW, microC;
            if (p.Is1x1())
            {
                microD = 16 * 2;
                microHW = 16 * 2;
                microC = 16 * 2;
            }
            else
            {
                microD = 16 * 2;
                microHW = 12;
                microC = 2;
            }
            SetAlgParam(microD, microHW, microC, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_alg.mode)
                _convert = Avx512bw::ConvolutionBf16NhwcConvertGemm;
            else
                _convert = Avx512bw::ConvolutionBf16NhwcConvertConv;
#else
            size_t microD = 16 * 2;
            size_t microHW = 16 * 2;
            size_t microC = 16 * 2;
            SetAlgParam(microD, microHW, microC, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_alg.mode)
                _convert = Avx512bf16::ConvolutionBf16NhwcConvertGemm;
            else
                _convert = Avx512bf16::ConvolutionBf16NhwcConvertConv;
#endif
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

        //-----------------------------------------------------------------------------------------

        void* SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam32f param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (Base::Bf16Soft(compatibility))
            {
                if (Base::SynetConvolution32fBf16Nhwc::Preferable(param))
                    return new Amx::SynetConvolution32fBf16Nhwc(param);
                else
                    return new Base::SynetConvolution32fBf16Gemm(param);
            }
#if defined(SIMD_AMX_EMULATE)
            return Avx512bw::SynetConvolution32fInit(batch, conv, compatibility);
#else
            return Avx512bf16::SynetConvolution32fInit(batch, conv, compatibility);
#endif
        }
    }
#endif
}
