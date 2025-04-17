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
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        typedef Base::SynetConvolution16bNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV0::PostprocessPtr PostprocessPtr;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const float* src = (float*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        Avx512bw::Float32ToBFloat16(src + sc, dst + sc * cD);
                    if (tailC)
                        Avx512bw::Float32ToBFloat16(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        static void Reorder16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const uint16_t* src = (uint16_t*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        Avx512bw::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Avx512bw::Copy(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int M> void Convolution16bNhwcSpecV0_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dD = a.macroD, dX = a.microC;
            const uint16_t* weight1 = weight0 + a.srcC * a.K * F;
            const uint16_t* src1 = src0 + 1 * dX;
            const uint16_t* src2 = src0 + 2 * dX;
            const uint16_t* src3 = src0 + 3 * dX;
            const uint16_t* src4 = src0 + 4 * dX;
            const uint16_t* src5 = src0 + 5 * dX;
            if (dstC > F)
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
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_loadu_ps(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_loadu_ps(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_loadu_ps(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_loadu_ps(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_loadu_ps(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_loadu_ps(dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_loadu_ps(dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_loadu_ps(dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_loadu_ps(dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_loadu_ps(dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_loadu_ps(dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_loadu_ps(dst + 0xb * dD + F);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + 6 * dX; offs0 < end; offs0 += 2, offs6 += 2)
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
                }
                if (M > 0x0) _mm512_storeu_ps(dst + 0x0 * dD + 0, d00), _mm512_storeu_ps(dst + 0x0 * dD + F, d01);
                if (M > 0x1) _mm512_storeu_ps(dst + 0x1 * dD + 0, d10), _mm512_storeu_ps(dst + 0x1 * dD + F, d11);
                if (M > 0x2) _mm512_storeu_ps(dst + 0x2 * dD + 0, d20), _mm512_storeu_ps(dst + 0x2 * dD + F, d21);
                if (M > 0x3) _mm512_storeu_ps(dst + 0x3 * dD + 0, d30), _mm512_storeu_ps(dst + 0x3 * dD + F, d31);
                if (M > 0x4) _mm512_storeu_ps(dst + 0x4 * dD + 0, d40), _mm512_storeu_ps(dst + 0x4 * dD + F, d41);
                if (M > 0x5) _mm512_storeu_ps(dst + 0x5 * dD + 0, d50), _mm512_storeu_ps(dst + 0x5 * dD + F, d51);
                if (M > 0x6) _mm512_storeu_ps(dst + 0x6 * dD + 0, d60), _mm512_storeu_ps(dst + 0x6 * dD + F, d61);
                if (M > 0x7) _mm512_storeu_ps(dst + 0x7 * dD + 0, d70), _mm512_storeu_ps(dst + 0x7 * dD + F, d71);
                if (M > 0x8) _mm512_storeu_ps(dst + 0x8 * dD + 0, d80), _mm512_storeu_ps(dst + 0x8 * dD + F, d81);
                if (M > 0x9) _mm512_storeu_ps(dst + 0x9 * dD + 0, d90), _mm512_storeu_ps(dst + 0x9 * dD + F, d91);
                if (M > 0xa) _mm512_storeu_ps(dst + 0xa * dD + 0, da0), _mm512_storeu_ps(dst + 0xa * dD + F, da1);
                if (M > 0xb) _mm512_storeu_ps(dst + 0xb * dD + 0, db0), _mm512_storeu_ps(dst + 0xb * dD + F, db1);
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
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + 6 * dX; offs0 < end; offs0 += 2, offs6 += 2)
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
                }
                if (M > 0x0) _mm512_storeu_ps(dst + 0x0 * dD + 0, d00);
                if (M > 0x1) _mm512_storeu_ps(dst + 0x1 * dD + 0, d10);
                if (M > 0x2) _mm512_storeu_ps(dst + 0x2 * dD + 0, d20);
                if (M > 0x3) _mm512_storeu_ps(dst + 0x3 * dD + 0, d30);
                if (M > 0x4) _mm512_storeu_ps(dst + 0x4 * dD + 0, d40);
                if (M > 0x5) _mm512_storeu_ps(dst + 0x5 * dD + 0, d50);
                if (M > 0x6) _mm512_storeu_ps(dst + 0x6 * dD + 0, d60);
                if (M > 0x7) _mm512_storeu_ps(dst + 0x7 * dD + 0, d70);
                if (M > 0x8) _mm512_storeu_ps(dst + 0x8 * dD + 0, d80);
                if (M > 0x9) _mm512_storeu_ps(dst + 0x9 * dD + 0, d90);
                if (M > 0xa) _mm512_storeu_ps(dst + 0xa * dD + 0, da0);
                if (M > 0xb) _mm512_storeu_ps(dst + 0xb * dD + 0, db0);
            }
        }

        typedef void(*Convolution16bNhwcSpecV0_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int zero, const uint16_t* weight0, float* dst);

        static Convolution16bNhwcSpecV0_2xM_Ptr GetConvolution16bNhwcSpecV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Convolution16bNhwcSpecV0_2xM<0x1>;
            case 0x2: return Convolution16bNhwcSpecV0_2xM<0x2>;
            case 0x3: return Convolution16bNhwcSpecV0_2xM<0x3>;
            case 0x4: return Convolution16bNhwcSpecV0_2xM<0x4>;
            case 0x5: return Convolution16bNhwcSpecV0_2xM<0x5>;
            case 0x6: return Convolution16bNhwcSpecV0_2xM<0x6>;
            case 0x7: return Convolution16bNhwcSpecV0_2xM<0x7>;
            case 0x8: return Convolution16bNhwcSpecV0_2xM<0x8>;
            case 0x9: return Convolution16bNhwcSpecV0_2xM<0x9>;
            case 0xa: return Convolution16bNhwcSpecV0_2xM<0xa>;
            case 0xb: return Convolution16bNhwcSpecV0_2xM<0xb>;
            case 0xc: return Convolution16bNhwcSpecV0_2xM<0xc>;
            }
            assert(0);
            return NULL;
        }

        static void Convolution16bNhwcSpecV0_2(const uint16_t* src, const ConvParam& p,
            const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, float* dst)
        {
            size_t nK = DivHi(srcC, a.microC) * a.K;
            size_t n1 = dstH * a.srcW - a.padH, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.srcC * a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcSpecV0_2xM_Ptr convolution_2xN = GetConvolution16bNhwcSpecV0_2xM(n);
            Convolution16bNhwcSpecV0_2xM_Ptr convolution_2xM = GetConvolution16bNhwcSpecV0_2xM(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                for (; i < nn; i += n)
                    convolution_2xN(src + i * dS, p, a, offs, nK, dC, zero, weight, dst + i * dD);
                for (; i < n1; i += m)
                    convolution_2xM(src + i * dS, p, a, offs, nK, dC, zero, weight, dst + i * dD);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type>  void Postprocess16bNhwcSpecV0(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F);
            __mmask16 tailD = TailMask16(dstC - dstCF);
            size_t rowGap = a.padH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                        Avx512bw::Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tailD)
                        Avx512bw::Postprocess<term, type>(src, bias, params, dc, dst, tailD);
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        template<SimdConvolutionActivationType type> void SetPostprocess(const ConvParam& p, const AlgParam& a, PostprocessPtr & postprocess)
        {
            if (p.dstT == SimdTensorData16b)
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast16b, type>;
            else
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast32f, type>;
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV0::SynetConvolution16bNhwcSpecV0(const ConvParam & p)
            : Avx2::SynetConvolution16bNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, 12, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV0;
            else
                _preprocess = Convert16bNhwcSpecV0;
            _convolution = Convolution16bNhwcSpecV0_2;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRelu: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationLeakyRelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRestrictRange: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationPrelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationElu: SetPostprocess<SimdConvolutionActivationElu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHswish: SetPostprocess<SimdConvolutionActivationHswish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationMish: SetPostprocess<SimdConvolutionActivationMish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHardSigmoid: SetPostprocess<SimdConvolutionActivationHardSigmoid>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationSwish: SetPostprocess<SimdConvolutionActivationSwish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationGelu: SetPostprocess<SimdConvolutionActivationGelu>(p, _alg, _postprocess); break;
            default: assert(0);
            }
        }
    }
#endif
}
