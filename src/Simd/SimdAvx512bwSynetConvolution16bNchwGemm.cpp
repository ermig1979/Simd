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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        typedef Base::SynetConvolution16bNchwGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNchwGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertF(const float* src, size_t stride, uint16_t*& dst, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 src0 = _mm512_maskz_loadu_ps(tail0, src);
            __m512 src1 = _mm512_maskz_loadu_ps(tail1, src + stride);
            _mm512_storeu_si512(dst, Float32ToBFloat16Interlived(src0, src1));
            dst += DF;
        }

        static void Convert16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const float* src = ((float*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j = 0, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (; j < N; j += a.F)
            {
                __mmask16 tail = TailMask16(N - j);
                for (k = 0; k < K2; k += 2)
                    ConvertF(src + k * dS, dS, dst, tail, tail);
                for (; k < K; k += 2)
                    ConvertF(src + k * dS, dS, dst, tail, 0);
                for (; k < KH; k += 2)
                {
                    SetZero(dst);
                    dst += DF;
                }
                src += a.F;
            }
        }

        SIMD_INLINE void ReorderF(const uint16_t* src, size_t stride, uint16_t*& dst, __mmask16 tail0, __mmask16 tail1)
        {
            static const __m512i PERM_IDX = _mm512_set_epi16(
                0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
                0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
            __m256i src0 = _mm256_maskz_loadu_epi16(tail0, src);
            __m256i src1 = _mm256_maskz_loadu_epi16(tail1, src + stride);
            __m512i _src = _mm512_inserti64x4(_mm512_castsi256_si512(src0), src1, 1);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, _src));
            dst += DF;
        }

        static void Reorder16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const uint16_t* src = ((uint16_t*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j = 0, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (; j < N; j += a.F)
            {
                __mmask16 tail = TailMask16(N - j);
                for (k = 0; k < K2; k += 2)
                    ReorderF(src + k * dS, dS, dst, tail, tail);
                for (; k < K; k += 2)
                    ReorderF(src + k * dS, dS, dst, tail, 0);
                for (; k < KH; k += 2)
                {
                    SetZero(dst);
                    dst += DF;
                }
                src += a.F;
            }
        }

        ////-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNchwGemm_2xM(const uint16_t* weight0, const ConvParam& p, const AlgParam& a, 
            size_t K, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, 
                w0, s00, s01, s10, s11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem;
            const uint16_t* src1 = src0 + K * F;
            const uint16_t* weight1 = weight0 + 1 * K;
            const uint16_t* weight2 = weight0 + 2 * K;
            const uint16_t* weight3 = weight0 + 3 * K;
            const uint16_t* weight4 = weight0 + 4 * K;
            const uint16_t* weight5 = weight0 + 5 * K;
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
                for (size_t k0 = 0, k6 = k0 + 6 * K; k0 < K; k0 += 2, k6 += 2)
                {
                    s01 = _mm512_loadu_ps((float*)src0);
                    s00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(s01), Base::Bf16::SHIFT));
                    s01 = _mm512_and_ps(s01, m);
                    s11 = _mm512_loadu_ps((float*)src1);
                    s10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(s11), Base::Bf16::SHIFT));
                    s11 = _mm512_and_ps(s11, m);
                    if (M > 0x0)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k0 - 1)), m);
                        d00 = _mm512_fmadd_ps(w0, s00, d00);
                        d01 = _mm512_fmadd_ps(w0, s10, d01);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k0 - 0)), m);
                        d00 = _mm512_fmadd_ps(w0, s01, d00);
                        d01 = _mm512_fmadd_ps(w0, s11, d01);
                    }
                    if (M > 0x1)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k0 - 1)), m);
                        d10 = _mm512_fmadd_ps(w0, s00, d10);
                        d11 = _mm512_fmadd_ps(w0, s10, d11);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k0 - 0)), m);
                        d10 = _mm512_fmadd_ps(w0, s01, d10);
                        d11 = _mm512_fmadd_ps(w0, s11, d11);
                    }
                    if (M > 0x2)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k0 - 1)), m);
                        d20 = _mm512_fmadd_ps(w0, s00, d20);
                        d21 = _mm512_fmadd_ps(w0, s10, d21);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k0 - 0)), m);
                        d20 = _mm512_fmadd_ps(w0, s01, d20);
                        d21 = _mm512_fmadd_ps(w0, s11, d21);
                    }
                    if (M > 0x3)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k0 - 1)), m);
                        d30 = _mm512_fmadd_ps(w0, s00, d30);
                        d31 = _mm512_fmadd_ps(w0, s10, d31);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k0 - 0)), m);
                        d30 = _mm512_fmadd_ps(w0, s01, d30);
                        d31 = _mm512_fmadd_ps(w0, s11, d31);
                    }
                    if (M > 0x4)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k0 - 1)), m);
                        d40 = _mm512_fmadd_ps(w0, s00, d40);
                        d41 = _mm512_fmadd_ps(w0, s10, d41);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k0 - 0)), m);
                        d40 = _mm512_fmadd_ps(w0, s01, d40);
                        d41 = _mm512_fmadd_ps(w0, s11, d41);
                    }
                    if (M > 0x5)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k0 - 1)), m);
                        d50 = _mm512_fmadd_ps(w0, s00, d50);
                        d51 = _mm512_fmadd_ps(w0, s10, d51);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k0 - 0)), m);
                        d50 = _mm512_fmadd_ps(w0, s01, d50);
                        d51 = _mm512_fmadd_ps(w0, s11, d51);
                    }
                    if (M > 0x6)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k6 - 1)), m);
                        d60 = _mm512_fmadd_ps(w0, s00, d60);
                        d61 = _mm512_fmadd_ps(w0, s10, d61);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k6 - 0)), m);
                        d60 = _mm512_fmadd_ps(w0, s01, d60);
                        d61 = _mm512_fmadd_ps(w0, s11, d61);
                    }
                    if (M > 0x7)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k6 - 1)), m);
                        d70 = _mm512_fmadd_ps(w0, s00, d70);
                        d71 = _mm512_fmadd_ps(w0, s10, d71);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k6 - 0)), m);
                        d70 = _mm512_fmadd_ps(w0, s01, d70);
                        d71 = _mm512_fmadd_ps(w0, s11, d71);
                    }
                    if (M > 0x8)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k6 - 1)), m);
                        d80 = _mm512_fmadd_ps(w0, s00, d80);
                        d81 = _mm512_fmadd_ps(w0, s10, d81);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k6 - 0)), m);
                        d80 = _mm512_fmadd_ps(w0, s01, d80);
                        d81 = _mm512_fmadd_ps(w0, s11, d81);
                    }
                    if (M > 0x9)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k6 - 1)), m);
                        d90 = _mm512_fmadd_ps(w0, s00, d90);
                        d91 = _mm512_fmadd_ps(w0, s10, d91);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k6 - 0)), m);
                        d90 = _mm512_fmadd_ps(w0, s01, d90);
                        d91 = _mm512_fmadd_ps(w0, s11, d91);
                    }
                    if (M > 0xa)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k6 - 1)), m);
                        da0 = _mm512_fmadd_ps(w0, s00, da0);
                        da1 = _mm512_fmadd_ps(w0, s10, da1);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k6 - 0)), m);
                        da0 = _mm512_fmadd_ps(w0, s01, da0);
                        da1 = _mm512_fmadd_ps(w0, s11, da1);
                    }
                    if (M > 0xb)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k6 - 1)), m);
                        db0 = _mm512_fmadd_ps(w0, s00, db0);
                        db1 = _mm512_fmadd_ps(w0, s10, db1);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k6 - 0)), m);
                        db0 = _mm512_fmadd_ps(w0, s01, db0);
                        db1 = _mm512_fmadd_ps(w0, s11, db1);
                    }
                    src0 += DF;
                    src1 += DF;
                }
                if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, bias, params, 0x0, tails[1]), dst += dD, buf += dB;
                if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, bias, params, 0x1, tails[1]), dst += dD, buf += dB;
                if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, bias, params, 0x2, tails[1]), dst += dD, buf += dB;
                if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, bias, params, 0x3, tails[1]), dst += dD, buf += dB;
                if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, bias, params, 0x4, tails[1]), dst += dD, buf += dB;
                if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, bias, params, 0x5, tails[1]), dst += dD, buf += dB;
                if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, bias, params, 0x6, tails[1]), dst += dD, buf += dB;
                if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, bias, params, 0x7, tails[1]), dst += dD, buf += dB;
                if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, bias, params, 0x8, tails[1]), dst += dD, buf += dB;
                if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, bias, params, 0x9, tails[1]), dst += dD, buf += dB;
                if (M > 0xa) Save2<term, type>(dst, buf, da0, da1, bias, params, 0xa, tails[1]), dst += dD, buf += dB;
                if (M > 0xb) Save2<term, type>(dst, buf, db0, db1, bias, params, 0xb, tails[1]), dst += dD, buf += dB;
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
                for (size_t k0 = 0, k6 = k0 + 6 * K; k0 < K; k0 += 2, k6 += 2)
                {
                    s01 = _mm512_loadu_ps((float*)src0);
                    s00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(s01), Base::Bf16::SHIFT));
                    s01 = _mm512_and_ps(s01, m);
                    if (M > 0x0)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k0 - 1)), m);
                        d00 = _mm512_fmadd_ps(w0, s00, d00);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k0 - 0)), m);
                        d00 = _mm512_fmadd_ps(w0, s01, d00);
                    }
                    if (M > 0x1)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k0 - 1)), m);
                        d10 = _mm512_fmadd_ps(w0, s00, d10);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k0 - 0)), m);
                        d10 = _mm512_fmadd_ps(w0, s01, d10);
                    }
                    if (M > 0x2)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k0 - 1)), m);
                        d20 = _mm512_fmadd_ps(w0, s00, d20);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k0 - 0)), m);
                        d20 = _mm512_fmadd_ps(w0, s01, d20);
                    }
                    if (M > 0x3)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k0 - 1)), m);
                        d30 = _mm512_fmadd_ps(w0, s00, d30);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k0 - 0)), m);
                        d30 = _mm512_fmadd_ps(w0, s01, d30);
                    }
                    if (M > 0x4)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k0 - 1)), m);
                        d40 = _mm512_fmadd_ps(w0, s00, d40);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k0 - 0)), m);
                        d40 = _mm512_fmadd_ps(w0, s01, d40);
                    }
                    if (M > 0x5)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k0 - 1)), m);
                        d50 = _mm512_fmadd_ps(w0, s00, d50);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k0 - 0)), m);
                        d50 = _mm512_fmadd_ps(w0, s01, d50);
                    }
                    if (M > 0x6)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k6 - 1)), m);
                        d60 = _mm512_fmadd_ps(w0, s00, d60);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight0 + k6 - 0)), m);
                        d60 = _mm512_fmadd_ps(w0, s01, d60);
                    }
                    if (M > 0x7)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k6 - 1)), m);
                        d70 = _mm512_fmadd_ps(w0, s00, d70);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight1 + k6 - 0)), m);
                        d70 = _mm512_fmadd_ps(w0, s01, d70);
                    }
                    if (M > 0x8)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k6 - 1)), m);
                        d80 = _mm512_fmadd_ps(w0, s00, d80);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight2 + k6 - 0)), m);
                        d80 = _mm512_fmadd_ps(w0, s01, d80);
                    }
                    if (M > 0x9)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k6 - 1)), m);
                        d90 = _mm512_fmadd_ps(w0, s00, d90);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight3 + k6 - 0)), m);
                        d90 = _mm512_fmadd_ps(w0, s01, d90);
                    }
                    if (M > 0xa)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k6 - 1)), m);
                        da0 = _mm512_fmadd_ps(w0, s00, da0);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight4 + k6 - 0)), m);
                        da0 = _mm512_fmadd_ps(w0, s01, da0);
                    }
                    if (M > 0xb)
                    {
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k6 - 1)), m);
                        db0 = _mm512_fmadd_ps(w0, s00, db0);
                        w0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(weight5 + k6 - 0)), m);
                        db0 = _mm512_fmadd_ps(w0, s01, db0);
                    }
                    src0 += DF;
                }
                if (M > 0x0) Save1<term, type>(dst, buf, d00, bias, params, 0x0, tails[0]), dst += dD, buf += dB;
                if (M > 0x1) Save1<term, type>(dst, buf, d10, bias, params, 0x1, tails[0]), dst += dD, buf += dB;
                if (M > 0x2) Save1<term, type>(dst, buf, d20, bias, params, 0x2, tails[0]), dst += dD, buf += dB;
                if (M > 0x3) Save1<term, type>(dst, buf, d30, bias, params, 0x3, tails[0]), dst += dD, buf += dB;
                if (M > 0x4) Save1<term, type>(dst, buf, d40, bias, params, 0x4, tails[0]), dst += dD, buf += dB;
                if (M > 0x5) Save1<term, type>(dst, buf, d50, bias, params, 0x5, tails[0]), dst += dD, buf += dB;
                if (M > 0x6) Save1<term, type>(dst, buf, d60, bias, params, 0x6, tails[0]), dst += dD, buf += dB;
                if (M > 0x7) Save1<term, type>(dst, buf, d70, bias, params, 0x7, tails[0]), dst += dD, buf += dB;
                if (M > 0x8) Save1<term, type>(dst, buf, d80, bias, params, 0x8, tails[0]), dst += dD, buf += dB;
                if (M > 0x9) Save1<term, type>(dst, buf, d90, bias, params, 0x9, tails[0]), dst += dD, buf += dB;
                if (M > 0xa) Save1<term, type>(dst, buf, da0, bias, params, 0xa, tails[0]), dst += dD, buf += dB;
                if (M > 0xb) Save1<term, type>(dst, buf, db0, bias, params, 0xb, tails[0]), dst += dD, buf += dB;
            }
        }

        typedef void(*Convolution16bNchwGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, 
            const uint16_t* weight0, const float* bias, const float* params, float* buf, uint8_t* dst, const __mmask16 tails[2]);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNchwGemm_2xM_Ptr GetConvolution16bNchwGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Convolution16bNchwGemm_2xM<term, type, 0x1>;
            case 0x2: return Convolution16bNchwGemm_2xM<term, type, 0x2>;
            case 0x3: return Convolution16bNchwGemm_2xM<term, type, 0x3>;
            case 0x4: return Convolution16bNchwGemm_2xM<term, type, 0x4>;
            case 0x5: return Convolution16bNchwGemm_2xM<term, type, 0x5>;
            case 0x6: return Convolution16bNchwGemm_2xM<term, type, 0x6>;
            case 0x7: return Convolution16bNchwGemm_2xM<term, type, 0x7>;
            case 0x8: return Convolution16bNchwGemm_2xM<term, type, 0x8>;
            case 0x9: return Convolution16bNchwGemm_2xM<term, type, 0x9>;
            case 0xa: return Convolution16bNchwGemm_2xM<term, type, 0xa>;
            case 0xb: return Convolution16bNchwGemm_2xM<term, type, 0xb>;
            case 0xc: return Convolution16bNchwGemm_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNchwGemm_2(const uint16_t* weight, const ConvParam& p, const AlgParam& a, 
            size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* src, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            Convolution16bNchwGemm_2xM_Ptr convolution_2xN = GetConvolution16bNchwGemm_2xM<term, type>(n);
            Convolution16bNchwGemm_2xM_Ptr convolution_2xM = GetConvolution16bNchwGemm_2xM<term, type>(m);

            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                __mmask16 tails[2] = { TailMask16(dS), TailMask16(dS - F) };
                const uint16_t* w = weight;
                float* b = buf + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                    convolution_2xN(w, p, a, K, zero, src, bias + i, params + i * dp, b, d, tails);
                for (; i < n1; i += m, w += m * dW, b += m * dB, d += m * dD)
                    convolution_2xM(w, p, a, K, zero, src, bias + i, params + i * dp, b, d, tails);
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNchwGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if(p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNchwGemm::SynetConvolution16bNchwGemm(const ConvParam & p)
            : Avx2::SynetConvolution16bNchwGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                if (_is1x1)
                    _convert = Reorder16bNchwGemm1x1;
                //else
                //    _convert = Reorder16bNhwcGemm;
            }
            else
            {
                if (_is1x1)
                    _convert = Convert16bNchwGemm1x1;
                //else
                //    _convert = Convert16bNhwcGemm;
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
