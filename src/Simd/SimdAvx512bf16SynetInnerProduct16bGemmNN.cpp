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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512BF16_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512bf16
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        SIMD_INLINE __m512bh LoadBf16(const uint16_t* p)
        {
            return (__m512bh)_mm512_loadu_si512((const void*)p);
        }

        SIMD_INLINE __m512bh PairBf16(const uint16_t* p)
        {
            return (__m512bh)_mm512_set1_epi32(*(const uint32_t*)p);
        }

        SIMD_INLINE void StoreBf16(uint8_t* ptr, __m512 value, __mmask16 tail)
        {
            _mm256_mask_storeu_epi16(ptr, tail, (__m256i)_mm512_cvtneps_pbh(value));
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1Bf(uint8_t* ptr, float* buf, __m512 val0, const __m512* bias, const __m512* params, __mmask16 tail)
        {
            if (term == Term16bLast16b)
                StoreBf16(ptr, Activate<type>(_mm512_add_ps(val0, bias[0]), params, 0), tail);
            else
                Avx512bw::Save1<term, type>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2Bf(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, __mmask16 tail)
        {
            if (term == Term16bLast16b)
            {
                StoreBf16(ptr + 0 * DF, Activate<type>(_mm512_add_ps(val0, bias[0]), params, 0), __mmask16(-1));
                StoreBf16(ptr + 1 * DF, Activate<type>(_mm512_add_ps(val1, bias[1]), params, 1), tail);
            }
            else
                Avx512bw::Save2<term, type>(ptr, buf, val0, val1, bias, params, tail);
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ConvertA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            if (p.K == a.aK)
            {
                Float32ToBFloat16(src, K * M, dst);
            }
            else
            {
                size_t KDF = Simd::AlignLo(p.K, DF);
                __mmask32 tail = TailMask32(p.K - KDF);
                __mmask32 save = TailMask32(a.aK - KDF);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t k = 0;
                    for (; k < KDF; k += DF)
                        Float32ToBFloat16(src + k, dst + k);
                    if (tail)
                        Float32ToBFloat16(src + k, dst + k, tail, save);
                    src += p.K;
                    dst += a.aK;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ReorderA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t KDF = Simd::AlignLo(p.K, DF);
            __mmask32 tail = TailMask32(p.K - KDF);
            __mmask32 save = TailMask32(a.aK - KDF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KDF; k += DF)
                    Copy(src + k, dst + k);
                if (tail)
                    Copy(src + k, dst + k, tail, save);
                src += p.K;
                dst += a.aK;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBn(const float* src, size_t stride, uint16_t* dst)
        {
            static const __m512i PERM = _mm512_set_epi16(
                31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8,
                23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
            __m512 even = _mm512_loadu_ps(src + 0 * stride);
            __m512 odd = _mm512_loadu_ps(src + 1 * stride);
            __m512i pack = (__m512i)_mm512_cvtne2ps_pbh(odd, even);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM, pack));
        }

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ConvertBn(ps + f, p.N, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void InnerProduct16bGemmNN_2xM(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t N, size_t K, int update, const uint16_t* B0, float* C, const __m512* bias, const __m512* params, uint8_t* dst)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71;
            size_t dC = a.cN, dA = a.aK, dD = p.N * a.eC;
            const uint16_t* B1 = B0 + a.bK * F;
            const uint16_t* A1 = A0 + 1 * dA;
            const uint16_t* A2 = A0 + 2 * dA;
            const uint16_t* A3 = A0 + 3 * dA;
            const uint16_t* A4 = A0 + 4 * dA;
            const uint16_t* A5 = A0 + 5 * dA;
            const uint16_t* A6 = A0 + 6 * dA;
            const uint16_t* A7 = A0 + 7 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = _mm512_loadu_ps(C + 0 * dC + 0), c01 = _mm512_loadu_ps(C + 0 * dC + F);
                    if (M > 1) c10 = _mm512_loadu_ps(C + 1 * dC + 0), c11 = _mm512_loadu_ps(C + 1 * dC + F);
                    if (M > 2) c20 = _mm512_loadu_ps(C + 2 * dC + 0), c21 = _mm512_loadu_ps(C + 2 * dC + F);
                    if (M > 3) c30 = _mm512_loadu_ps(C + 3 * dC + 0), c31 = _mm512_loadu_ps(C + 3 * dC + F);
                    if (M > 4) c40 = _mm512_loadu_ps(C + 4 * dC + 0), c41 = _mm512_loadu_ps(C + 4 * dC + F);
                    if (M > 5) c50 = _mm512_loadu_ps(C + 5 * dC + 0), c51 = _mm512_loadu_ps(C + 5 * dC + F);
                    if (M > 6) c60 = _mm512_loadu_ps(C + 6 * dC + 0), c61 = _mm512_loadu_ps(C + 6 * dC + F);
                    if (M > 7) c70 = _mm512_loadu_ps(C + 7 * dC + 0), c71 = _mm512_loadu_ps(C + 7 * dC + F);
                }
                else
                {
                    if (M > 0) c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                    if (M > 1) c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                    if (M > 2) c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                    if (M > 3) c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                    if (M > 4) c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                    if (M > 5) c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
                    if (M > 6) c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
                    if (M > 7) c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
                }
                size_t k = 0;
                for (; k + 4 <= K; k += 4)
                {
                    __m512bh b0 = LoadBf16(B0);
                    __m512bh b1 = LoadBf16(B1);
                    __m512bh b2 = LoadBf16(B0 + DF);
                    __m512bh b3 = LoadBf16(B1 + DF);
                    if (M > 0) { __m512bh a = PairBf16(A0 + k); c00 = _mm512_dpbf16_ps(c00, a, b0); c01 = _mm512_dpbf16_ps(c01, a, b1); }
                    if (M > 1) { __m512bh a = PairBf16(A1 + k); c10 = _mm512_dpbf16_ps(c10, a, b0); c11 = _mm512_dpbf16_ps(c11, a, b1); }
                    if (M > 2) { __m512bh a = PairBf16(A2 + k); c20 = _mm512_dpbf16_ps(c20, a, b0); c21 = _mm512_dpbf16_ps(c21, a, b1); }
                    if (M > 3) { __m512bh a = PairBf16(A3 + k); c30 = _mm512_dpbf16_ps(c30, a, b0); c31 = _mm512_dpbf16_ps(c31, a, b1); }
                    if (M > 4) { __m512bh a = PairBf16(A4 + k); c40 = _mm512_dpbf16_ps(c40, a, b0); c41 = _mm512_dpbf16_ps(c41, a, b1); }
                    if (M > 5) { __m512bh a = PairBf16(A5 + k); c50 = _mm512_dpbf16_ps(c50, a, b0); c51 = _mm512_dpbf16_ps(c51, a, b1); }
                    if (M > 6) { __m512bh a = PairBf16(A6 + k); c60 = _mm512_dpbf16_ps(c60, a, b0); c61 = _mm512_dpbf16_ps(c61, a, b1); }
                    if (M > 7) { __m512bh a = PairBf16(A7 + k); c70 = _mm512_dpbf16_ps(c70, a, b0); c71 = _mm512_dpbf16_ps(c71, a, b1); }
                    if (M > 0) { __m512bh a = PairBf16(A0 + k + 2); c00 = _mm512_dpbf16_ps(c00, a, b2); c01 = _mm512_dpbf16_ps(c01, a, b3); }
                    if (M > 1) { __m512bh a = PairBf16(A1 + k + 2); c10 = _mm512_dpbf16_ps(c10, a, b2); c11 = _mm512_dpbf16_ps(c11, a, b3); }
                    if (M > 2) { __m512bh a = PairBf16(A2 + k + 2); c20 = _mm512_dpbf16_ps(c20, a, b2); c21 = _mm512_dpbf16_ps(c21, a, b3); }
                    if (M > 3) { __m512bh a = PairBf16(A3 + k + 2); c30 = _mm512_dpbf16_ps(c30, a, b2); c31 = _mm512_dpbf16_ps(c31, a, b3); }
                    if (M > 4) { __m512bh a = PairBf16(A4 + k + 2); c40 = _mm512_dpbf16_ps(c40, a, b2); c41 = _mm512_dpbf16_ps(c41, a, b3); }
                    if (M > 5) { __m512bh a = PairBf16(A5 + k + 2); c50 = _mm512_dpbf16_ps(c50, a, b2); c51 = _mm512_dpbf16_ps(c51, a, b3); }
                    if (M > 6) { __m512bh a = PairBf16(A6 + k + 2); c60 = _mm512_dpbf16_ps(c60, a, b2); c61 = _mm512_dpbf16_ps(c61, a, b3); }
                    if (M > 7) { __m512bh a = PairBf16(A7 + k + 2); c70 = _mm512_dpbf16_ps(c70, a, b2); c71 = _mm512_dpbf16_ps(c71, a, b3); }
                    B0 += 2 * DF;
                    B1 += 2 * DF;
                }
                for (; k < K; k += 2)
                {
                    __m512bh b0 = LoadBf16(B0);
                    __m512bh b1 = LoadBf16(B1);
                    if (M > 0) { __m512bh a = PairBf16(A0 + k); c00 = _mm512_dpbf16_ps(c00, a, b0); c01 = _mm512_dpbf16_ps(c01, a, b1); }
                    if (M > 1) { __m512bh a = PairBf16(A1 + k); c10 = _mm512_dpbf16_ps(c10, a, b0); c11 = _mm512_dpbf16_ps(c11, a, b1); }
                    if (M > 2) { __m512bh a = PairBf16(A2 + k); c20 = _mm512_dpbf16_ps(c20, a, b0); c21 = _mm512_dpbf16_ps(c21, a, b1); }
                    if (M > 3) { __m512bh a = PairBf16(A3 + k); c30 = _mm512_dpbf16_ps(c30, a, b0); c31 = _mm512_dpbf16_ps(c31, a, b1); }
                    if (M > 4) { __m512bh a = PairBf16(A4 + k); c40 = _mm512_dpbf16_ps(c40, a, b0); c41 = _mm512_dpbf16_ps(c41, a, b1); }
                    if (M > 5) { __m512bh a = PairBf16(A5 + k); c50 = _mm512_dpbf16_ps(c50, a, b0); c51 = _mm512_dpbf16_ps(c51, a, b1); }
                    if (M > 6) { __m512bh a = PairBf16(A6 + k); c60 = _mm512_dpbf16_ps(c60, a, b0); c61 = _mm512_dpbf16_ps(c61, a, b1); }
                    if (M > 7) { __m512bh a = PairBf16(A7 + k); c70 = _mm512_dpbf16_ps(c70, a, b0); c71 = _mm512_dpbf16_ps(c71, a, b1); }
                    B0 += DF;
                    B1 += DF;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0) Save2Bf<term, type>(dst, C, c00, c01, bias, params, tail), C += dC, dst += dD;
                if (M > 1) Save2Bf<term, type>(dst, C, c10, c11, bias, params, tail), C += dC, dst += dD;
                if (M > 2) Save2Bf<term, type>(dst, C, c20, c21, bias, params, tail), C += dC, dst += dD;
                if (M > 3) Save2Bf<term, type>(dst, C, c30, c31, bias, params, tail), C += dC, dst += dD;
                if (M > 4) Save2Bf<term, type>(dst, C, c40, c41, bias, params, tail), C += dC, dst += dD;
                if (M > 5) Save2Bf<term, type>(dst, C, c50, c51, bias, params, tail), C += dC, dst += dD;
                if (M > 6) Save2Bf<term, type>(dst, C, c60, c61, bias, params, tail), C += dC, dst += dD;
                if (M > 7) Save2Bf<term, type>(dst, C, c70, c71, bias, params, tail), C += dC, dst += dD;
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = _mm512_loadu_ps(C + 0 * dC + 0);
                    if (M > 1) c10 = _mm512_loadu_ps(C + 1 * dC + 0);
                    if (M > 2) c20 = _mm512_loadu_ps(C + 2 * dC + 0);
                    if (M > 3) c30 = _mm512_loadu_ps(C + 3 * dC + 0);
                    if (M > 4) c40 = _mm512_loadu_ps(C + 4 * dC + 0);
                    if (M > 5) c50 = _mm512_loadu_ps(C + 5 * dC + 0);
                    if (M > 6) c60 = _mm512_loadu_ps(C + 6 * dC + 0);
                    if (M > 7) c70 = _mm512_loadu_ps(C + 7 * dC + 0);
                }
                else
                {
                    if (M > 0) c00 = _mm512_setzero_ps();
                    if (M > 1) c10 = _mm512_setzero_ps();
                    if (M > 2) c20 = _mm512_setzero_ps();
                    if (M > 3) c30 = _mm512_setzero_ps();
                    if (M > 4) c40 = _mm512_setzero_ps();
                    if (M > 5) c50 = _mm512_setzero_ps();
                    if (M > 6) c60 = _mm512_setzero_ps();
                    if (M > 7) c70 = _mm512_setzero_ps();
                }
                size_t k = 0;
                for (; k + 4 <= K; k += 4)
                {
                    __m512bh b0 = LoadBf16(B0);
                    __m512bh b2 = LoadBf16(B0 + DF);
                    if (M > 0) { __m512bh a = PairBf16(A0 + k); c00 = _mm512_dpbf16_ps(c00, a, b0); a = PairBf16(A0 + k + 2); c00 = _mm512_dpbf16_ps(c00, a, b2); }
                    if (M > 1) { __m512bh a = PairBf16(A1 + k); c10 = _mm512_dpbf16_ps(c10, a, b0); a = PairBf16(A1 + k + 2); c10 = _mm512_dpbf16_ps(c10, a, b2); }
                    if (M > 2) { __m512bh a = PairBf16(A2 + k); c20 = _mm512_dpbf16_ps(c20, a, b0); a = PairBf16(A2 + k + 2); c20 = _mm512_dpbf16_ps(c20, a, b2); }
                    if (M > 3) { __m512bh a = PairBf16(A3 + k); c30 = _mm512_dpbf16_ps(c30, a, b0); a = PairBf16(A3 + k + 2); c30 = _mm512_dpbf16_ps(c30, a, b2); }
                    if (M > 4) { __m512bh a = PairBf16(A4 + k); c40 = _mm512_dpbf16_ps(c40, a, b0); a = PairBf16(A4 + k + 2); c40 = _mm512_dpbf16_ps(c40, a, b2); }
                    if (M > 5) { __m512bh a = PairBf16(A5 + k); c50 = _mm512_dpbf16_ps(c50, a, b0); a = PairBf16(A5 + k + 2); c50 = _mm512_dpbf16_ps(c50, a, b2); }
                    if (M > 6) { __m512bh a = PairBf16(A6 + k); c60 = _mm512_dpbf16_ps(c60, a, b0); a = PairBf16(A6 + k + 2); c60 = _mm512_dpbf16_ps(c60, a, b2); }
                    if (M > 7) { __m512bh a = PairBf16(A7 + k); c70 = _mm512_dpbf16_ps(c70, a, b0); a = PairBf16(A7 + k + 2); c70 = _mm512_dpbf16_ps(c70, a, b2); }
                    B0 += 2 * DF;
                }
                for (; k < K; k += 2)
                {
                    __m512bh b0 = LoadBf16(B0);
                    if (M > 0) { __m512bh a = PairBf16(A0 + k); c00 = _mm512_dpbf16_ps(c00, a, b0); }
                    if (M > 1) { __m512bh a = PairBf16(A1 + k); c10 = _mm512_dpbf16_ps(c10, a, b0); }
                    if (M > 2) { __m512bh a = PairBf16(A2 + k); c20 = _mm512_dpbf16_ps(c20, a, b0); }
                    if (M > 3) { __m512bh a = PairBf16(A3 + k); c30 = _mm512_dpbf16_ps(c30, a, b0); }
                    if (M > 4) { __m512bh a = PairBf16(A4 + k); c40 = _mm512_dpbf16_ps(c40, a, b0); }
                    if (M > 5) { __m512bh a = PairBf16(A5 + k); c50 = _mm512_dpbf16_ps(c50, a, b0); }
                    if (M > 6) { __m512bh a = PairBf16(A6 + k); c60 = _mm512_dpbf16_ps(c60, a, b0); }
                    if (M > 7) { __m512bh a = PairBf16(A7 + k); c70 = _mm512_dpbf16_ps(c70, a, b0); }
                    B0 += DF;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0) Save1Bf<term, type>(dst, C, c00, bias, params, tail), C += dC, dst += dD;
                if (M > 1) Save1Bf<term, type>(dst, C, c10, bias, params, tail), C += dC, dst += dD;
                if (M > 2) Save1Bf<term, type>(dst, C, c20, bias, params, tail), C += dC, dst += dD;
                if (M > 3) Save1Bf<term, type>(dst, C, c30, bias, params, tail), C += dC, dst += dD;
                if (M > 4) Save1Bf<term, type>(dst, C, c40, bias, params, tail), C += dC, dst += dD;
                if (M > 5) Save1Bf<term, type>(dst, C, c50, bias, params, tail), C += dC, dst += dD;
                if (M > 6) Save1Bf<term, type>(dst, C, c60, bias, params, tail), C += dC, dst += dD;
                if (M > 7) Save1Bf<term, type>(dst, C, c70, bias, params, tail), C += dC, dst += dD;
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, int update, const uint16_t* B0, float* C, const __m512* bias, const __m512* params, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> GemmNN_2xM_Ptr GetGemmNN_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InnerProduct16bGemmNN_2xM<term, type, 1>;
            case 2: return InnerProduct16bGemmNN_2xM<term, type, 2>;
            case 3: return InnerProduct16bGemmNN_2xM<term, type, 3>;
            case 4: return InnerProduct16bGemmNN_2xM<term, type, 4>;
            case 5: return InnerProduct16bGemmNN_2xM<term, type, 5>;
            case 6: return InnerProduct16bGemmNN_2xM<term, type, 6>;
            case 7: return InnerProduct16bGemmNN_2xM<term, type, 7>;
            case 8: return InnerProduct16bGemmNN_2xM<term, type, 8>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C, int post, const float* bias, const float* params, float* sum, uint8_t* dst)
        {
            size_t m1 = M, m = 8;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.bK * DF, dC = a.cN, dD = p.N * a.eC;
            GemmNN_2xM_Ptr gemm_2xM = post ? GetGemmNN_2xM<term, type>(m) : GetGemmNN_2xM<Term16bInterim, type>(m);
            GemmNN_2xM_Ptr gemm_2xT = post ? GetGemmNN_2xM<term, type>(t) : GetGemmNN_2xM<Term16bInterim, type>(t);

            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bias[0] = _mm512_loadu_ps(bias + j + 0);
                _bias[1] = _mm512_loadu_ps(bias + j + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + j + 0);
                    _params[1] = _mm512_loadu_ps(params + j + F);
                }
                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, _params, dst + i * dD);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, _params, dst + i * dD);
                B += dB;
                C += dN;
                dst += DF * a.eC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetGemm(const InnerProductParam16b& p, GemmPtr& gemm)
        {
            if (p.typeC == SimdTensorData16b)
                gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast16b, type>;
            else
                gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast32f, type>;
        }

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Avx512bw::SynetInnerProduct16bGemmNN(p)
        {
            SetAlgParam(F, 8, F * 2, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData16b)
                    _prepA = InnerProduct16bGemmNN_ReorderA;
                else
                    _prepA = InnerProduct16bGemmNN_ConvertA;
            }
            else
                _prepA = NULL;
            if ((p.typeB == SimdTensorData32f || p.constB) && !p.transB)
                _prepB = InnerProduct16bGemmNN_ConvertBn;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationRelu: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationLeakyRelu: SetGemm<SimdConvolutionActivationPrelu>(p, _gemm); break;
            case SimdConvolutionActivationRestrictRange: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationPrelu: SetGemm<SimdConvolutionActivationPrelu>(p, _gemm); break;
            case SimdConvolutionActivationElu: SetGemm<SimdConvolutionActivationElu>(p, _gemm); break;
            case SimdConvolutionActivationHswish: SetGemm<SimdConvolutionActivationHswish>(p, _gemm); break;
            case SimdConvolutionActivationMish: SetGemm<SimdConvolutionActivationMish>(p, _gemm); break;
            case SimdConvolutionActivationHardSigmoid: SetGemm<SimdConvolutionActivationHardSigmoid>(p, _gemm); break;
            case SimdConvolutionActivationSwish: SetGemm<SimdConvolutionActivationSwish>(p, _gemm); break;
            case SimdConvolutionActivationGelu: SetGemm<SimdConvolutionActivationGelu>(p, _gemm); break;
            default: assert(0);
            }
        }
    }
#endif
}
