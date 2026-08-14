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
#include "Simd/SimdSynetQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmV0::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmV0::GemmPtr GemmPtr;

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmV0_PrepA_8u(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
        {
            const size_t A = svcntb();
            size_t KA = Simd::AlignLo(p.K, A);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KA; k += A)
                    Copy(src + k, dst + k);
                for (; k < p.K; ++k)
                    dst[k] = src[k];
                for (; k < a.aK; ++k)
                    dst[k] = 0;
                src += p.K;
                dst += a.aK;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedInnerProductGemmV0_2xM(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, const svint32_t& bias0, const svint32_t& bias1,
            const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, int32_t* buf, uint8_t* C)
        {
            const size_t F = svcntw(), A = F * 4, DF = F * 2;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            svuint8_t a0;
            svint8_t b0, b1;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            const int8_t* B1 = B0 + a.bK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            const uint8_t* A5 = A0 + 5 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, buf + 0x0 * dB + 0), c01 = svld1_s32(body32, buf + 0x0 * dB + F);
                    if (M > 0x1) c10 = svld1_s32(body32, buf + 0x1 * dB + 0), c11 = svld1_s32(body32, buf + 0x1 * dB + F);
                    if (M > 0x2) c20 = svld1_s32(body32, buf + 0x2 * dB + 0), c21 = svld1_s32(body32, buf + 0x2 * dB + F);
                    if (M > 0x3) c30 = svld1_s32(body32, buf + 0x3 * dB + 0), c31 = svld1_s32(body32, buf + 0x3 * dB + F);
                    if (M > 0x4) c40 = svld1_s32(body32, buf + 0x4 * dB + 0), c41 = svld1_s32(body32, buf + 0x4 * dB + F);
                    if (M > 0x5) c50 = svld1_s32(body32, buf + 0x5 * dB + 0), c51 = svld1_s32(body32, buf + 0x5 * dB + F);
                    if (M > 0x6) c60 = svld1_s32(body32, buf + 0x6 * dB + 0), c61 = svld1_s32(body32, buf + 0x6 * dB + F);
                    if (M > 0x7) c70 = svld1_s32(body32, buf + 0x7 * dB + 0), c71 = svld1_s32(body32, buf + 0x7 * dB + F);
                    if (M > 0x8) c80 = svld1_s32(body32, buf + 0x8 * dB + 0), c81 = svld1_s32(body32, buf + 0x8 * dB + F);
                    if (M > 0x9) c90 = svld1_s32(body32, buf + 0x9 * dB + 0), c91 = svld1_s32(body32, buf + 0x9 * dB + F);
                    if (M > 0xA) cA0 = svld1_s32(body32, buf + 0xA * dB + 0), cA1 = svld1_s32(body32, buf + 0xA * dB + F);
                    if (M > 0xB) cB0 = svld1_s32(body32, buf + 0xB * dB + 0), cB1 = svld1_s32(body32, buf + 0xB * dB + F);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0), c01 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0), c11 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0), c21 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0), c31 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0), c41 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0), c51 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0), c61 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0), c71 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0), c81 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0), c91 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0), cA1 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0), cB1 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    b1 = svld1_s8(body8, B1);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0), Madd4<false>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0), Madd4<false>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0), Madd4<false>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0), Madd4<false>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0), Madd4<false>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0), Madd4<false>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0), Madd4<false>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0), Madd4<false>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0), Madd4<false>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0), Madd4<false>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0), Madd4<false>(cA1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0), Madd4<false>(cB1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (N == DF)
                {
                    if (M > 0x0) Save2<term>(C, buf, c00, c01, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x1) Save2<term>(C, buf, c10, c11, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x2) Save2<term>(C, buf, c20, c21, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x3) Save2<term>(C, buf, c30, c31, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x4) Save2<term>(C, buf, c40, c41, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x5) Save2<term>(C, buf, c50, c51, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x6) Save2<term>(C, buf, c60, c61, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x7) Save2<term>(C, buf, c70, c71, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x8) Save2<term>(C, buf, c80, c81, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0x9) Save2<term>(C, buf, c90, c91, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0xA) Save2<term>(C, buf, cA0, cA1, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                    if (M > 0xB) Save2<term>(C, buf, cB0, cB1, bias0, bias1, norm0, norm1, zero), C += dC, buf += dB;
                }
                else
                {
                    N -= F;
                    if (M > 0x0) Save2<term>(C, buf, c00, c01, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x1) Save2<term>(C, buf, c10, c11, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x2) Save2<term>(C, buf, c20, c21, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x3) Save2<term>(C, buf, c30, c31, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x4) Save2<term>(C, buf, c40, c41, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x5) Save2<term>(C, buf, c50, c51, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x6) Save2<term>(C, buf, c60, c61, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x7) Save2<term>(C, buf, c70, c71, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x8) Save2<term>(C, buf, c80, c81, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0x9) Save2<term>(C, buf, c90, c91, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0xA) Save2<term>(C, buf, cA0, cA1, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                    if (M > 0xB) Save2<term>(C, buf, cB0, cB1, bias0, bias1, norm0, norm1, zero, N), C += dC, buf += dB;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, buf + 0x0 * dB);
                    if (M > 0x1) c10 = svld1_s32(body32, buf + 0x1 * dB);
                    if (M > 0x2) c20 = svld1_s32(body32, buf + 0x2 * dB);
                    if (M > 0x3) c30 = svld1_s32(body32, buf + 0x3 * dB);
                    if (M > 0x4) c40 = svld1_s32(body32, buf + 0x4 * dB);
                    if (M > 0x5) c50 = svld1_s32(body32, buf + 0x5 * dB);
                    if (M > 0x6) c60 = svld1_s32(body32, buf + 0x6 * dB);
                    if (M > 0x7) c70 = svld1_s32(body32, buf + 0x7 * dB);
                    if (M > 0x8) c80 = svld1_s32(body32, buf + 0x8 * dB);
                    if (M > 0x9) c90 = svld1_s32(body32, buf + 0x9 * dB);
                    if (M > 0xA) cA0 = svld1_s32(body32, buf + 0xA * dB);
                    if (M > 0xB) cB0 = svld1_s32(body32, buf + 0xB * dB);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0);
                    B0 += A;
                }
                if (N == F)
                {
                    if (M > 0x0) Save1<term>(C, buf, c00, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x1) Save1<term>(C, buf, c10, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x2) Save1<term>(C, buf, c20, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x3) Save1<term>(C, buf, c30, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x4) Save1<term>(C, buf, c40, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x5) Save1<term>(C, buf, c50, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x6) Save1<term>(C, buf, c60, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x7) Save1<term>(C, buf, c70, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x8) Save1<term>(C, buf, c80, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0x9) Save1<term>(C, buf, c90, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0xA) Save1<term>(C, buf, cA0, bias0, norm0, zero), C += dC, buf += dB;
                    if (M > 0xB) Save1<term>(C, buf, cB0, bias0, norm0, zero), C += dC, buf += dB;
                }
                else
                {
                    if (M > 0x0) Save1<term>(C, buf, c00, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x1) Save1<term>(C, buf, c10, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x2) Save1<term>(C, buf, c20, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x3) Save1<term>(C, buf, c30, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x4) Save1<term>(C, buf, c40, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x5) Save1<term>(C, buf, c50, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x6) Save1<term>(C, buf, c60, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x7) Save1<term>(C, buf, c70, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x8) Save1<term>(C, buf, c80, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0x9) Save1<term>(C, buf, c90, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0xA) Save1<term>(C, buf, cA0, bias0, norm0, zero, N), C += dC, buf += dB;
                    if (M > 0xB) Save1<term>(C, buf, cB0, bias0, norm0, zero, N), C += dC, buf += dB;
                }
            }
        }

        typedef void(*QuantizedInnerProductGemmV0_2xM_Ptr)(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, const svint32_t& bias0, const svint32_t& bias1,
            const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, int32_t* buf, uint8_t* C);

        template<Term8iType term> QuantizedInnerProductGemmV0_2xM_Ptr GetQuantizedInnerProductGemmV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedInnerProductGemmV0_2xM<term, 0x1>;
            case 0x2: return QuantizedInnerProductGemmV0_2xM<term, 0x2>;
            case 0x3: return QuantizedInnerProductGemmV0_2xM<term, 0x3>;
            case 0x4: return QuantizedInnerProductGemmV0_2xM<term, 0x4>;
            case 0x5: return QuantizedInnerProductGemmV0_2xM<term, 0x5>;
            case 0x6: return QuantizedInnerProductGemmV0_2xM<term, 0x6>;
            case 0x7: return QuantizedInnerProductGemmV0_2xM<term, 0x7>;
            case 0x8: return QuantizedInnerProductGemmV0_2xM<term, 0x8>;
            case 0x9: return QuantizedInnerProductGemmV0_2xM<term, 0x9>;
            case 0xA: return QuantizedInnerProductGemmV0_2xM<term, 0xA>;
            case 0xB: return QuantizedInnerProductGemmV0_2xM<term, 0xB>;
            case 0xC: return QuantizedInnerProductGemmV0_2xM<term, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedInnerProductGemmV0_2(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, 
            int update, const int8_t* B, int32_t* buf, int post, const int32_t* bias, const float* norm, uint32_t zero, uint8_t* C)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            QuantizedInnerProductGemmV0_2xM_Ptr gemm_2xN = post ? GetQuantizedInnerProductGemmV0_2xM<term>(n) : GetQuantizedInnerProductGemmV0_2xM<Term8iInterim>(n);
            QuantizedInnerProductGemmV0_2xM_Ptr gemm_2xM = post ? GetQuantizedInnerProductGemmV0_2xM<term>(m) : GetQuantizedInnerProductGemmV0_2xM<Term8iInterim>(m);

            svint32_t _zero = svdup_n_s32(zero);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                svint32_t _bias0 = svld1_s32(body, bias + j + 0);
                svint32_t _bias1 = svld1_s32(body, bias + j + F);
                svfloat32_t _norm0 = svld1_f32(body, norm + j + 0);
                svfloat32_t _norm1 = svld1_f32(body, norm + j + F);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, _bias0, _bias1, _norm0, _norm1, _zero, buf + i * dB, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, _bias0, _bias1, _norm0, _norm1, _zero, buf + i * dB, C + i * dC);
                B += a.bK * DF;
                buf += DF;
                C += DF * a.eC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmV0::SynetQuantizedInnerProductGemmV0(const QuantizedInnerProductParam& p)
            : Base::SynetQuantizedInnerProductGemmV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData8u)
                    _prepA = QuantizedInnerProductGemmV0_PrepA_8u;
                else
                    _prepA = NULL;
            }
            if (p.typeC == SimdTensorData8u)
                _gemm = QuantizedInnerProductGemmV0_2<Term8iLast8u>;
            else
                _gemm = NULL;
        }
    }
#endif
}
