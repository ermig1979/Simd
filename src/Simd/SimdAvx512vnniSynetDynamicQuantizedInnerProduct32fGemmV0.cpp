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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct32f.h"
#include "Simd/SimdSynetDynamicQuantizedActivation.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512vnni
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct32fGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct32fGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void StoreBody1(int32_t* ptr, __m512i val0)
        {
            _mm512_storeu_epi32(ptr, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, __m512i val0, __m512i val1)
        {
            _mm512_storeu_si512((__m512i*)ptr + 0, val0);
            _mm512_storeu_si512((__m512i*)ptr + 1, val1);
        }

        template<int M> void DqipGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            __m512i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b0, b1;
            size_t dS = a.sN, dA = a.aK;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            const uint8_t* A5 = A0 + 5 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0), c01 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 1);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0), c11 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 1);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0), c21 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 1);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0), c31 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 1);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0), c41 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 1);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0), c51 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 1);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0), c61 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 1);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0), c71 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 1);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0), c81 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 1);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0), c91 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 1);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0), ca1 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 1);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0), cb1 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 1);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512(), c81 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512(), c91 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512(), ca1 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512(), cb1 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
                    b1 = _mm512_loadu_si512((__m512i*)B1);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0), Madd4<false>(ca1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0), Madd4<false>(cb1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (M > 0x0) StoreBody2(sum + 0x0 * dS, c00, c01);
                if (M > 0x1) StoreBody2(sum + 0x1 * dS, c10, c11);
                if (M > 0x2) StoreBody2(sum + 0x2 * dS, c20, c21);
                if (M > 0x3) StoreBody2(sum + 0x3 * dS, c30, c31);
                if (M > 0x4) StoreBody2(sum + 0x4 * dS, c40, c41);
                if (M > 0x5) StoreBody2(sum + 0x5 * dS, c50, c51);
                if (M > 0x6) StoreBody2(sum + 0x6 * dS, c60, c61);
                if (M > 0x7) StoreBody2(sum + 0x7 * dS, c70, c71);
                if (M > 0x8) StoreBody2(sum + 0x8 * dS, c80, c81);
                if (M > 0x9) StoreBody2(sum + 0x9 * dS, c90, c91);
                if (M > 0xA) StoreBody2(sum + 0xA * dS, ca0, ca1);
                if (M > 0xB) StoreBody2(sum + 0xB * dS, cb0, cb1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0);
                    B0 += A;
                }
                if (M > 0x0) StoreBody1(sum + 0x0 * dS, c00);
                if (M > 0x1) StoreBody1(sum + 0x1 * dS, c10);
                if (M > 0x2) StoreBody1(sum + 0x2 * dS, c20);
                if (M > 0x3) StoreBody1(sum + 0x3 * dS, c30);
                if (M > 0x4) StoreBody1(sum + 0x4 * dS, c40);
                if (M > 0x5) StoreBody1(sum + 0x5 * dS, c50);
                if (M > 0x6) StoreBody1(sum + 0x6 * dS, c60);
                if (M > 0x7) StoreBody1(sum + 0x7 * dS, c70);
                if (M > 0x8) StoreBody1(sum + 0x8 * dS, c80);
                if (M > 0x9) StoreBody1(sum + 0x9 * dS, c90);
                if (M > 0xA) StoreBody1(sum + 0xA * dS, ca0);
                if (M > 0xB) StoreBody1(sum + 0xB * dS, cb0);
            }
        }

        typedef void(*DqipGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE DqipGemmV0_GemmBody_2xM_Ptr GetDqipGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return DqipGemmV0_GemmBody_2xM<1>;
            case 0x2: return DqipGemmV0_GemmBody_2xM<2>;
            case 0x3: return DqipGemmV0_GemmBody_2xM<3>;
            case 0x4: return DqipGemmV0_GemmBody_2xM<4>;
            case 0x5: return DqipGemmV0_GemmBody_2xM<5>;
            case 0x6: return DqipGemmV0_GemmBody_2xM<6>;
            case 0x7: return DqipGemmV0_GemmBody_2xM<7>;
            case 0x8: return DqipGemmV0_GemmBody_2xM<8>;
            case 0x9: return DqipGemmV0_GemmBody_2xM<9>;
            case 0xA: return DqipGemmV0_GemmBody_2xM<10>;
            case 0xB: return DqipGemmV0_GemmBody_2xM<11>;
            case 0xC: return DqipGemmV0_GemmBody_2xM<12>;
            }
            assert(0);
            return NULL;
        }

        static void DqipGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.sN, dA = a.aK;
            DqipGemmV0_GemmBody_2xM_Ptr gemm_2xN = GetDqipGemmV0_GemmBody_2xM(n);
            DqipGemmV0_GemmBody_2xM_Ptr gemm_2xM = GetDqipGemmV0_GemmBody_2xM(m);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                B += a.aK * DF;
                sum += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void DqipGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, float* C)
        {
            __m512i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b0, b1;
            size_t dS = a.sN, dA = a.aK, dC = p.N;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            const uint8_t* A5 = A0 + 5 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0), c01 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 1);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0), c11 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 1);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0), c21 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 1);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0), c31 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 1);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0), c41 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 1);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0), c51 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 1);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0), c61 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 1);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0), c71 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 1);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0), c81 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 1);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0), c91 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 1);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0), ca1 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 1);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0), cb1 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 1);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512(), c81 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512(), c91 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512(), ca1 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512(), cb1 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
                    b1 = _mm512_loadu_si512((__m512i*)B1);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0), Madd4<false>(ca1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0), Madd4<false>(cb1, a0, b1);
                    B0 += A, B1 += A;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0x0) Save2<type>(C + 0x0 * dC, c00, c01, corr, norm, bias, params, tail);
                if (M > 0x1) Save2<type>(C + 0x1 * dC, c10, c11, corr, norm, bias, params, tail);
                if (M > 0x2) Save2<type>(C + 0x2 * dC, c20, c21, corr, norm, bias, params, tail);
                if (M > 0x3) Save2<type>(C + 0x3 * dC, c30, c31, corr, norm, bias, params, tail);
                if (M > 0x4) Save2<type>(C + 0x4 * dC, c40, c41, corr, norm, bias, params, tail);
                if (M > 0x5) Save2<type>(C + 0x5 * dC, c50, c51, corr, norm, bias, params, tail);
                if (M > 0x6) Save2<type>(C + 0x6 * dC, c60, c61, corr, norm, bias, params, tail);
                if (M > 0x7) Save2<type>(C + 0x7 * dC, c70, c71, corr, norm, bias, params, tail);
                if (M > 0x8) Save2<type>(C + 0x8 * dC, c80, c81, corr, norm, bias, params, tail);
                if (M > 0x9) Save2<type>(C + 0x9 * dC, c90, c91, corr, norm, bias, params, tail);
                if (M > 0xA) Save2<type>(C + 0xA * dC, ca0, ca1, corr, norm, bias, params, tail);
                if (M > 0xB) Save2<type>(C + 0xB * dC, cb0, cb1, corr, norm, bias, params, tail);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0);
                    B0 += A;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0x0) Save1<type>(C + 0x0 * dC, c00, corr, norm, bias, params, tail);
                if (M > 0x1) Save1<type>(C + 0x1 * dC, c10, corr, norm, bias, params, tail);
                if (M > 0x2) Save1<type>(C + 0x2 * dC, c20, corr, norm, bias, params, tail);
                if (M > 0x3) Save1<type>(C + 0x3 * dC, c30, corr, norm, bias, params, tail);
                if (M > 0x4) Save1<type>(C + 0x4 * dC, c40, corr, norm, bias, params, tail);
                if (M > 0x5) Save1<type>(C + 0x5 * dC, c50, corr, norm, bias, params, tail);
                if (M > 0x6) Save1<type>(C + 0x6 * dC, c60, corr, norm, bias, params, tail);
                if (M > 0x7) Save1<type>(C + 0x7 * dC, c70, corr, norm, bias, params, tail);
                if (M > 0x8) Save1<type>(C + 0x8 * dC, c80, corr, norm, bias, params, tail);
                if (M > 0x9) Save1<type>(C + 0x9 * dC, c90, corr, norm, bias, params, tail);
                if (M > 0xA) Save1<type>(C + 0xA * dC, ca0, corr, norm, bias, params, tail);
                if (M > 0xB) Save1<type>(C + 0xB * dC, cb0, corr, norm, bias, params, tail);
            }
        }

        typedef void(*DqipGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const __m512i* corr, const __m512* norm, const __m512* bias, const __m512* params, float* C);

        template<SimdConvolutionActivationType type> SIMD_INLINE DqipGemmV0_GemmLast_2xM_Ptr GetDqipGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return DqipGemmV0_GemmLast_2xM<type, 1>;
            case 0x2: return DqipGemmV0_GemmLast_2xM<type, 2>;
            case 0x3: return DqipGemmV0_GemmLast_2xM<type, 3>;
            case 0x4: return DqipGemmV0_GemmLast_2xM<type, 4>;
            case 0x5: return DqipGemmV0_GemmLast_2xM<type, 5>;
            case 0x6: return DqipGemmV0_GemmLast_2xM<type, 6>;
            case 0x7: return DqipGemmV0_GemmLast_2xM<type, 7>;
            case 0x8: return DqipGemmV0_GemmLast_2xM<type, 8>;
            case 0x9: return DqipGemmV0_GemmLast_2xM<type, 9>;
            case 0xA: return DqipGemmV0_GemmLast_2xM<type, 10>;
            case 0xB: return DqipGemmV0_GemmLast_2xM<type, 11>;
            case 0xC: return DqipGemmV0_GemmLast_2xM<type, 12>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void DqipGemmV0_GemmLast(const uint8_t* A, const DynamicQuantizedInnerProductParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* corr, const float* norm, const float* bias, const float* params, float* C)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.sN, dA = a.aK, dC = p.N;
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqipGemmV0_GemmLast_2xM<type>(n);
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqipGemmV0_GemmLast_2xM<type>(m);

            __m512i _corr[2];
            __m512 _norm[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _corr[0] = _mm512_loadu_si512((__m512i*)(corr + j) + 0);
                _corr[1] = _mm512_loadu_si512((__m512i*)(corr + j) + 1);
                _norm[0] = _mm512_loadu_ps(norm + j + 0);
                _norm[1] = _mm512_loadu_ps(norm + j + F);
                _bias[0] = _mm512_loadu_ps(bias + j + 0);
                _bias[1] = _mm512_loadu_ps(bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + j + 0);
                    _params[1] = _mm512_loadu_ps(params + j + F);
                }
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, _corr, _norm, _bias, _params, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, _corr, _norm, _bias, _params, C + i * dC);
                B += a.aK * DF;
                sum += DF;
                C += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct32fGemmV0::SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p)
            : Avx512bw::SynetDynamicQuantizedInnerProduct32fGemmV0(p)
        {
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _gemmBody = DqipGemmV0_GemmBody;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: _gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationGelu>; break;
            default:
                assert(0);
            }
        }
    }
#endif
}
