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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdSynetQuantizedActivation.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void StoreBody1(int32_t* ptr, __m128i val0)
        {
            _mm_storeu_si128((__m128i*)ptr + 0, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, __m128i val0, __m128i val1)
        {
            _mm_storeu_si128((__m128i*)ptr + 0, val0);
            _mm_storeu_si128((__m128i*)ptr + 1, val1);
        }

        template<int M> void DqipGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            __m128i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b0, b1;
            size_t dS = a.sN, dA = a.aK;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = _mm_loadu_si128((__m128i*)(sum + 0 * dS) + 0), c01 = _mm_loadu_si128((__m128i*)(sum + 0 * dS) + 1);
                    if (M > 1) c10 = _mm_loadu_si128((__m128i*)(sum + 1 * dS) + 0), c11 = _mm_loadu_si128((__m128i*)(sum + 1 * dS) + 1);
                    if (M > 2) c20 = _mm_loadu_si128((__m128i*)(sum + 2 * dS) + 0), c21 = _mm_loadu_si128((__m128i*)(sum + 2 * dS) + 1);
                    if (M > 3) c30 = _mm_loadu_si128((__m128i*)(sum + 3 * dS) + 0), c31 = _mm_loadu_si128((__m128i*)(sum + 3 * dS) + 1);
                    if (M > 4) c40 = _mm_loadu_si128((__m128i*)(sum + 4 * dS) + 0), c41 = _mm_loadu_si128((__m128i*)(sum + 4 * dS) + 1);
                }
                else
                {
                    if (M > 0) c00 = _mm_setzero_si128(), c01 = _mm_setzero_si128();
                    if (M > 1) c10 = _mm_setzero_si128(), c11 = _mm_setzero_si128();
                    if (M > 2) c20 = _mm_setzero_si128(), c21 = _mm_setzero_si128();
                    if (M > 3) c30 = _mm_setzero_si128(), c31 = _mm_setzero_si128();
                    if (M > 4) c40 = _mm_setzero_si128(), c41 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm_loadu_si128((__m128i*)B0);
                    b1 = _mm_loadu_si128((__m128i*)B1);
                    if (M > 0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0), Madd4<true>(c01, a0, b1);
                    if (M > 1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0), Madd4<true>(c11, a0, b1);
                    if (M > 2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0), Madd4<true>(c21, a0, b1);
                    if (M > 3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0), Madd4<true>(c31, a0, b1);
                    if (M > 4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0), Madd4<true>(c41, a0, b1);
                    B0 += A, B1 += A;
                }
                if (M > 0) StoreBody2(sum + 0 * dS, c00, c01);
                if (M > 1) StoreBody2(sum + 1 * dS, c10, c11);
                if (M > 2) StoreBody2(sum + 2 * dS, c20, c21);
                if (M > 3) StoreBody2(sum + 3 * dS, c30, c31);
                if (M > 4) StoreBody2(sum + 4 * dS, c40, c41);
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = _mm_loadu_si128((__m128i*)(sum + 0 * dS) + 0);
                    if (M > 1) c10 = _mm_loadu_si128((__m128i*)(sum + 1 * dS) + 0);
                    if (M > 2) c20 = _mm_loadu_si128((__m128i*)(sum + 2 * dS) + 0);
                    if (M > 3) c30 = _mm_loadu_si128((__m128i*)(sum + 3 * dS) + 0);
                    if (M > 4) c40 = _mm_loadu_si128((__m128i*)(sum + 4 * dS) + 0);
                }
                else
                {
                    if (M > 0) c00 = _mm_setzero_si128();
                    if (M > 1) c10 = _mm_setzero_si128();
                    if (M > 2) c20 = _mm_setzero_si128();
                    if (M > 3) c30 = _mm_setzero_si128();
                    if (M > 4) c40 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm_loadu_si128((__m128i*)B0);
                    if (M > 0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0);
                    if (M > 1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0);
                    if (M > 2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0);
                    if (M > 3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0);
                    if (M > 4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0);
                    B0 += A;
                }
                if (M > 0) StoreBody1(sum + 0 * dS, c00);
                if (M > 1) StoreBody1(sum + 1 * dS, c10);
                if (M > 2) StoreBody1(sum + 2 * dS, c20);
                if (M > 3) StoreBody1(sum + 3 * dS, c30);
                if (M > 4) StoreBody1(sum + 4 * dS, c40);
            }
        }

        typedef void(*DqipGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE DqipGemmV0_GemmBody_2xM_Ptr GetDqipGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return DqipGemmV0_GemmBody_2xM<1>;
            case 2: return DqipGemmV0_GemmBody_2xM<2>;
            case 3: return DqipGemmV0_GemmBody_2xM<3>;
            case 4: return DqipGemmV0_GemmBody_2xM<4>;
            case 5: return DqipGemmV0_GemmBody_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        static void DqipGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t*sum)
        {
            size_t n = 5;
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

        SynetDynamicQuantizedInnerProductGemmV0::SynetDynamicQuantizedInnerProductGemmV0(const DynamicQuantizedInnerProductParam& p)
            : Base::SynetDynamicQuantizedInnerProductGemmV0(p)
        {
            SetAlgParam(F, 5, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _gemmBody = DqipGemmV0_GemmBody;
        }
    }
#endif
}
