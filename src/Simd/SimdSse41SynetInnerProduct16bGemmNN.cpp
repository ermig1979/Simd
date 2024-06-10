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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdSynetInnerProduct16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sse41
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        template<int M> void InnerProduct16bGemmNN_2xM(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, 
            size_t N, size_t K, int update, const uint16_t* B0, float* C)
        {
            __m128 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b00, b01, b10, b11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dC = a.aN, dA = a.aK;
            const uint16_t* B1 = B0 + a.aK * F;
            const uint16_t* A1 = A0 + 1 * dA;
            const uint16_t* A2 = A0 + 2 * dA;
            const uint16_t* A3 = A0 + 3 * dA;
            const uint16_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = _mm_loadu_ps(C + 0 * dC + 0), c01 = _mm_loadu_ps(C + 0 * dC + F);
                    if (M > 1) c10 = _mm_loadu_ps(C + 1 * dC + 0), c11 = _mm_loadu_ps(C + 1 * dC + F);
                    if (M > 2) c20 = _mm_loadu_ps(C + 2 * dC + 0), c21 = _mm_loadu_ps(C + 2 * dC + F);
                    if (M > 3) c30 = _mm_loadu_ps(C + 3 * dC + 0), c31 = _mm_loadu_ps(C + 3 * dC + F);
                    if (M > 4) c40 = _mm_loadu_ps(C + 4 * dC + 0), c41 = _mm_loadu_ps(C + 4 * dC + F);
                }                
                else
                {
                    if (M > 0) c00 = _mm_setzero_ps(), c01 = _mm_setzero_ps();
                    if (M > 1) c10 = _mm_setzero_ps(), c11 = _mm_setzero_ps();
                    if (M > 2) c20 = _mm_setzero_ps(), c21 = _mm_setzero_ps();
                    if (M > 3) c30 = _mm_setzero_ps(), c31 = _mm_setzero_ps();
                    if (M > 4) c40 = _mm_setzero_ps(), c41 = _mm_setzero_ps();
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b01 = _mm_loadu_ps((float*)B0);
                    b00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(b01), Base::Bf16::SHIFT));
                    b01 = _mm_and_ps(b01, m);
                    b11 = _mm_loadu_ps((float*)B1);
                    b10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(b11), Base::Bf16::SHIFT));
                    b11 = _mm_and_ps(b11, m);
                    if (M > 0)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A0 + k - 1)), m);
                        c00 = _mm_add_ps(_mm_mul_ps(a0, b00), c00);
                        c01 = _mm_add_ps(_mm_mul_ps(a0, b10), c01);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A0 + k - 0)), m);
                        c00 = _mm_add_ps(_mm_mul_ps(a0, b01), c00);
                        c01 = _mm_add_ps(_mm_mul_ps(a0, b11), c01);
                    }
                    if (M > 1)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A1 + k - 1)), m);
                        c10 = _mm_add_ps(_mm_mul_ps(a0, b00), c10);
                        c11 = _mm_add_ps(_mm_mul_ps(a0, b10), c11);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A1 + k - 0)), m);
                        c10 = _mm_add_ps(_mm_mul_ps(a0, b01), c10);
                        c11 = _mm_add_ps(_mm_mul_ps(a0, b11), c11);
                    }
                    if (M > 2)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A2 + k - 1)), m);
                        c20 = _mm_add_ps(_mm_mul_ps(a0, b00), c20);
                        c21 = _mm_add_ps(_mm_mul_ps(a0, b10), c21);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A2 + k - 0)), m);
                        c20 = _mm_add_ps(_mm_mul_ps(a0, b01), c20);
                        c21 = _mm_add_ps(_mm_mul_ps(a0, b11), c21);
                    }
                    if (M > 3)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A3 + k - 1)), m);
                        c30 = _mm_add_ps(_mm_mul_ps(a0, b00), c30);
                        c31 = _mm_add_ps(_mm_mul_ps(a0, b10), c31);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A3 + k - 0)), m);
                        c30 = _mm_add_ps(_mm_mul_ps(a0, b01), c30);
                        c31 = _mm_add_ps(_mm_mul_ps(a0, b11), c31);
                    }
                    if (M > 4)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A4 + k - 1)), m);
                        c40 = _mm_add_ps(_mm_mul_ps(a0, b00), c40);
                        c41 = _mm_add_ps(_mm_mul_ps(a0, b10), c41);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A4 + k - 0)), m);
                        c40 = _mm_add_ps(_mm_mul_ps(a0, b01), c40);
                        c41 = _mm_add_ps(_mm_mul_ps(a0, b11), c41);
                    }
                    B0 += DF;
                    B1 += DF;
                }
                if (N == DF)
                {
                    if (M > 0) Save2(C, c00, c01), C += dC;
                    if (M > 1) Save2(C, c10, c11), C += dC;
                    if (M > 2) Save2(C, c20, c21), C += dC;
                    if (M > 3) Save2(C, c30, c31), C += dC;
                    if (M > 4) Save2(C, c40, c41), C += dC;
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0) Save2(C, c00, c01, tail), C += dC;
                    if (M > 1) Save2(C, c10, c11, tail), C += dC;
                    if (M > 2) Save2(C, c20, c21, tail), C += dC;
                    if (M > 3) Save2(C, c30, c31, tail), C += dC;
                    if (M > 4) Save2(C, c40, c41, tail), C += dC;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = _mm_loadu_ps(C + 0 * dC + 0);
                    if (M > 1) c10 = _mm_loadu_ps(C + 1 * dC + 0);
                    if (M > 2) c20 = _mm_loadu_ps(C + 2 * dC + 0);
                    if (M > 3) c30 = _mm_loadu_ps(C + 3 * dC + 0);
                    if (M > 4) c40 = _mm_loadu_ps(C + 4 * dC + 0);
                }
                else
                {
                    if (M > 0) c00 = _mm_setzero_ps();
                    if (M > 1) c10 = _mm_setzero_ps();
                    if (M > 2) c20 = _mm_setzero_ps();
                    if (M > 3) c30 = _mm_setzero_ps();
                    if (M > 4) c40 = _mm_setzero_ps();
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b01 = _mm_loadu_ps((float*)B0);
                    b00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(b01), Base::Bf16::SHIFT));
                    b01 = _mm_and_ps(b01, m);
                    if (M > 0)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A0 + k - 1)), m);
                        c00 = _mm_add_ps(_mm_mul_ps(a0, b00), c00);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A0 + k - 0)), m);
                        c00 = _mm_add_ps(_mm_mul_ps(a0, b01), c00);
                    }
                    if (M > 1)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A1 + k - 1)), m);
                        c10 = _mm_add_ps(_mm_mul_ps(a0, b00), c10);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A1 + k - 0)), m);
                        c10 = _mm_add_ps(_mm_mul_ps(a0, b01), c10);
                    }
                    if (M > 2)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A2 + k - 1)), m);
                        c20 = _mm_add_ps(_mm_mul_ps(a0, b00), c20);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A2 + k - 0)), m);
                        c20 = _mm_add_ps(_mm_mul_ps(a0, b01), c20);
                    }
                    if (M > 3)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A3 + k - 1)), m);
                        c30 = _mm_add_ps(_mm_mul_ps(a0, b00), c30);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A3 + k - 0)), m);
                        c30 = _mm_add_ps(_mm_mul_ps(a0, b01), c30);
                    }
                    if (M > 4)
                    {
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A4 + k - 1)), m);
                        c40 = _mm_add_ps(_mm_mul_ps(a0, b00), c40);
                        a0 = _mm_and_ps(_mm_set1_ps(*(float*)(A4 + k - 0)), m);
                        c40 = _mm_add_ps(_mm_mul_ps(a0, b01), c40);
                    }
                    B0 += DF;
                }
                if (N == F)
                {
                    if (M > 0) Save1(C, c00), C += dC;
                    if (M > 1) Save1(C, c10), C += dC;
                    if (M > 2) Save1(C, c20), C += dC;
                    if (M > 3) Save1(C, c30), C += dC;
                    if (M > 4) Save1(C, c40), C += dC;
                }
                else
                {
                    size_t tail = N;
                    if (M > 0) Save1(C, c00, tail), C += dC;
                    if (M > 1) Save1(C, c10, tail), C += dC;
                    if (M > 2) Save1(C, c20, tail), C += dC;
                    if (M > 3) Save1(C, c30, tail), C += dC;
                    if (M > 4) Save1(C, c40, tail), C += dC;
                }
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, int update, const uint16_t* B0, float* C);

        static GemmNN_2xM_Ptr GetInnerProduct16bGemmNN_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InnerProduct16bGemmNN_2xM<1>;
            case 2: return InnerProduct16bGemmNN_2xM<2>;
            case 3: return InnerProduct16bGemmNN_2xM<3>;
            case 4: return InnerProduct16bGemmNN_2xM<4>;
            case 5: return InnerProduct16bGemmNN_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        static void InnerProduct16bGemmNN_2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C)
        {
            size_t m1 = M, m = 5;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.aK * DF, dC = p.N;
            GemmNN_2xM_Ptr gemm_2xM = GetInnerProduct16bGemmNN_2xM(m);
            GemmNN_2xM_Ptr gemm_2xT = GetInnerProduct16bGemmNN_2xM(t);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC);
                B += dB;
                C += dN;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Base::SynetInnerProduct16bGemmNN(p)
        {
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _gemm = InnerProduct16bGemmNN_2;
        }
    }
#endif
}
