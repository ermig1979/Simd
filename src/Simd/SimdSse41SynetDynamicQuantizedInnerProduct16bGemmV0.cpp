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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"
#include "Simd/SimdSynetDynamicQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdMax.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct16bGemmV0_ConvertAf(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const float* src = (float*)src8;
            size_t K = p.K, K4 = AlignLo(K, 4), K16 = AlignLo(K, 16), KA = a.aK, k;
            const int qmin = 0, qmax = 255;
            for (size_t i = 0; i < M; ++i)
            {
                __m128 _min = _mm_set1_ps(FLT_MAX);
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                for (k = 0; k < K4; k += 4)
                {
                    __m128 _src = _mm_loadu_ps(src + k);
                    _min = _mm_min_ps(_src, _min);
                    _max = _mm_max_ps(_src, _max);
                }
                _min = MinVal32f(_min);
                _max = MaxVal32f(_max);
                for (; k < K; k += 1)
                {
                    __m128 _src = _mm_load_ss(src + k);
                    _min = _mm_min_ss(_src, _min);
                    _max = _mm_max_ss(_src, _max);
                }
                _min = _mm_min_ss(_min, _mm_setzero_ps());
                _max = _mm_max_ss(_max, _mm_setzero_ps());
                float min, max;
                _mm_store_ss(&min, _min);
                _mm_store_ss(&max, _max);
                scale[i] = max == min ? 1.0f : (max - min) / float(qmax - qmin);
                float initialZeroPoint = qmin - min / scale[i];
                zero[i] = Base::NearByInt(Simd::Max(float(qmin), Simd::Min(float(qmax), initialZeroPoint)));
                __m128 _norm = _mm_set1_ps(1.0f / scale[i]);
                __m128i _zero = _mm_set1_epi32(zero[i]);
                for (k = 0; k < K16; k += 16)
                    QuantizeLinear16(src + k, _norm, _zero, dst + k);
                for (; k < K4; k += 4)
                    QuantizeLinear4(src + k, _norm, _zero, dst + k);
                for (; k < K; k += 1)
                    QuantizeLinear1(src + k, _norm, _zero, dst + k);
                for (; k < KA; ++k)
                    dst[k] = 0;
                src += K;
                dst += KA;
            }
        }

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

        template<int M> void Dqip16bGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            __m128i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b0, b1;
            size_t dS = a.cN, dA = a.aK;
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

        typedef void(*Dqip16bGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE Dqip16bGemmV0_GemmBody_2xM_Ptr GetDqip16bGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Dqip16bGemmV0_GemmBody_2xM<1>;
            case 2: return Dqip16bGemmV0_GemmBody_2xM<2>;
            case 3: return Dqip16bGemmV0_GemmBody_2xM<3>;
            case 4: return Dqip16bGemmV0_GemmBody_2xM<4>;
            case 5: return Dqip16bGemmV0_GemmBody_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        static void Dqip16bGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            size_t n = 5;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK;
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmBody_2xM(n);
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmBody_2xM(m);

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

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Dqip16bGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const __m128i* bSums, const float* aScale, const __m128* bScale, const __m128* bias, const __m128* params, uint8_t* C)
        {
            __m128i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b0, b1;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
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
                if (N == DF)
                {
                    if (M > 0) Save2<term, type>(C + 0 * dC, c00, c01, aZero[0], bSums, aScale[0], bScale, bias, params);
                    if (M > 1) Save2<term, type>(C + 1 * dC, c10, c11, aZero[1], bSums, aScale[1], bScale, bias, params);
                    if (M > 2) Save2<term, type>(C + 2 * dC, c20, c21, aZero[2], bSums, aScale[2], bScale, bias, params);
                    if (M > 3) Save2<term, type>(C + 3 * dC, c30, c31, aZero[3], bSums, aScale[3], bScale, bias, params);
                    if (M > 4) Save2<term, type>(C + 4 * dC, c40, c41, aZero[4], bSums, aScale[4], bScale, bias, params);
                }
                else
                {
                    if (M > 0) Save2<term, type>(C + 0 * dC, c00, c01, aZero[0], bSums, aScale[0], bScale, bias, params, N);
                    if (M > 1) Save2<term, type>(C + 1 * dC, c10, c11, aZero[1], bSums, aScale[1], bScale, bias, params, N);
                    if (M > 2) Save2<term, type>(C + 2 * dC, c20, c21, aZero[2], bSums, aScale[2], bScale, bias, params, N);
                    if (M > 3) Save2<term, type>(C + 3 * dC, c30, c31, aZero[3], bSums, aScale[3], bScale, bias, params, N);
                    if (M > 4) Save2<term, type>(C + 4 * dC, c40, c41, aZero[4], bSums, aScale[4], bScale, bias, params, N);
                }
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
                if (N == F)
                {
                    if (M > 0) Save1<term, type>(C + 0 * dC, c00, aZero[0], bSums, aScale[0], bScale, bias, params);
                    if (M > 1) Save1<term, type>(C + 1 * dC, c10, aZero[1], bSums, aScale[1], bScale, bias, params);
                    if (M > 2) Save1<term, type>(C + 2 * dC, c20, aZero[2], bSums, aScale[2], bScale, bias, params);
                    if (M > 3) Save1<term, type>(C + 3 * dC, c30, aZero[3], bSums, aScale[3], bScale, bias, params);
                    if (M > 4) Save1<term, type>(C + 4 * dC, c40, aZero[4], bSums, aScale[4], bScale, bias, params);
                }
                else
                {
                    if (M > 0) Save1<term, type>(C + 0 * dC, c00, aZero[0], bSums, aScale[0], bScale, bias, params, N);
                    if (M > 1) Save1<term, type>(C + 1 * dC, c10, aZero[1], bSums, aScale[1], bScale, bias, params, N);
                    if (M > 2) Save1<term, type>(C + 2 * dC, c20, aZero[2], bSums, aScale[2], bScale, bias, params, N);
                    if (M > 3) Save1<term, type>(C + 3 * dC, c30, aZero[3], bSums, aScale[3], bScale, bias, params, N);
                    if (M > 4) Save1<term, type>(C + 4 * dC, c40, aZero[4], bSums, aScale[4], bScale, bias, params, N);
                }
            }
        }

        typedef void(*Dqip16bGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, 
            int32_t* buf, const int32_t* aZero, const __m128i* bSums, const float * aScale, const __m128* bScale, const __m128* bias, const __m128* params, uint8_t* C);

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE Dqip16bGemmV0_GemmLast_2xM_Ptr GetDqip16bGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Dqip16bGemmV0_GemmLast_2xM<term, type, 1>;
            case 2: return Dqip16bGemmV0_GemmLast_2xM<term, type, 2>;
            case 3: return Dqip16bGemmV0_GemmLast_2xM<term, type, 3>;
            case 4: return Dqip16bGemmV0_GemmLast_2xM<term, type, 4>;
            case 5: return Dqip16bGemmV0_GemmLast_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Dqip16bGemmV0_GemmLast(const uint8_t* A, const DqipParam& p, 
            const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C)
        {
            size_t n = 5;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmLast_2xM<term, type>(n);
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmLast_2xM<term, type>(m);

            __m128i _bSums[2];
            __m128 _bScale[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = _mm_set1_ps(params[0]);
                _params[1] = _mm_set1_ps(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                _bSums[0] = _mm_loadu_si128((__m128i*)(bSums + j) + 0);
                _bSums[1] = _mm_loadu_si128((__m128i*)(bSums + j) + 1);
                _bScale[0] = _mm_loadu_ps(bScale + j + 0);
                _bScale[1] = _mm_loadu_ps(bScale + j + F);
                _bias[0] = _mm_loadu_ps(bias + j + 0);
                _bias[1] = _mm_loadu_ps(bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + j + 0);
                    _params[1] = _mm_loadu_ps(params + j + F);
                }
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
                B += a.aK * DF;
                sum += DF;
                C += DF * a.eC;
            }
        }

        template<Term16bType term> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationGelu>; break;
            default:
                assert(0);
            }
        }

        void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            if (p.typeC == SimdTensorData16b)
                SetGemmLast<Term16bLast16b>(p, gemmLast);
            else if(p.typeC == SimdTensorData32f)
                SetGemmLast<Term16bLast32f>(p, gemmLast);
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct16bGemmV0::SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p)
            : Base::SynetDynamicQuantizedInnerProduct16bGemmV0(p)
        {
            SetAlgParam(F, 5, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if(p.typeA == SimdTensorData32f)
                _convertA = SynetDynamicQuantizedInnerProduct16bGemmV0_ConvertAf;
            _gemmBody = Dqip16bGemmV0_GemmBody;
            SetGemmLast(p, _gemmLast);
        }
    }
#endif
}
