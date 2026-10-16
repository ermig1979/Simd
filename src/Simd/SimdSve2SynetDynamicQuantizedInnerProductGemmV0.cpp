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
#include "Simd/SimdSynetDynamicQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSve2.h"

#include <cfloat>

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProductGemmV0_Convert(const float* src, float norm, uint8_t zero, const DqipParam& p, const AlgParam& a, size_t N, uint8_t* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            size_t K = p.K, KA = a.aK;
            svfloat32_t _norm = svdup_n_f32(norm);
            svint32_t _zero = svdup_n_s32(zero);
            for (size_t j = 0; j < N; ++j)
            {
                size_t k = 0;
                for (; k + F <= K; k += F)
                    QuantizeLinear(src + k, _norm, _zero, dst + k, body);
                if (k < K)
                    QuantizeLinear(src + k, _norm, _zero, dst + k, svwhilelt_b32(k, K));
                for (k = K; k < KA; ++k)
                    dst[k] = 0;
                src += K;
                dst += KA;
            }
        }

        //-----------------------------------------------------------------------------------------

        void MinMax32f(const float* src, size_t size, float* min, float* max)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t vmin = svdup_n_f32(FLT_MAX);
            svfloat32_t vmax = svdup_n_f32(-FLT_MAX);
            size_t i = 0;
            for (; i + 4 * F <= size; i += 4 * F)
            {
                svfloat32_t s0 = svld1_f32(body, src + i + 0 * F);
                svfloat32_t s1 = svld1_f32(body, src + i + 1 * F);
                svfloat32_t s2 = svld1_f32(body, src + i + 2 * F);
                svfloat32_t s3 = svld1_f32(body, src + i + 3 * F);
                vmin = svmin_f32_x(body, vmin, svmin_f32_x(body, svmin_f32_x(body, s0, s1), svmin_f32_x(body, s2, s3)));
                vmax = svmax_f32_x(body, vmax, svmax_f32_x(body, svmax_f32_x(body, s0, s1), svmax_f32_x(body, s2, s3)));
            }
            for (; i + F <= size; i += F)
            {
                svfloat32_t s0 = svld1_f32(body, src + i);
                vmin = svmin_f32_x(body, vmin, s0);
                vmax = svmax_f32_x(body, vmax, s0);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t s0 = svld1_f32(tail, src + i);
                vmin = svmin_f32_m(tail, vmin, s0);
                vmax = svmax_f32_m(tail, vmax, s0);
            }
            *min = svminv_f32(body, vmin);
            *max = svmaxv_f32(body, vmax);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void StoreBody1(int32_t* ptr, const svint32_t& val0)
        {
            svst1_s32(svptrue_b32(), ptr, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, const svint32_t& val0, const svint32_t& val1)
        {
            const size_t F = svcntw();
            svst1_s32(svptrue_b32(), ptr + 0, val0);
            svst1_s32(svptrue_b32(), ptr + F, val1);
        }

        template<int M> void DqipGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            svuint8_t a0;
            svint8_t b0, b1;
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
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS + 0), c01 = svld1_s32(body32, sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS + 0), c11 = svld1_s32(body32, sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS + 0), c21 = svld1_s32(body32, sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS + 0), c31 = svld1_s32(body32, sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS + 0), c41 = svld1_s32(body32, sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS + 0), c51 = svld1_s32(body32, sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS + 0), c61 = svld1_s32(body32, sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS + 0), c71 = svld1_s32(body32, sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS + 0), c81 = svld1_s32(body32, sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS + 0), c91 = svld1_s32(body32, sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS + 0), cA1 = svld1_s32(body32, sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS + 0), cB1 = svld1_s32(body32, sum + 0xB * dS + F);
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
                if (M > 0xA) StoreBody2(sum + 0xA * dS, cA0, cA1);
                if (M > 0xB) StoreBody2(sum + 0xB * dS, cB0, cB1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS);
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
                if (M > 0xA) StoreBody1(sum + 0xA * dS, cA0);
                if (M > 0xB) StoreBody1(sum + 0xB * dS, cB0);
            }
        }

        typedef void(*DqipGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE DqipGemmV0_GemmBody_2xM_Ptr GetDqipGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return DqipGemmV0_GemmBody_2xM<0x1>;
            case 0x2: return DqipGemmV0_GemmBody_2xM<0x2>;
            case 0x3: return DqipGemmV0_GemmBody_2xM<0x3>;
            case 0x4: return DqipGemmV0_GemmBody_2xM<0x4>;
            case 0x5: return DqipGemmV0_GemmBody_2xM<0x5>;
            case 0x6: return DqipGemmV0_GemmBody_2xM<0x6>;
            case 0x7: return DqipGemmV0_GemmBody_2xM<0x7>;
            case 0x8: return DqipGemmV0_GemmBody_2xM<0x8>;
            case 0x9: return DqipGemmV0_GemmBody_2xM<0x9>;
            case 0xA: return DqipGemmV0_GemmBody_2xM<0xA>;
            case 0xB: return DqipGemmV0_GemmBody_2xM<0xB>;
            case 0xC: return DqipGemmV0_GemmBody_2xM<0xC>;
            }
            assert(0);
            return NULL;
        }

        static void DqipGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            const size_t F = svcntw(), DF = F * 2;
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
            const svint32_t& corr0, const svint32_t& corr1, const svfloat32_t& norm0, const svfloat32_t& norm1,
            const svfloat32_t& bias0, const svfloat32_t& bias1, const svfloat32_t& param0, const svfloat32_t& param1, float* C)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            svuint8_t a0;
            svint8_t b0, b1;
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
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS + 0), c01 = svld1_s32(body32, sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS + 0), c11 = svld1_s32(body32, sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS + 0), c21 = svld1_s32(body32, sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS + 0), c31 = svld1_s32(body32, sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS + 0), c41 = svld1_s32(body32, sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS + 0), c51 = svld1_s32(body32, sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS + 0), c61 = svld1_s32(body32, sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS + 0), c71 = svld1_s32(body32, sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS + 0), c81 = svld1_s32(body32, sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS + 0), c91 = svld1_s32(body32, sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS + 0), cA1 = svld1_s32(body32, sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS + 0), cB1 = svld1_s32(body32, sum + 0xB * dS + F);
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
                if (N == F * 2)
                {
                    if (M > 0x0) Save2<type>(C + 0x0 * dC, c00, c01, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x1) Save2<type>(C + 0x1 * dC, c10, c11, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x2) Save2<type>(C + 0x2 * dC, c20, c21, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x3) Save2<type>(C + 0x3 * dC, c30, c31, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x4) Save2<type>(C + 0x4 * dC, c40, c41, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x5) Save2<type>(C + 0x5 * dC, c50, c51, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x6) Save2<type>(C + 0x6 * dC, c60, c61, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x7) Save2<type>(C + 0x7 * dC, c70, c71, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x8) Save2<type>(C + 0x8 * dC, c80, c81, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0x9) Save2<type>(C + 0x9 * dC, c90, c91, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0xA) Save2<type>(C + 0xA * dC, cA0, cA1, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                    if (M > 0xB) Save2<type>(C + 0xB * dC, cB0, cB1, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1);
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0x0) Save2<type>(C + 0x0 * dC, c00, c01, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x1) Save2<type>(C + 0x1 * dC, c10, c11, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x2) Save2<type>(C + 0x2 * dC, c20, c21, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x3) Save2<type>(C + 0x3 * dC, c30, c31, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x4) Save2<type>(C + 0x4 * dC, c40, c41, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x5) Save2<type>(C + 0x5 * dC, c50, c51, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x6) Save2<type>(C + 0x6 * dC, c60, c61, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x7) Save2<type>(C + 0x7 * dC, c70, c71, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x8) Save2<type>(C + 0x8 * dC, c80, c81, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0x9) Save2<type>(C + 0x9 * dC, c90, c91, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0xA) Save2<type>(C + 0xA * dC, cA0, cA1, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                    if (M > 0xB) Save2<type>(C + 0xB * dC, cB0, cB1, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, tail);
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS);
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
                    if (M > 0x0) Save1<type>(C + 0x0 * dC, c00, corr0, norm0, bias0, param0, param1);
                    if (M > 0x1) Save1<type>(C + 0x1 * dC, c10, corr0, norm0, bias0, param0, param1);
                    if (M > 0x2) Save1<type>(C + 0x2 * dC, c20, corr0, norm0, bias0, param0, param1);
                    if (M > 0x3) Save1<type>(C + 0x3 * dC, c30, corr0, norm0, bias0, param0, param1);
                    if (M > 0x4) Save1<type>(C + 0x4 * dC, c40, corr0, norm0, bias0, param0, param1);
                    if (M > 0x5) Save1<type>(C + 0x5 * dC, c50, corr0, norm0, bias0, param0, param1);
                    if (M > 0x6) Save1<type>(C + 0x6 * dC, c60, corr0, norm0, bias0, param0, param1);
                    if (M > 0x7) Save1<type>(C + 0x7 * dC, c70, corr0, norm0, bias0, param0, param1);
                    if (M > 0x8) Save1<type>(C + 0x8 * dC, c80, corr0, norm0, bias0, param0, param1);
                    if (M > 0x9) Save1<type>(C + 0x9 * dC, c90, corr0, norm0, bias0, param0, param1);
                    if (M > 0xA) Save1<type>(C + 0xA * dC, cA0, corr0, norm0, bias0, param0, param1);
                    if (M > 0xB) Save1<type>(C + 0xB * dC, cB0, corr0, norm0, bias0, param0, param1);
                }
                else
                {
                    size_t tail = N;
                    if (M > 0x0) Save1<type>(C + 0x0 * dC, c00, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x1) Save1<type>(C + 0x1 * dC, c10, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x2) Save1<type>(C + 0x2 * dC, c20, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x3) Save1<type>(C + 0x3 * dC, c30, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x4) Save1<type>(C + 0x4 * dC, c40, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x5) Save1<type>(C + 0x5 * dC, c50, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x6) Save1<type>(C + 0x6 * dC, c60, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x7) Save1<type>(C + 0x7 * dC, c70, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x8) Save1<type>(C + 0x8 * dC, c80, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0x9) Save1<type>(C + 0x9 * dC, c90, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0xA) Save1<type>(C + 0xA * dC, cA0, corr0, norm0, bias0, param0, param1, tail);
                    if (M > 0xB) Save1<type>(C + 0xB * dC, cB0, corr0, norm0, bias0, param0, param1, tail);
                }
            }
        }

        typedef void(*DqipGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const svint32_t& corr0, const svint32_t& corr1, const svfloat32_t& norm0, const svfloat32_t& norm1,
            const svfloat32_t& bias0, const svfloat32_t& bias1, const svfloat32_t& param0, const svfloat32_t& param1, float* C);

        template<SimdConvolutionActivationType type> SIMD_INLINE DqipGemmV0_GemmLast_2xM_Ptr GetDqipGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return DqipGemmV0_GemmLast_2xM<type, 0x1>;
            case 0x2: return DqipGemmV0_GemmLast_2xM<type, 0x2>;
            case 0x3: return DqipGemmV0_GemmLast_2xM<type, 0x3>;
            case 0x4: return DqipGemmV0_GemmLast_2xM<type, 0x4>;
            case 0x5: return DqipGemmV0_GemmLast_2xM<type, 0x5>;
            case 0x6: return DqipGemmV0_GemmLast_2xM<type, 0x6>;
            case 0x7: return DqipGemmV0_GemmLast_2xM<type, 0x7>;
            case 0x8: return DqipGemmV0_GemmLast_2xM<type, 0x8>;
            case 0x9: return DqipGemmV0_GemmLast_2xM<type, 0x9>;
            case 0xA: return DqipGemmV0_GemmLast_2xM<type, 0xA>;
            case 0xB: return DqipGemmV0_GemmLast_2xM<type, 0xB>;
            case 0xC: return DqipGemmV0_GemmLast_2xM<type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void DqipGemmV0_GemmLast(const uint8_t* A, const DynamicQuantizedInnerProductParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* corr, const float* norm, const float* bias, const float* params, float* C)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.sN, dA = a.aK, dC = p.N;
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqipGemmV0_GemmLast_2xM<type>(n);
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqipGemmV0_GemmLast_2xM<type>(m);

            svfloat32_t param0 = svdup_n_f32(0.0f), param1 = svdup_n_f32(0.0f);
            if (type != SimdConvolutionActivationIdentity)
            {
                param0 = svdup_n_f32(params[0]);
                param1 = svdup_n_f32(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                svint32_t corr0 = svld1_s32(body, corr + j + 0);
                svint32_t corr1 = svld1_s32(body, corr + j + F);
                svfloat32_t norm0 = svld1_f32(body, norm + j + 0);
                svfloat32_t norm1 = svld1_f32(body, norm + j + F);
                svfloat32_t bias0 = svld1_f32(body, bias + j + 0);
                svfloat32_t bias1 = svld1_f32(body, bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(body, params + j + 0);
                    param1 = svld1_f32(body, params + j + F);
                }
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, corr0, corr1, norm0, norm1, bias0, bias1, param0, param1, C + i * dC);
                B += a.aK * DF;
                sum += DF;
                C += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProductGemmV0::SynetDynamicQuantizedInnerProductGemmV0(const DynamicQuantizedInnerProductParam& p)
            : Base::SynetDynamicQuantizedInnerProductGemmV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _minMax32f = MinMax32f;
            _convert = SynetDynamicQuantizedInnerProductGemmV0_Convert;
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
