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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct32fGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct32fGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct32fGemmV0_Convert(const float* src, float norm, uint8_t zero, const DqipParam& p, const AlgParam& a, size_t N, uint8_t* dst)
        {
            size_t K = p.K, K4 = AlignLo(K, 4), K16 = AlignLo(K, 16), KA = a.aK;
            float32x4_t _norm = vdupq_n_f32(norm);
            int32x4_t _zero = vdupq_n_s32(zero);
            for (size_t j = 0; j < N; ++j)
            {
                size_t k = 0;
                for (; k < K16; k += 16)
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

        SIMD_INLINE void StoreBody1(int32_t* ptr, int32x4_t val0)
        {
            vst1q_s32(ptr, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, int32x4_t val0, int32x4_t val1)
        {
            vst1q_s32(ptr + 0, val0);
            vst1q_s32(ptr + F, val1);
        }

        template<int M> void DqipGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            int32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41;
            uint8x16_t a0;
            int8x16_t b0, b1;
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
                    if (M > 0) c00 = vld1q_s32(sum + 0 * dS + 0), c01 = vld1q_s32(sum + 0 * dS + F);
                    if (M > 1) c10 = vld1q_s32(sum + 1 * dS + 0), c11 = vld1q_s32(sum + 1 * dS + F);
                    if (M > 2) c20 = vld1q_s32(sum + 2 * dS + 0), c21 = vld1q_s32(sum + 2 * dS + F);
                    if (M > 3) c30 = vld1q_s32(sum + 3 * dS + 0), c31 = vld1q_s32(sum + 3 * dS + F);
                    if (M > 4) c40 = vld1q_s32(sum + 4 * dS + 0), c41 = vld1q_s32(sum + 4 * dS + F);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
                    if (M > 1) c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
                    if (M > 2) c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
                    if (M > 3) c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
                    if (M > 4) c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    b1 = vld1q_s8(B1);
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
                    if (M > 0) c00 = vld1q_s32(sum + 0 * dS + 0);
                    if (M > 1) c10 = vld1q_s32(sum + 1 * dS + 0);
                    if (M > 2) c20 = vld1q_s32(sum + 2 * dS + 0);
                    if (M > 3) c30 = vld1q_s32(sum + 3 * dS + 0);
                    if (M > 4) c40 = vld1q_s32(sum + 4 * dS + 0);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_s32(0);
                    if (M > 1) c10 = vdupq_n_s32(0);
                    if (M > 2) c20 = vdupq_n_s32(0);
                    if (M > 3) c30 = vdupq_n_s32(0);
                    if (M > 4) c40 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
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
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
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

        template<SimdConvolutionActivationType type, int M> void DqipGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32x4_t* corr, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, float* C)
        {
            int32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41;
            uint8x16_t a0;
            int8x16_t b0, b1;
            size_t dS = a.sN, dA = a.aK, dC = p.N;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = vld1q_s32(sum + 0 * dS + 0), c01 = vld1q_s32(sum + 0 * dS + F);
                    if (M > 1) c10 = vld1q_s32(sum + 1 * dS + 0), c11 = vld1q_s32(sum + 1 * dS + F);
                    if (M > 2) c20 = vld1q_s32(sum + 2 * dS + 0), c21 = vld1q_s32(sum + 2 * dS + F);
                    if (M > 3) c30 = vld1q_s32(sum + 3 * dS + 0), c31 = vld1q_s32(sum + 3 * dS + F);
                    if (M > 4) c40 = vld1q_s32(sum + 4 * dS + 0), c41 = vld1q_s32(sum + 4 * dS + F);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
                    if (M > 1) c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
                    if (M > 2) c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
                    if (M > 3) c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
                    if (M > 4) c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    b1 = vld1q_s8(B1);
                    if (M > 0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0), Madd4<true>(c01, a0, b1);
                    if (M > 1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0), Madd4<true>(c11, a0, b1);
                    if (M > 2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0), Madd4<true>(c21, a0, b1);
                    if (M > 3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0), Madd4<true>(c31, a0, b1);
                    if (M > 4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0), Madd4<true>(c41, a0, b1);
                    B0 += A, B1 += A;
                }
                if (N == DF)
                {
                    if (M > 0) Save2<type>(C + 0 * dC, c00, c01, corr, norm, bias, params);
                    if (M > 1) Save2<type>(C + 1 * dC, c10, c11, corr, norm, bias, params);
                    if (M > 2) Save2<type>(C + 2 * dC, c20, c21, corr, norm, bias, params);
                    if (M > 3) Save2<type>(C + 3 * dC, c30, c31, corr, norm, bias, params);
                    if (M > 4) Save2<type>(C + 4 * dC, c40, c41, corr, norm, bias, params);
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0) Save2<type>(C + 0 * dC, c00, c01, corr, norm, bias, params, tail);
                    if (M > 1) Save2<type>(C + 1 * dC, c10, c11, corr, norm, bias, params, tail);
                    if (M > 2) Save2<type>(C + 2 * dC, c20, c21, corr, norm, bias, params, tail);
                    if (M > 3) Save2<type>(C + 3 * dC, c30, c31, corr, norm, bias, params, tail);
                    if (M > 4) Save2<type>(C + 4 * dC, c40, c41, corr, norm, bias, params, tail);
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = vld1q_s32(sum + 0 * dS + 0);
                    if (M > 1) c10 = vld1q_s32(sum + 1 * dS + 0);
                    if (M > 2) c20 = vld1q_s32(sum + 2 * dS + 0);
                    if (M > 3) c30 = vld1q_s32(sum + 3 * dS + 0);
                    if (M > 4) c40 = vld1q_s32(sum + 4 * dS + 0);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_s32(0);
                    if (M > 1) c10 = vdupq_n_s32(0);
                    if (M > 2) c20 = vdupq_n_s32(0);
                    if (M > 3) c30 = vdupq_n_s32(0);
                    if (M > 4) c40 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    if (M > 0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0);
                    if (M > 1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0);
                    if (M > 2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0);
                    if (M > 3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0);
                    if (M > 4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0);
                    B0 += A;
                }
                if (N == F)
                {
                    if (M > 0) Save1<type>(C + 0 * dC, c00, corr, norm, bias, params);
                    if (M > 1) Save1<type>(C + 1 * dC, c10, corr, norm, bias, params);
                    if (M > 2) Save1<type>(C + 2 * dC, c20, corr, norm, bias, params);
                    if (M > 3) Save1<type>(C + 3 * dC, c30, corr, norm, bias, params);
                    if (M > 4) Save1<type>(C + 4 * dC, c40, corr, norm, bias, params);
                }
                else
                {
                    size_t tail = N;
                    if (M > 0) Save1<type>(C + 0 * dC, c00, corr, norm, bias, params, tail);
                    if (M > 1) Save1<type>(C + 1 * dC, c10, corr, norm, bias, params, tail);
                    if (M > 2) Save1<type>(C + 2 * dC, c20, corr, norm, bias, params, tail);
                    if (M > 3) Save1<type>(C + 3 * dC, c30, corr, norm, bias, params, tail);
                    if (M > 4) Save1<type>(C + 4 * dC, c40, corr, norm, bias, params, tail);
                }
            }
        }

        typedef void(*DqipGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32x4_t* corr, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, float* C);

        template<SimdConvolutionActivationType type> SIMD_INLINE DqipGemmV0_GemmLast_2xM_Ptr GetDqipGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return DqipGemmV0_GemmLast_2xM<type, 1>;
            case 2: return DqipGemmV0_GemmLast_2xM<type, 2>;
            case 3: return DqipGemmV0_GemmLast_2xM<type, 3>;
            case 4: return DqipGemmV0_GemmLast_2xM<type, 4>;
            case 5: return DqipGemmV0_GemmLast_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void DqipGemmV0_GemmLast(const uint8_t* A, const DynamicQuantizedInnerProductParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* corr, const float* norm, const float* bias, const float* params, float* C)
        {
            size_t n = 5;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.sN, dA = a.aK, dC = p.N;
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqipGemmV0_GemmLast_2xM<type>(n);
            DqipGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqipGemmV0_GemmLast_2xM<type>(m);

            int32x4_t _corr[2];
            float32x4_t _norm[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = vdupq_n_f32(params[0]);
                _params[1] = vdupq_n_f32(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                _corr[0] = vld1q_s32(corr + j + 0);
                _corr[1] = vld1q_s32(corr + j + F);
                _norm[0] = vld1q_f32(norm + j + 0);
                _norm[1] = vld1q_f32(norm + j + F);
                _bias[0] = vld1q_f32(bias + j + 0);
                _bias[1] = vld1q_f32(bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = vld1q_f32(params + j + 0);
                    _params[1] = vld1q_f32(params + j + F);
                }
                size_t dN = Simd::Min(DF, N - j);
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
            : Base::SynetDynamicQuantizedInnerProduct32fGemmV0(p)
        {
            SetAlgParam(F, 5, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _minMax32f = MinMax32f;
            _convert = SynetDynamicQuantizedInnerProduct32fGemmV0_Convert;
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
