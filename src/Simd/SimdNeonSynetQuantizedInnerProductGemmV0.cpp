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
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmV0::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmV0::GemmPtr GemmPtr;

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmV0_PrepA_8u(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
        {
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
            size_t K, size_t N, int update, const int8_t* B0, const int32x4_t* bias, const float32x4_t* norm, const int32x4_t& zero, int32_t* buf, uint8_t* C)
        {
            int32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41;
            uint8x16_t a0;
            int8x16_t b0, b1;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            const int8_t* B1 = B0 + a.bK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = vld1q_s32(buf + 0 * dB + 0), c01 = vld1q_s32(buf + 0 * dB + F);
                    if (M > 1) c10 = vld1q_s32(buf + 1 * dB + 0), c11 = vld1q_s32(buf + 1 * dB + F);
                    if (M > 2) c20 = vld1q_s32(buf + 2 * dB + 0), c21 = vld1q_s32(buf + 2 * dB + F);
                    if (M > 3) c30 = vld1q_s32(buf + 3 * dB + 0), c31 = vld1q_s32(buf + 3 * dB + F);
                    if (M > 4) c40 = vld1q_s32(buf + 4 * dB + 0), c41 = vld1q_s32(buf + 4 * dB + F);
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
                    if (M > 0) Save2<term>(C, buf, c00, c01, bias, norm, zero), C += dC, buf += dB;
                    if (M > 1) Save2<term>(C, buf, c10, c11, bias, norm, zero), C += dC, buf += dB;
                    if (M > 2) Save2<term>(C, buf, c20, c21, bias, norm, zero), C += dC, buf += dB;
                    if (M > 3) Save2<term>(C, buf, c30, c31, bias, norm, zero), C += dC, buf += dB;
                    if (M > 4) Save2<term>(C, buf, c40, c41, bias, norm, zero), C += dC, buf += dB;
                }
                else
                {
                    N -= F;
                    if (M > 0) Save2<term>(C, buf, c00, c01, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 1) Save2<term>(C, buf, c10, c11, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 2) Save2<term>(C, buf, c20, c21, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 3) Save2<term>(C, buf, c30, c31, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 4) Save2<term>(C, buf, c40, c41, bias, norm, zero, N), C += dC, buf += dB;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = vld1q_s32(buf + 0 * dB);
                    if (M > 1) c10 = vld1q_s32(buf + 1 * dB);
                    if (M > 2) c20 = vld1q_s32(buf + 2 * dB);
                    if (M > 3) c30 = vld1q_s32(buf + 3 * dB);
                    if (M > 4) c40 = vld1q_s32(buf + 4 * dB);
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
                    if (M > 0) Save1<term>(C, buf, c00, bias, norm, zero), C += dC, buf += dB;
                    if (M > 1) Save1<term>(C, buf, c10, bias, norm, zero), C += dC, buf += dB;
                    if (M > 2) Save1<term>(C, buf, c20, bias, norm, zero), C += dC, buf += dB;
                    if (M > 3) Save1<term>(C, buf, c30, bias, norm, zero), C += dC, buf += dB;
                    if (M > 4) Save1<term>(C, buf, c40, bias, norm, zero), C += dC, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term>(C, buf, c00, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 1) Save1<term>(C, buf, c10, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 2) Save1<term>(C, buf, c20, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 3) Save1<term>(C, buf, c30, bias, norm, zero, N), C += dC, buf += dB;
                    if (M > 4) Save1<term>(C, buf, c40, bias, norm, zero, N), C += dC, buf += dB;
                }
            }
        }

        typedef void(*QuantizedInnerProductGemmV0_2xM_Ptr)(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, const int32x4_t* bias, const float32x4_t* norm, const int32x4_t& zero, int32_t* buf, uint8_t* C);

        template<Term8iType term> QuantizedInnerProductGemmV0_2xM_Ptr GetQuantizedInnerProductGemmV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return QuantizedInnerProductGemmV0_2xM<term, 1>;
            case 2: return QuantizedInnerProductGemmV0_2xM<term, 2>;
            case 3: return QuantizedInnerProductGemmV0_2xM<term, 3>;
            case 4: return QuantizedInnerProductGemmV0_2xM<term, 4>;
            case 5: return QuantizedInnerProductGemmV0_2xM<term, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedInnerProductGemmV0_2(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, 
            int update, const int8_t* B, int32_t* buf, int post, const int32_t* bias, const float* norm, uint32_t zero, uint8_t* C)
        {
            size_t n = 5;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            QuantizedInnerProductGemmV0_2xM_Ptr gemm_2xN = post ? GetQuantizedInnerProductGemmV0_2xM<term>(n) : GetQuantizedInnerProductGemmV0_2xM<Term8iInterim>(n);
            QuantizedInnerProductGemmV0_2xM_Ptr gemm_2xM = post ? GetQuantizedInnerProductGemmV0_2xM<term>(m) : GetQuantizedInnerProductGemmV0_2xM<Term8iInterim>(m);

            float32x4_t _norm[2];
            int32x4_t _bias[2], _zero = vdupq_n_s32(zero);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bias[0] = vld1q_s32(bias + j + 0);
                _bias[1] = vld1q_s32(bias + j + F);
                _norm[0] = vld1q_f32(norm + j + 0);
                _norm[1] = vld1q_f32(norm + j + F);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, _bias, _norm, _zero, buf + i * dB, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, _bias, _norm, _zero, buf + i * dB, C + i * dC);
                B += a.bK * DF;
                buf += DF;
                C += DF * a.eC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmV0::SynetQuantizedInnerProductGemmV0(const QuantizedInnerProductParam& p)
            : Base::SynetQuantizedInnerProductGemmV0(p)
        {
            SetAlgParam(F, 5, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
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
