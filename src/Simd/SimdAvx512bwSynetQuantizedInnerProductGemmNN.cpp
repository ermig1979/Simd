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
#include "Simd/SimdSynetQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmNN::GemmPtr GemmPtr;

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmNN_PrepA_8u(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
        {
            size_t KA = Simd::AlignLo(p.K, A);
            __mmask64 srcTail = TailMask64(p.K - KA), dstTail = TailMask64(a.aK - KA);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KA; k += A)
                    Copy(src + k, dst + k);
                if(dstTail)
                    Copy(src + k, dst + k, srcTail, dstTail);
                src += p.K;
                dst += a.aK;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedInnerProductGemm_2xM(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* C)
        {
            __m512i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b0, b1;
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
                    if (M > 0x0) c00 = _mm512_loadu_si512(buf + 0x0 * dB + 0), c01 = _mm512_loadu_si512(buf + 0x0 * dB + F);
                    if (M > 0x1) c10 = _mm512_loadu_si512(buf + 0x1 * dB + 0), c11 = _mm512_loadu_si512(buf + 0x1 * dB + F);
                    if (M > 0x2) c20 = _mm512_loadu_si512(buf + 0x2 * dB + 0), c21 = _mm512_loadu_si512(buf + 0x2 * dB + F);
                    if (M > 0x3) c30 = _mm512_loadu_si512(buf + 0x3 * dB + 0), c31 = _mm512_loadu_si512(buf + 0x3 * dB + F);
                    if (M > 0x4) c40 = _mm512_loadu_si512(buf + 0x4 * dB + 0), c41 = _mm512_loadu_si512(buf + 0x4 * dB + F);
                    if (M > 0x5) c50 = _mm512_loadu_si512(buf + 0x5 * dB + 0), c51 = _mm512_loadu_si512(buf + 0x5 * dB + F);
                    if (M > 0x6) c60 = _mm512_loadu_si512(buf + 0x6 * dB + 0), c61 = _mm512_loadu_si512(buf + 0x6 * dB + F);
                    if (M > 0x7) c70 = _mm512_loadu_si512(buf + 0x7 * dB + 0), c71 = _mm512_loadu_si512(buf + 0x7 * dB + F);
                    if (M > 0x8) c80 = _mm512_loadu_si512(buf + 0x8 * dB + 0), c81 = _mm512_loadu_si512(buf + 0x8 * dB + F);
                    if (M > 0x9) c90 = _mm512_loadu_si512(buf + 0x9 * dB + 0), c91 = _mm512_loadu_si512(buf + 0x9 * dB + F);
                    if (M > 0xA) ca0 = _mm512_loadu_si512(buf + 0xA * dB + 0), ca1 = _mm512_loadu_si512(buf + 0xA * dB + F);
                    if (M > 0xB) cb0 = _mm512_loadu_si512(buf + 0xB * dB + 0), cb1 = _mm512_loadu_si512(buf + 0xB * dB + F);
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
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<true>(c00, a0, b0), Madd4<true>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<true>(c10, a0, b0), Madd4<true>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<true>(c20, a0, b0), Madd4<true>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<true>(c30, a0, b0), Madd4<true>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<true>(c40, a0, b0), Madd4<true>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<true>(c50, a0, b0), Madd4<true>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<true>(c60, a0, b0), Madd4<true>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<true>(c70, a0, b0), Madd4<true>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<true>(c80, a0, b0), Madd4<true>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<true>(c90, a0, b0), Madd4<true>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<true>(ca0, a0, b0), Madd4<true>(ca1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<true>(cb0, a0, b0), Madd4<true>(cb1, a0, b1);
                    B0 += A, B1 += A;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0x0) Save2<term>(C, buf, c00, c01, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x1) Save2<term>(C, buf, c10, c11, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x2) Save2<term>(C, buf, c20, c21, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x3) Save2<term>(C, buf, c30, c31, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x4) Save2<term>(C, buf, c40, c41, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x5) Save2<term>(C, buf, c50, c51, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x6) Save2<term>(C, buf, c60, c61, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x7) Save2<term>(C, buf, c70, c71, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x8) Save2<term>(C, buf, c80, c81, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x9) Save2<term>(C, buf, c90, c91, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0xA) Save2<term>(C, buf, ca0, ca1, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0xB) Save2<term>(C, buf, cb0, cb1, bias, norm, zero, tail), C += dC, buf += dB;
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512(buf + 0x0 * dB + 0);
                    if (M > 0x1) c10 = _mm512_loadu_si512(buf + 0x1 * dB + 0);
                    if (M > 0x2) c20 = _mm512_loadu_si512(buf + 0x2 * dB + 0);
                    if (M > 0x3) c30 = _mm512_loadu_si512(buf + 0x3 * dB + 0);
                    if (M > 0x4) c40 = _mm512_loadu_si512(buf + 0x4 * dB + 0);
                    if (M > 0x5) c50 = _mm512_loadu_si512(buf + 0x5 * dB + 0);
                    if (M > 0x6) c60 = _mm512_loadu_si512(buf + 0x6 * dB + 0);
                    if (M > 0x7) c70 = _mm512_loadu_si512(buf + 0x7 * dB + 0);
                    if (M > 0x8) c80 = _mm512_loadu_si512(buf + 0x8 * dB + 0);
                    if (M > 0x9) c90 = _mm512_loadu_si512(buf + 0x9 * dB + 0);
                    if (M > 0xA) ca0 = _mm512_loadu_si512(buf + 0xA * dB + 0);
                    if (M > 0xB) cb0 = _mm512_loadu_si512(buf + 0xB * dB + 0);
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
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<true>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<true>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<true>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<true>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<true>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<true>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<true>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<true>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<true>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<true>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<true>(ca0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<true>(cb0, a0, b0);
                    B0 += A;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0x0) Save1<term>(C, buf, c00, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x1) Save1<term>(C, buf, c10, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x2) Save1<term>(C, buf, c20, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x3) Save1<term>(C, buf, c30, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x4) Save1<term>(C, buf, c40, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x5) Save1<term>(C, buf, c50, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x6) Save1<term>(C, buf, c60, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x7) Save1<term>(C, buf, c70, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x8) Save1<term>(C, buf, c80, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0x9) Save1<term>(C, buf, c90, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0xA) Save1<term>(C, buf, ca0, bias, norm, zero, tail), C += dC, buf += dB;
                if (M > 0xB) Save1<term>(C, buf, cb0, bias, norm, zero, tail), C += dC, buf += dB;
            }
        }

        typedef void(*QuantizedInnerProductGemm_2xM_Ptr)(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* C);

        template<Term8iType term> QuantizedInnerProductGemm_2xM_Ptr GetQuantizedInnerProductGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedInnerProductGemm_2xM<term, 0x1>;
            case 0x2: return QuantizedInnerProductGemm_2xM<term, 0x2>;
            case 0x3: return QuantizedInnerProductGemm_2xM<term, 0x3>;
            case 0x4: return QuantizedInnerProductGemm_2xM<term, 0x4>;
            case 0x5: return QuantizedInnerProductGemm_2xM<term, 0x5>;
            case 0x6: return QuantizedInnerProductGemm_2xM<term, 0x6>;
            case 0x7: return QuantizedInnerProductGemm_2xM<term, 0x7>;
            case 0x8: return QuantizedInnerProductGemm_2xM<term, 0x8>;
            case 0x9: return QuantizedInnerProductGemm_2xM<term, 0x9>;
            case 0xA: return QuantizedInnerProductGemm_2xM<term, 0xA>;
            case 0xB: return QuantizedInnerProductGemm_2xM<term, 0xB>;
            case 0xC: return QuantizedInnerProductGemm_2xM<term, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedInnerProductGemm_2(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, 
            int update, const int8_t* B, int32_t* buf, int post, const int32_t* bias, const float* norm, uint32_t zero, uint8_t* C)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            QuantizedInnerProductGemm_2xM_Ptr gemm_2xN = post ? GetQuantizedInnerProductGemm_2xM<term>(n) : GetQuantizedInnerProductGemm_2xM<Term8iInterim>(n);
            QuantizedInnerProductGemm_2xM_Ptr gemm_2xM = post ? GetQuantizedInnerProductGemm_2xM<term>(m) : GetQuantizedInnerProductGemm_2xM<Term8iInterim>(m);

            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bias[0] = _mm512_loadu_si512((__m512i*)(bias + j) + 0);
                _bias[1] = _mm512_loadu_si512((__m512i*)(bias + j) + 1);
                _norm[0] = _mm512_loadu_ps(norm + j + 0);
                _norm[1] = _mm512_loadu_ps(norm + j + F);
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

        SynetQuantizedInnerProductGemmNN::SynetQuantizedInnerProductGemmNN(const QuantizedInnerProductParam& p)
            : Avx2::SynetQuantizedInnerProductGemmNN(p)
        {
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData8u)
                    _prepA = QuantizedInnerProductGemmNN_PrepA_8u;
                else
                    _prepA = NULL;
            }
            if (p.typeC == SimdTensorData8u)
                _gemm = QuantizedInnerProductGemm_2<Term8iLast8u>;
            else
                _gemm = NULL;// QuantizedInnerProductGemm_2<Term8iLast32f>;
        }
    }
#endif
}
