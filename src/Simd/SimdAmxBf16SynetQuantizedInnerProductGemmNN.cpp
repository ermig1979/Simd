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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmNN::GemmPtr GemmPtr;

        //-------------------------------------------------------------------------------------------------

        template<Term8iType term, int cfg> void QuantizedInnerProductGemm_32x32(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, uint8_t* C)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.bK * F;
            if (cfg)
                SetTileConf2x2(M, N);
            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
                _tile_stream_loadd(1, sum + F, strideS);
                _tile_stream_loadd(2, sum + 16 * dS + 0, strideS);
                _tile_stream_loadd(3, sum + 16 * dS + F, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            _tile_loadd(6, B0 + k * 16, strideB);
            for (; k < K64; A1 += stepA)
            {
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_stream_loadd(5, A1, strideA);
                _tile_dpbusd(0, 4, 6);
                _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                _tile_dpbusd(2, 5, 6);
                k += 64;
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_stream_loadd(5, A1, strideA);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);

            _tile_stored(0, sum + 0, strideS);
            _tile_stored(1, sum + F, strideS);
            _tile_stored(2, sum + 16 * dS + 0, strideS);
            _tile_stored(3, sum + 16 * dS + F, strideS);
            if (term == Term8iLast8u)
            {
                __mmask32 tailN = TailMask32(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply8u2x8(C + i * dC, dC, sum + i * dS, dS, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    Apply8u2(C + i * dC, sum + i * dS, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
            else
            {
                TileMoveToMemory(sum + 0, dS);
                TileMoveToMemory(sum + F, dS);
                TileMoveToMemory(sum + 16 * dS + 0, dS);
                TileMoveToMemory(sum + 16 * dS + F, dS);
            }
        }

        template<Term8iType term, int cfg> void QuantizedInnerProductGemm_32x16(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, uint8_t* C)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;

            if (cfg)
                SetTileConf2x1(M, N);
            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
                _tile_stream_loadd(2, sum + 16 * dS + 0, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(2);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            for (; k < K64; A1 += stepA, k += 64)
            {
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_stream_loadd(5, A1, strideA);
                _tile_dpbusd(0, 4, 6);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_loadd(6, B0 + k * 16, strideB);
            _tile_stream_loadd(5, A1, strideA);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(2, 5, 6);

            _tile_stored(0, sum + 0, strideS);
            _tile_stored(2, sum + 16 * dS + 0, strideS);
            if (term == Term8iLast8u)
            {
                __mmask16 tailN = TailMask16(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply1x8<term>(C + i * dC, dC, sum + i * dS, dS, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    Apply1<term>(C + i * dC, sum + i * dS, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
            else
            {
                TileMoveToMemory(sum + 0, dS);
                TileMoveToMemory(sum + 16 * dS + 0, dS);
            }
        }

        template<Term8iType term, int cfg> void QuantizedInnerProductGemm_16x32(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, uint8_t* C)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const int8_t* B1 = B0 + a.bK * F;

            if (cfg)
                SetTileConf1x2(M, N);
            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
                _tile_stream_loadd(1, sum + F, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_loadd(6, B0 + k * 16, strideB);
            for (; k < K64; A0 += stepA)
            {
                _tile_stream_loadd(4, A0, strideA);
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_dpbusd(0, 4, 6);
                k += 64;
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stream_loadd(4, A0, strideA);
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);

            _tile_stored(0, sum + 0, strideS);
            _tile_stored(1, sum + F, strideS);
            if (term == Term8iLast8u)
            {
                __mmask32 tailN = TailMask32(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply8u2x8(C + i * dC, dC, sum + i * dS, dS, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    Apply8u2(C + i * dC, sum + i * dS, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
            else
            {
                TileMoveToMemory(sum + 0, dS);
                TileMoveToMemory(sum + F, dS);
            }
        }

        template<Term8iType term, int cfg> void QuantizedInnerProductGemm_16x16(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, uint8_t* C)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;

            if (cfg)
                SetTileConf1x1(M, N);
            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
            }
            else
            {
                _tile_zero(0);
            }

            for (size_t k = 0; k < K; A0 += stepA, k += 64)
            {
                _tile_stream_loadd(4, A0, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(0, 4, 6);
            }

            _tile_stored(0, sum + 0, strideS);
            if (term == Term8iLast8u)
            {
                __mmask16 tailN = TailMask16(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply1x8<term>(C + i * dC, dC, sum + i * dS, dS, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    Apply1<term>(C + i * dC, sum + i * dS, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
            else
            {
                TileMoveToMemory(sum + 0, dS);
            }
        }

        typedef void(*QuantizedInnerProductGemmPtr)(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, uint8_t* C);

        template<Term8iType term> void QuantizedInnerProductGemm_2(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, 
            int update, const int8_t* B, int32_t* buf, int post, const int32_t* bias, const float* norm, uint32_t zero, uint8_t* C)
        {
            size_t n = 32;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;

            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            if (Mn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 0 && p.K == a.aK);
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), Mn = M - m;
                QuantizedInnerProductGemmPtr body_2 = QuantizedInnerProductGemm_32x32<term, 0>;
                QuantizedInnerProductGemmPtr tail_2 = m > 16 ? QuantizedInnerProductGemm_32x32<term, 0> : QuantizedInnerProductGemm_16x32<term, 0>;
                QuantizedInnerProductGemmPtr body_1 = QuantizedInnerProductGemm_32x16<term, 0>;
                QuantizedInnerProductGemmPtr tail_1 = m > 16 ? QuantizedInnerProductGemm_32x16<term, 0> : QuantizedInnerProductGemm_16x16<term, 0>;
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + j) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + j) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + j + 0);
                    _norm[1] = _mm512_loadu_ps(norm + j + F);
                    size_t i = 0;
                    if (dN > F)
                    {
                        for (; i < Mn; i += n)
                            body_2(A + i * dA, p, a, n, dN, K, update, B, _bias, _norm, _zero, buf + i * dB, C + i * dC);
                        if (m)
                            tail_2(A + Mn * dA, p, a, m, dN, K, update, B, _bias, _norm, _zero, buf + i * dB, C + Mn * dC);
                    }
                    else
                    {
                        for (; i < Mn; i += n)
                            body_1(A + i * dA, p, a, n, dN, K, update, B, _bias, _norm, _zero, buf + i * dB, C + i * dC);
                        if (m)
                            tail_1(A + Mn * dA, p, a, m, dN, K, update, B, _bias, _norm, _zero, buf + i * dB, C + Mn * dC);
                    }
                    B += a.bK * DF;
                    buf += DF;
                    C += DF * a.eC;
                }
            }
            else
            {
                QuantizedInnerProductGemmPtr tail_2 = m > 16 ? QuantizedInnerProductGemm_32x32<term, 0> : QuantizedInnerProductGemm_16x32<term, 0>;
                QuantizedInnerProductGemmPtr tail_1 = m > 16 ? QuantizedInnerProductGemm_32x16<term, 0> : QuantizedInnerProductGemm_16x16<term, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + j) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + j) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + j + 0);
                    _norm[1] = _mm512_loadu_ps(norm + j + F);
                    if (dN > F)
                        tail_2(A, p, a, m, dN, K, update, B, _bias, _norm, _zero, buf, C);
                    else
                        tail_1(A, p, a, m, dN, K, update, B, _bias, _norm, _zero, buf, C);
                    B += a.bK * DF;
                    buf += DF;
                    C += DF * a.eC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmNN::SynetQuantizedInnerProductGemmNN(const QuantizedInnerProductParam& p)
            : Avx512vnni::SynetQuantizedInnerProductGemmNN(p)
        {
            SetAlgParam(F, F * 2, F * 2, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (p.M > 1)
            {
                if (p.typeC == SimdTensorData8u)
                    _gemm = QuantizedInnerProductGemm_2<Term8iLast8u>;
                else
                    _gemm = NULL;// QuantizedInnerProductGemm_2<Term8iLast32f>;
            }
        }
    }
#endif
}
