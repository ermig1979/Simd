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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)    
    namespace AmxBf16
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ConvertA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            if (p.K == a.aK)
            {
                Float32ToBFloat16(src, K * M, dst);
            }
            else
            {
                size_t KDF = Simd::AlignLo(p.K, DF);
                __mmask32 tail = TailMask32(p.K - KDF);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t k = 0;
                    for (; k < KDF; k += DF)
                        Float32ToBFloat16(src + k, dst + k);
                    if(tail)
                        Float32ToBFloat16(src + k, dst + k, tail);
                    src += p.K;
                    dst += a.aK;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, int cfg> void InnerProduct16bGemmNN_2x2(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0, int post, __m512* bias, uint8_t* dst)
        {
            int dC = (int)a.cN, dA = (int)a.aK, dD = int(p.N * a.eC);
            int strideA = dA * 2, strideB = 64, strideC = dC * 4;
            const uint16_t *A1 = A0 + dA * 16, *B1 = B0 + a.bK * 16;
            float* C1 = C0 + 16 * dC;

            if (cfg)
                SetTileConf2x2(M, N);
            if (update)
            {
                _tile_stream_loadd(0, C0 + 0, strideC);
                _tile_stream_loadd(1, C0 + F, strideC);
                _tile_stream_loadd(2, C1 + 0, strideC);
                _tile_stream_loadd(3, C1 + F, strideC);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }

            int K32 = (int)K - 32, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            _tile_loadd(6, B0, strideB);
            for (; k < K32;)
            {
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_stream_loadd(5, A1 + k, strideA);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                k += 32;
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_stream_loadd(5, A1 + k, strideA);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(1, C0 + F, strideC);
            _tile_stored(2, C1 + 0, strideC);
            _tile_stored(3, C1 + F, strideC);
            if (post)
            {
                __mmask16 tail = TailMask16(N - F);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply2x8<term, SimdConvolutionActivationIdentity>(dst + i * dD, dD, C0 + i * dC, dC, bias, NULL, tail);
                for (; i < M; ++i)
                    Apply2<term, SimdConvolutionActivationIdentity>(dst + i * dD, C0 + i * dC, bias, NULL, tail);
            }
            else
            {
                TileMoveToMemory(C0 + 0, dC);
                TileMoveToMemory(C0 + F, dC);
                TileMoveToMemory(C1 + 0, dC);
                TileMoveToMemory(C1 + F, dC);
            }
        }

        template<Term16bType term, int cfg> void InnerProduct16bGemmNN_2x1(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0, int post, __m512* bias, uint8_t* dst)
        {
            int dC = (int)a.cN, dA = (int)a.aK, dD = int(p.N * a.eC);
            int strideA = dA * 2, strideB = 64, strideC = dC * 4;
            const uint16_t* A1 = A0 + dA * 16;
            float* C1 = C0 + 16 * dC;

            if (cfg)
                SetTileConf2x1(M, N);
            if (update)
            {
                _tile_stream_loadd(0, C0 + 0, strideC);
                _tile_stream_loadd(2, C1 + 0, strideC);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(2);
            }

            int K32 = (int)K - 32, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            for (; k < K32;)
            {
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_stream_loadd(5, A1 + k, strideA);
                _tile_dpbf16ps(0, 4, 6);
                k += 32;
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, B0 + k * 16, strideB);
            _tile_stream_loadd(5, A1 + k, strideA);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(2, C1 + 0, strideC);
            if (post)
            {
                __mmask16 tail = TailMask16(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply1x8<term, SimdConvolutionActivationIdentity>(dst + i * dD, dD, C0 + i * dC, dC, bias, NULL, tail);
                for (; i < M; ++i)
                    Apply1<term, SimdConvolutionActivationIdentity>(dst + i * dD, C0 + i * dC, bias, NULL, tail);
            }
            else
            {
                TileMoveToMemory(C0 + 0, dC);
                TileMoveToMemory(C1 + 0, dC);
            }
        }

        template<Term16bType term, int cfg> void InnerProduct16bGemmNN_1x2(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0, int post, __m512* bias, uint8_t* dst)
        {
            int dC = (int)a.cN, dA = (int)a.aK, dD = int(p.N * a.eC);
            int strideA = dA * 2, strideB = 64, strideC = dC * 4;
            const uint16_t* B1 = B0 + a.bK * 16;

            if (cfg)
                SetTileConf1x2(M, N);
            if (update)
            {
                _tile_stream_loadd(0, C0 + 0, strideC);
                _tile_stream_loadd(1, C0 + F, strideC);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int K32 = (int)K - 32, k = 0;
            _tile_loadd(6, B0 + k * 16, strideB);
            for (; k < K32;)
            {
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_dpbf16ps(0, 4, 6);
                k += 32;
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, A0 + k, strideA);
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(1, C0 + F, strideC);
            if (post)
            {
                __mmask16 tail = TailMask16(N - F);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply2x8<term, SimdConvolutionActivationIdentity>(dst + i * dD, dD, C0 + i * dC, dC, bias, NULL, tail);
                for (; i < M; ++i)
                    Apply2<term, SimdConvolutionActivationIdentity>(dst + i * dD, C0 + i * dC, bias, NULL, tail);
            }
            else
            {
                TileMoveToMemory(C0 + 0, dC);
                TileMoveToMemory(C0 + F, dC);
            }
        }

        template<Term16bType term, int cfg> void InnerProduct16bGemmNN_1x1(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0, int post, __m512* bias, uint8_t* dst)
        {
            int dC = (int)a.cN, dA = (int)a.aK, dD = int(p.N * a.eC);
            int strideA = dA * 2, strideB = 64, strideC = dC * 4;

            if (cfg)
                SetTileConf1x1(M, N);
            if (update)
            {
                _tile_stream_loadd(0, C0 + 0, strideC);
            }
            else
            {
                _tile_zero(0);
            }
            for (size_t k = 0; k < K; k += 32)
            {
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, C0 + 0, strideC);
            if (post)
            {
                __mmask16 tail = TailMask16(N);
                size_t M8 = AlignLo(M, 8), i = 0;
                for (; i < M8; i += 8)
                    Apply1x8<term, SimdConvolutionActivationIdentity>(dst + i * dD, dD, C0 + i * dC, dC, bias, NULL, tail);
                for (; i < M; ++i)
                    Apply1<term, SimdConvolutionActivationIdentity>(dst + i * dD, C0 + i * dC, bias, NULL, tail);
            }
            else
            {
                TileMoveToMemory(C0 + 0, dC);
            }
        }

        typedef void(*GemmNN_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C, int post, __m512* bias, uint8_t* dst);

        template<Term16bType term> void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C, int post, const float* bias, uint8_t* dst)
        {
            size_t m = 32, m1 = M, mm = AlignLo(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.bK * DF, dC = (a.macroK < a.aK || a.macroN != a.aN || a.macroM != a.aM || C == (float*)dst) ? a.cN : 0, dD = p.N * a.eC;
            __m512 _bias[2];
            if (mm)
            {
                t = AlignHi(t, 16), mm = m1 - t;
                GemmNN_Ptr body_2 = InnerProduct16bGemmNN_2x2<term, 0>;
                GemmNN_Ptr tail_2 = t > 16 ? InnerProduct16bGemmNN_2x2<term, 0> : InnerProduct16bGemmNN_1x2<term, 0>;
                GemmNN_Ptr body_1 = InnerProduct16bGemmNN_2x1<term, 0>;
                GemmNN_Ptr tail_1 = t > 16 ? InnerProduct16bGemmNN_2x1<term, 0> : InnerProduct16bGemmNN_1x1<term, 0>;
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    _bias[0] = _mm512_loadu_ps(bias + j + 0);
                    _bias[1] = _mm512_loadu_ps(bias + j + F);
                    size_t dN = Simd::Min(DF, N - j);
                    size_t i = 0;
                    if (dN > F)
                    {
                        for (; i < mm; i += m)
                            body_2(A + i * dA, p, a, m, dN, K, update, B, C + i * dC, post, _bias, dst + i * dD);
                        if (t)
                            tail_2(A + mm * dA, p, a, t, dN, K, update, B, C + i * dC, post, _bias, dst + mm * dD);
                    }
                    else
                    {
                        for (; i < mm; i += m)
                            body_1(A + i * dA, p, a, m, dN, K, update, B, C + i * dC, post, _bias, dst + i * dD);
                        if (t)
                            tail_1(A + mm * dA, p, a, t, dN, K, update, B, C + i * dC, post, _bias, dst + mm * dD);
                    }
                    B += dB;
                    C += dN;
                    dst += DF * a.eC;
                }
            }
            else
            {
                GemmNN_Ptr tail_2 = t > 16 ? InnerProduct16bGemmNN_2x2<term, 0> : InnerProduct16bGemmNN_1x2<term, 0>;
                GemmNN_Ptr tail_1 = t > 16 ? InnerProduct16bGemmNN_2x1<term, 0> : InnerProduct16bGemmNN_1x1<term, 0>;
                if (t > 16)
                    SetTileConf2x2(t, 32);
                else
                    SetTileConf1x2(t, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    _bias[0] = _mm512_loadu_ps(bias + j + 0);
                    _bias[1] = _mm512_loadu_ps(bias + j + F);
                    size_t dN = Simd::Min(DF, N - j);
                    if (dN > F)
                        tail_2(A, p, a, t, dN, K, update, B, C, post, _bias, dst);
                    else
                        tail_1(A, p, a, t, dN, K, update, B, C, post, _bias, dst);
                    B += dB;
                    C += dN;
                    dst += DF * a.eC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Avx512bw::SynetInnerProduct16bGemmNN(p)
        {
            if (p.K < F)
                return;
            SetAlgParam(F, F * 2, F * 2, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData16b)
                    _prepA = Avx512bw::InnerProduct16bGemmNN_ReorderA;
                else
                    _prepA = InnerProduct16bGemmNN_ConvertA;
            }
            if (p.typeC == SimdTensorData16b)
                _gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast16b>;
            else
                _gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast32f>;
        }
    }
#endif
}
