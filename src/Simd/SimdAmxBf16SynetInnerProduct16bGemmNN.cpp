/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))   
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

        template<bool loadConf> void InnerProduct16bGemmNN_2x2(const uint16_t* A0, 
            const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0)
        {
            size_t dC = a.cN, dA = a.aK;
            int strideA = (int)dA * 2, strideB = 64, strideC = (int)dC * 4;
            const uint16_t *A1 = A0 + dA * 16, *B1 = B0 + a.aK * 16;
            float* C1 = C0 + 16 * dC;
            if (loadConf)
            {
                TileConf conf;
                conf.rows[0] = 16;
                conf.rows[1] = 16;
                conf.rows[2] = uint8_t(M - 16);
                conf.rows[3] = uint8_t(M - 16);
                conf.rows[4] = 16;
                conf.rows[5] = uint8_t(M - 16);
                conf.rows[6] = 16;
                conf.rows[7] = 16;
                conf.colsb[0] = 64;
                conf.colsb[1] = uint16_t(N - 16) * 4;
                conf.colsb[2] = 64;
                conf.colsb[3] = uint16_t(N - 16) * 4;
                conf.colsb[4] = 64;
                conf.colsb[5] = 64;
                conf.colsb[6] = 64;
                conf.colsb[7] = uint16_t(N - 16) * 4;
                _tile_loadconfig(&conf);
            }
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
            for (size_t k = 0; k < K; k += 32)
            {
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stream_loadd(5, A1 + k, strideA);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(1, C0 + F, strideC);
            _tile_stored(2, C1 + 0, strideC);
            _tile_stored(3, C1 + F, strideC);
            TileMoveToMemory(C0 + 0, dC);
            TileMoveToMemory(C0 + F, dC);
            TileMoveToMemory(C1 + 0, dC);
            TileMoveToMemory(C1 + F, dC);
        }

        void InnerProduct16bGemmNN_2x1(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0)
        {
            size_t dC = a.cN, dA = a.aK;
            int strideA = (int)dA * 2, strideB = 64, strideC = (int)dC * 4;
            const uint16_t* A1 = A0 + dA * 16;
            float* C1 = C0 + 16 * dC;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(M - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(M - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(N * 4);
            conf.colsb[2] = uint16_t(N * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = uint16_t(N * 4);
            _tile_loadconfig(&conf);

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
            for (size_t k = 0; k < K; k += 32)
            {
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, A1 + k, strideA);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(2, C1 + 0, strideC);
            TileMoveToMemory(C0 + 0, dC);
            TileMoveToMemory(C1 + 0, dC);
        }

        void InnerProduct16bGemmNN_1x2(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0)
        {
            size_t dC = a.cN, dA = a.aK;
            int strideA = (int)dA * 2, strideB = 64, strideC = (int)dC * 4;
            const uint16_t* B1 = B0 + a.aK * 16;

            TileConf conf;
            conf.rows[0] = uint8_t(M);
            conf.rows[1] = uint8_t(M);
            conf.rows[4] = uint8_t(M);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t(N - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t(N - 16) * 4;
            _tile_loadconfig(&conf);

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
            for (size_t k = 0; k < K; k += 32)
            {
                _tile_stream_loadd(4, A0 + k, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stored(0, C0 + 0, strideC);
            _tile_stored(1, C0 + F, strideC);
            TileMoveToMemory(C0 + 0, dC);
            TileMoveToMemory(C0 + F, dC);
        }

        void InnerProduct16bGemmNN_1x1(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C0)
        {
            size_t dC = a.cN, dA = a.aK;
            int strideA = (int)dA * 2, strideB = 64, strideC = (int)dC * 4;

            TileConf conf;
            conf.rows[0] = uint8_t(M);
            conf.rows[4] = uint8_t(M);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(N * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(N * 4);
            _tile_loadconfig(&conf);

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
            TileMoveToMemory(C0 + 0, dC);
        }

        typedef void(*GemmNN_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B0, float* C);

        static void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C)
        {
            size_t m = 32, m1 = M, mm = AlignLo(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.aK * DF, dC = a.cN;
            GemmNN_Ptr tail_2 = t > 16 ? InnerProduct16bGemmNN_2x2<true> : InnerProduct16bGemmNN_1x2;
            GemmNN_Ptr tail_1 = t > 16 ? InnerProduct16bGemmNN_2x1 : InnerProduct16bGemmNN_1x1;

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                if (dN > F)
                {
                    for (; i < mm; i += m)
                        InnerProduct16bGemmNN_2x2<true>(A + i * dA, p, a, m, dN, K, update, B, C + i * dC);
                    if (t)
                        tail_2(A + i * dA, p, a, t, dN, K, update, B, C + i * dC);
                }
                else
                {
                    for (; i < mm; i += m)
                        InnerProduct16bGemmNN_2x1(A + i * dA, p, a, m, dN, K, update, B, C + i * dC);
                    if (t)
                        tail_1(A + i * dA, p, a, t, dN, K, update, B, C + i * dC);
                }
                B += dB;
                C += dN;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Avx512bw::SynetInnerProduct16bGemmNN(p)
        {
            SetAlgParam(F, F * 2, F * 2, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData32f)
                    _prepA = InnerProduct16bGemmNN_ConvertA;
            }
            _gemm = InnerProduct16bGemmNN_Gemm2;
        }
    }
#endif
}
