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
#include "Simd/SimdTile.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        static void DqipGemmV0_GemmBody2x2(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.sN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.aK * F;
            int32_t* sum1 = sum0 + 16 * dS;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(1, sum0 + F, strideS);
                _tile_stream_loadd(2, sum1 + 0, strideS);
                _tile_stream_loadd(3, sum1 + F, strideS);
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

            _tile_stored(0, sum0 + 0, strideS);
            TileMoveToMemory(sum0 + 0, dS);
            _tile_stored(1, sum0 + F, strideS);
            TileMoveToMemory(sum0 + F, dS);
            _tile_stored(2, sum1 + 0, strideS);
            TileMoveToMemory(sum1 + 0, dS);
            _tile_stored(3, sum1 + F, strideS);
            TileMoveToMemory(sum1 + F, dS);
        }

        static void DqipGemmV0_GemmBody2x1(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.sN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            int32_t* sum1 = sum0 + 16 * dS;

            if (update)
            {
                _tile_stream_loadd(0, sum0, strideS);
                _tile_stream_loadd(2, sum1, strideS);
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

            _tile_stored(0, sum0, strideS);
            TileMoveToMemory(sum0, dS);
            _tile_stored(2, sum1, strideS);
            TileMoveToMemory(sum1, dS);
        }

        static void DqipGemmV0_GemmBody1x2(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.sN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const int8_t* B1 = B0 + a.aK * F;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(1, sum0 + F, strideS);
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

            _tile_stored(0, sum0 + 0, strideS);
            TileMoveToMemory(sum0 + 0, dS);
            _tile_stored(1, sum0 + F, strideS);
            TileMoveToMemory(sum0 + F, dS);
        }

        static void DqipGemmV0_GemmBody1x1(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.sN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideS);
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

            _tile_stored(0, sum0 + 0, strideS);
            TileMoveToMemory(sum0 + 0, dS);
        }

        typedef void(*DqipGemmV0_GemmBodyPtr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum);

        static void DqipGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            size_t n = 32;
            size_t Mn = AlignLo(M, n), m = AlignHi(M - Mn, 16);
            size_t dS = a.sN, dA = a.aK;

            DqipGemmV0_GemmBodyPtr body_2 = DqipGemmV0_GemmBody2x2;
            DqipGemmV0_GemmBodyPtr tail_2 = m > 16 ? DqipGemmV0_GemmBody2x2 : DqipGemmV0_GemmBody1x2;
            DqipGemmV0_GemmBodyPtr body_1 = DqipGemmV0_GemmBody2x1;
            DqipGemmV0_GemmBodyPtr tail_1 = m > 16 ? DqipGemmV0_GemmBody2x1 : DqipGemmV0_GemmBody1x1;
            SetTileConfFull();
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                if (dN > F)
                {
                    for (; i < Mn; i += n)
                        body_2(A + i * dA, p, a, K, update, B, sum + i * dS);
                    if (m)
                        tail_2(A + Mn * dA, p, a, K, update, B, sum + i * dS);
                }
                else
                {
                    for (; i < Mn; i += n)
                        body_1(A + i * dA, p, a, K, update, B, sum + i * dS);
                    if (m)
                        tail_1(A + Mn * dA, p, a, K, update, B, sum + i * dS);
                }
                B += a.aK * DF;
                sum += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProductGemmV0::SynetDynamicQuantizedInnerProductGemmV0(const DynamicQuantizedInnerProductParam& p)
            : Avx512vnni::SynetDynamicQuantizedInnerProductGemmV0(p)
        {
            SetAlgParam(F, F * 2, F * 2, 64, Base::AlgCacheL1(), int(Base::AlgCacheL2()*0.5f), Base::AlgCacheL3());
            _gemmBody = DqipGemmV0_GemmBody;
        }
    }
#endif
}
