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
#include "Simd/SimdTile.h"
#include "Simd/SimdSet.h"
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

        static SIMD_INLINE void QuantizeLinear16(const float* src, __m512 norm, __m512i zero, uint8_t* dst, __mmask16 tail = -1)
        {
            __m512i i0 = QuantizeLinear(_mm512_maskz_loadu_ps(tail, src), norm, zero);
            _mm_storeu_si128((__m128i*)dst, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i0, K_ZERO), K_ZERO)));
        }

        static SIMD_INLINE void QuantizeLinear64(const float* src, __m512 norm, __m512i zero, uint8_t* dst)
        {
            __m512i i0 = QuantizeLinear(_mm512_loadu_ps(src + 0 * F), norm, zero);
            __m512i i1 = QuantizeLinear(_mm512_loadu_ps(src + 1 * F), norm, zero);
            __m512i i2 = QuantizeLinear(_mm512_loadu_ps(src + 2 * F), norm, zero);
            __m512i i3 = QuantizeLinear(_mm512_loadu_ps(src + 3 * F), norm, zero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
        }

        void SynetDynamicQuantizedInnerProductGemmV0_Convert(const float* src, float norm, uint8_t zero, const DqipParam& p, const AlgParam& a, size_t M, uint8_t* dst)
        {
            size_t K = p.K, K16 = AlignLo(K, 16), K64 = AlignLo(K, 64), KA = a.aK;
            __m512 _norm = _mm512_set1_ps(norm);
            __m512i _zero = _mm512_set1_epi32(zero);
            __mmask16 tailK = TailMask16(K - K16);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
                size_t k = 0;
                uint8_t* pd = dst;
                for (; k < K64; k += 64, pd += 1024)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear64(src + k + j * K, _norm, _zero, pd + j * 64);
                    for (; j < 16; ++j)
                        SetZero(pd + j * 64, _mm512_setzero_si512());
                }
                for (; k < K16; k += 16, pd += 16)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + k + j * K, _norm, _zero, pd + j * 64);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < K; k += 16, pd += 16)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + k + j * K, _norm, _zero, pd + j * 64, tailK);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < KA; k += 16, pd += 16)
                {
                    size_t j = 0;
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                src += K * 16;
                dst += a.aK * 16;
            }
        }

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

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int N, int apply> SIMD_INLINE void DqipGemmV0_GemmLast1xMx2(
            const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const __m512i* corr, 
            const __m512* norm, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, int32_t* buf2, float* C, __mmask16 tailN)
        {
            int dC = int(p.N), dA = (int)a.aK, dB = 16, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.aK * F;

            if (update)
            {
                int dS = (int)a.sN, strideS = dS * 4;
                if (N > 0) _tile_stream_loadd(0, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(1, sum0 + F, strideS);
                sum0 += 16 * dS;
                if (N > 0) _tile_stream_loadd(2, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(3, sum0 + F, strideS);
            }
            else
            {
                if (N > 0) _tile_zero(0);
                if (N > 1) _tile_zero(1);
                if (N > 0) _tile_zero(2);
                if (N > 1) _tile_zero(3);
            }

            int K64 = (int)K - 64, aK64 = apply ? (8 * 64 / apply - 64) : 0, k = 0, i = 0;

            _tile_stream_loadd(4, A0, strideA);
            if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
            for (; k < aK64; A1 += stepA)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                _tile_stream_loadd(5, A1, strideA);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                if (N > 0) _tile_dpbusd(2, 5, 6);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(3, 5, 7);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            }
            for (; k < K64; A1 += stepA)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                _tile_stream_loadd(5, A1, strideA);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                if (N > 0) _tile_dpbusd(2, 5, 6);
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(3, 5, 7);
            }
            if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
            _tile_stream_loadd(5, A1, strideA);
            if (N > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
            buf2 += 16 * DF;
            if (N > 0) _tile_dpbusd(2, 5, 6);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(2, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(3, 5, 7);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(3, buf2 + F, DA);
        }

        template<SimdConvolutionActivationType type, int N, int apply> SIMD_INLINE void DqipGemmV0_GemmLast1xMx1(
            const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const __m512i* corr,
            const __m512* norm, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, int32_t* buf2, float* C, __mmask16 tailN)
        {
            int dC = int(p.N), dA = (int)a.aK, dB = 16, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.aK * F;

            if (update)
            {
                int dS = (int)a.sN, strideS = dS * 4;
                if (N > 0) _tile_stream_loadd(0, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(1, sum0 + F, strideS);
            }
            else
            {
                if (N > 0) _tile_zero(0);
                if (N > 1) _tile_zero(1);
            }

            int K64 = (int)K - 64, aK64 = apply ? (8 * 64 / apply - 64) : 0, k = 0, i = 0;

            _tile_stream_loadd(4, A0, strideA);
            if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
            for (; k < aK64;)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
            }
            for (; k < K64;)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
            }
            if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
            if (N > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (apply) ApplyMxN<type, N, apply>(C + i * dC, dC, buf1 + i * DF, corr, norm, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
        }

        template<SimdConvolutionActivationType type, int N, int apply> void DqipGemmV0_GemmLastNxMx2(const uint8_t* A0, 
            const DqipParam& p, const AlgParam& a, size_t M, size_t K, int update, const int8_t* B0, const __m512i* corr,
            const __m512* norm, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, float* C, __mmask16 tailN)
        {
            int dS = (int)a.sN, dC = (int)p.N, dA = (int)a.aK;
            int32_t* buf2 = buf1 + 1024;

            size_t ci = 0, pi = 0;
            DqipGemmV0_GemmLast1xMx2<type, N, 0>(A0, p, a, K, update, B0,
                corr, norm, bias, params, sum0, NULL, buf2, C, tailN), ci += 32;
            for (; ci < M; pi = ci, ci += 32)
            {
                Swap(buf1, buf2);
                if (ci + 16 >= M)
                    DqipGemmV0_GemmLast1xMx1<type, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        corr, norm, bias, params, sum0 + ci * dS, buf1, buf2, C + pi * dC, tailN);
                else
                    DqipGemmV0_GemmLast1xMx2<type, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        corr, norm, bias, params, sum0 + ci * dS, buf1, buf2, C + pi * dC, tailN);
            }
            float* C1 = C + pi * dC;
            M -= pi;
            size_t i = 0, M8 = M & (~7);
            for (; i < M8; i += 8)
                ApplyMxN<type, N, 8>(C1 + i * dC, dC, buf2 + i * DF, corr, norm, bias, params, tailN);
            for (; i < M; ++i)
                ApplyMxN<type, N, 1>(C1 + i * dC, dC, buf2 + i * DF, corr, norm, bias, params, tailN);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int apply> void DqipGemmV0_GemmLast(const uint8_t* A, const DqipParam& p, 
            const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf, 
            const int32_t* corr, const float* norm, const float* bias, const float* params, float* C)
        {
            size_t n = 32;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.sN, dC = p.N, dA = a.aK;
            __m512i _corr[2];
            __m512 _norm[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }

            SetTileConfFull();
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                __mmask16 tailN = dN > F ? TailMask16(dN - F) : TailMask16(dN);
                _corr[0] = _mm512_loadu_si512((__m512i*)(corr + j) + 0);
                _corr[1] = _mm512_loadu_si512((__m512i*)(corr + j) + 1);
                _norm[0] = _mm512_loadu_ps(norm + j + 0);
                _norm[1] = _mm512_loadu_ps(norm + j + F);
                _bias[0] = _mm512_loadu_ps(bias + j + 0);
                _bias[1] = _mm512_loadu_ps(bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + j + 0);
                    _params[1] = _mm512_loadu_ps(params + j + F);
                }
                size_t i = 0;
                if (dN > F)
                    DqipGemmV0_GemmLastNxMx2<type, 2, apply>(A + i * dA, p, a, M, K, update, B, _corr, _norm, _bias, _params, sum + i * dS, buf, C + i * dC, tailN);
                else
                    DqipGemmV0_GemmLastNxMx2<type, 1, apply>(A + i * dA, p, a, M, K, update, B, _corr, _norm, _bias, _params, sum + i * dS, buf, C + i * dC, tailN);
                B += a.aK * DF;
                sum += DF;
                C += DF;
            }
        }

        template<int apply> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationIdentity, apply>; break;
            case SimdConvolutionActivationRelu: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationRelu, apply>; break;
            case SimdConvolutionActivationLeakyRelu: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationLeakyRelu, apply>; break;
            case SimdConvolutionActivationRestrictRange: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationRestrictRange, apply>; break;
            case SimdConvolutionActivationPrelu: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationPrelu, apply>; break;
            case SimdConvolutionActivationElu: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationElu, apply>; break;
            case SimdConvolutionActivationHswish: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationHswish, apply>; break;
            case SimdConvolutionActivationMish: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationMish, apply>; break;
            case SimdConvolutionActivationHardSigmoid: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationHardSigmoid, apply>; break;
            case SimdConvolutionActivationSwish: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationSwish, apply>; break;
            case SimdConvolutionActivationGelu: gemmLast = DqipGemmV0_GemmLast<SimdConvolutionActivationGelu, apply>; break;
            default:
                assert(0);
            }
        }

        void SetGemmLast(const DqipParam& p, const AlgParam& a, GemmLastPtr& gemmLast)
        {
            size_t K = p.K - AlignLoAny(p.K, a.macroK);
            if (K > 448)
                SetGemmLast<1>(p, gemmLast);
            else if (K > 192)
                SetGemmLast<2>(p, gemmLast);
            else if (K > 64)
                SetGemmLast<4>(p, gemmLast);
            else
                SetGemmLast<8>(p, gemmLast);
        }

        //-------------------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProductGemmV0::SynetDynamicQuantizedInnerProductGemmV0(const DynamicQuantizedInnerProductParam& p)
            : Avx512vnni::SynetDynamicQuantizedInnerProductGemmV0(p)
        {
            if (p.M == 1)
                return;
            SetAlgParam(F, F * 2, F * 2, 64, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5f), Base::AlgCacheL3());
            _alg.reorder = 1;
            _convert = SynetDynamicQuantizedInnerProductGemmV0_Convert;
            _gemmBody = DqipGemmV0_GemmBody;
            SetGemmLast(p, _alg, _gemmLast);
        }
    }
#endif
}
