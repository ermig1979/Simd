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
#include "Simd/SimdMinMax.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdMath.h"

#include <cstring>

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::GemmLastPtr GemmLastPtr;

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

        static SIMD_INLINE void QuantA(const float* src, size_t K, float* scale, int32_t* zero, __m512& norm, __m512i& z)
        {
            size_t K16 = AlignLo(K, 16), k = 0;
            __m512 _min = _mm512_setzero_ps();
            __m512 _max = _mm512_setzero_ps();
            for (; k < K16; k += 16)
            {
                __m512 s = _mm512_loadu_ps(src + k);
                _min = _mm512_min_ps(s, _min);
                _max = _mm512_max_ps(s, _max);
            }
            if (k < K)
            {
                __m512 s = _mm512_maskz_loadu_ps(TailMask16(K - k), src + k);
                _min = _mm512_min_ps(s, _min);
                _max = _mm512_max_ps(s, _max);
            }
            __m128 _min4 = MinVal32f(_min);
            __m128 _max4 = MaxVal32f(_max);
            __m128 _1 = _mm_set1_ps(1.0f), _255 = _mm_set1_ps(255.0f);
            __m128 _scale = Sse41::Combine(_mm_cmpeq_ss(_min4, _max4), _1, _mm_div_ss(_mm_sub_ps(_max4, _min4), _255));
            _mm_store_ss(scale, _scale);
            z = _mm512_broadcastd_epi32(_mm_cvtps_epi32(_mm_max_ss(_mm_setzero_ps(),
                _mm_min_ss(_mm_sub_ss(_mm_setzero_ps(), _mm_div_ss(_min4, _scale)), _255))));
            _mm_storeu_si32(zero, _mm512_castsi512_si128(z));
            norm = _mm512_broadcastss_ps(_mm_div_ss(_1, _scale));
        }

        void Dqip16bGemmV0_ConvertAf(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const float* src0 = (float*)src8;
            size_t K = p.K, K16 = AlignLo(K, 16), K64 = AlignLo(K, 64), KA = a.aK;
            for (size_t i = 0; i < M; i += 16)
            {
                const float* src = src0 + i * K;
                size_t m = Min(i + 16, M) - i;
                __m512 norm[16];
                __m512i z[16];
                for (size_t j = 0; j < m; ++j)
                    QuantA(src + j * K, K, scale + i + j, zero + i + j, norm[j], z[j]);
                size_t k = 0;
                uint8_t* pd = dst;
                for (; k < K64; k += 64, pd += 1024)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear64(src + j * K + k, norm[j], z[j], pd + j * 64);
                    for (; j < 16; ++j)
                        SetZero(pd + j * 64, _mm512_setzero_si512());
                }
                for (; k < K16; k += 16, pd += 16)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + j * K + k, norm[j], z[j], pd + j * 64);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < K; k += 16, pd += 16)
                {
                    __mmask16 tailK = TailMask16(K - k);
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + j * K + k, norm[j], z[j], pd + j * 64, tailK);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < KA; k += 16, pd += 16)
                {
                    for (size_t j = 0; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                dst += a.aK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        static SIMD_INLINE void QuantizeLinear16(const uint16_t* src, __m512 norm, __m512i zero, uint8_t* dst, __mmask16 tail = -1)
        {
            __m512i i0 = QuantizeLinear(BFloat16ToFloat32(_mm256_maskz_loadu_epi16(tail, src)), norm, zero);
            _mm_storeu_si128((__m128i*)dst, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i0, K_ZERO), K_ZERO)));
        }

        static SIMD_INLINE void QuantizeLinear64(const uint16_t* src, __m512 norm, __m512i zero, uint8_t* dst)
        {
            __m512i i0 = QuantizeLinear(BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)src + 0)), norm, zero);
            __m512i i1 = QuantizeLinear(BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)src + 1)), norm, zero);
            __m512i i2 = QuantizeLinear(BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)src + 2)), norm, zero);
            __m512i i3 = QuantizeLinear(BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)src + 3)), norm, zero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
        }

        static SIMD_INLINE void QuantAb(const uint16_t* src, size_t K, float* scale, int32_t* zero, __m512& norm, __m512i& z)
        {
            size_t K16 = AlignLo(K, 16), k = 0;
            __m512 _min = _mm512_setzero_ps();
            __m512 _max = _mm512_setzero_ps();
            for (; k < K16; k += 16)
            {
                __m512 s = BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)(src + k)));
                _min = _mm512_min_ps(s, _min);
                _max = _mm512_max_ps(s, _max);
            }
            if (k < K)
            {
                __m512 s = BFloat16ToFloat32(_mm256_maskz_loadu_epi16(TailMask16(K - k), src + k));
                _min = _mm512_min_ps(s, _min);
                _max = _mm512_max_ps(s, _max);
            }
            __m128 _min4 = MinVal32f(_min);
            __m128 _max4 = MaxVal32f(_max);
            __m128 _1 = _mm_set1_ps(1.0f), _255 = _mm_set1_ps(255.0f);
            __m128 _scale = Sse41::Combine(_mm_cmpeq_ss(_min4, _max4), _1, _mm_div_ss(_mm_sub_ps(_max4, _min4), _255));
            _mm_store_ss(scale, _scale);
            z = _mm512_broadcastd_epi32(_mm_cvtps_epi32(_mm_max_ss(_mm_setzero_ps(),
                _mm_min_ss(_mm_sub_ss(_mm_setzero_ps(), _mm_div_ss(_min4, _scale)), _255))));
            _mm_storeu_si32(zero, _mm512_castsi512_si128(z));
            norm = _mm512_broadcastss_ps(_mm_div_ss(_1, _scale));
        }

        void Dqip16bGemmV0_ConvertAb(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const uint16_t* src0 = (uint16_t*)src8;
            size_t K = p.K, K16 = AlignLo(K, 16), K64 = AlignLo(K, 64), KA = a.aK;
            for (size_t i = 0; i < M; i += 16)
            {
                const uint16_t* src = src0 + i * K;
                size_t m = Min(i + 16, M) - i;
                __m512 norm[16];
                __m512i z[16];
                for (size_t j = 0; j < m; ++j)
                    QuantAb(src + j * K, K, scale + i + j, zero + i + j, norm[j], z[j]);
                size_t k = 0;
                uint8_t* pd = dst;
                for (; k < K64; k += 64, pd += 1024)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear64(src + j * K + k, norm[j], z[j], pd + j * 64);
                    for (; j < 16; ++j)
                        SetZero(pd + j * 64, _mm512_setzero_si512());
                }
                for (; k < K16; k += 16, pd += 16)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + j * K + k, norm[j], z[j], pd + j * 64);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < K; k += 16, pd += 16)
                {
                    __mmask16 tailK = TailMask16(K - k);
                    size_t j = 0;
                    for (; j < m; ++j)
                        QuantizeLinear16(src + j * K + k, norm[j], z[j], pd + j * 64, tailK);
                    for (; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                for (; k < KA; k += 16, pd += 16)
                {
                    for (size_t j = 0; j < 16; ++j)
                        Sse41::SetZero(pd + j * 64);
                }
                dst += a.aK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        static void Dqip16bGemmV0_GemmBody2x2(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
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

        static void Dqip16bGemmV0_GemmBody2x1(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
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

        static void Dqip16bGemmV0_GemmBody1x2(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
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

        static void Dqip16bGemmV0_GemmBody1x1(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.aK, strideS = dS * 4, strideB = 64;
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

        typedef void(*Dqip16bGemmV0_GemmBodyPtr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, int32_t* sum);

        static void Dqip16bGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            size_t n = 32;
            size_t Mn = AlignLo(M, n), m = AlignHi(M - Mn, 16);
            size_t dS = a.cN, dA = a.aK;

            Dqip16bGemmV0_GemmBodyPtr body_2 = Dqip16bGemmV0_GemmBody2x2;
            Dqip16bGemmV0_GemmBodyPtr tail_2 = m > 16 ? Dqip16bGemmV0_GemmBody2x2 : Dqip16bGemmV0_GemmBody1x2;
            Dqip16bGemmV0_GemmBodyPtr body_1 = Dqip16bGemmV0_GemmBody2x1;
            Dqip16bGemmV0_GemmBodyPtr tail_1 = m > 16 ? Dqip16bGemmV0_GemmBody2x1 : Dqip16bGemmV0_GemmBody1x1;
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

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply> SIMD_INLINE void Dqip16bGemmV0_GemmLast1xMx2(
            const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const int32_t* aZero, const __m512i* bSums,
            const float* aScale, const __m512* bScale, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, int32_t* buf2, uint8_t* C, __mmask16 tailN)
        {
            int dC = int(p.N * a.eC), dA = (int)a.aK, dB = 16, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.aK * F;

            if (update)
            {
                int dS = (int)a.cN, strideS = dS * 4;
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
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                _tile_stream_loadd(5, A1, strideA);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                if (N > 0) _tile_dpbusd(2, 5, 6);
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(3, 5, 7);
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
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
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
            buf2 += 16 * DF;
            if (N > 0) _tile_dpbusd(2, 5, 6);
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(2, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(3, 5, 7);
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(3, buf2 + F, DA);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply> SIMD_INLINE void Dqip16bGemmV0_GemmLast1xMx1(
            const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const int32_t* aZero, const __m512i* bSums,
            const float* aScale, const __m512* bScale, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, int32_t* buf2, uint8_t* C, __mmask16 tailN)
        {
            int dC = int(p.N * a.eC), dA = (int)a.aK, dB = 16, strideB = 64;
            int stepA = a.reorder ? 1024 : 64, strideA = a.reorder ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.aK * F;

            if (update)
            {
                int dS = (int)a.cN, strideS = dS * 4;
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
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
                ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
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
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (apply) ApplyMxN<term, type, N, apply>(C + i * dC, dC, buf1 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply> void Dqip16bGemmV0_GemmLastNxMx2(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t M, size_t K, int update, const int8_t* B0, const int32_t* aZero, const __m512i* bSums,
            const float* aScale, const __m512* bScale, const __m512* bias, const __m512* params, int32_t* sum0, int32_t* buf1, uint8_t* C, __mmask16 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.aK;
            int32_t* buf2 = buf1 + 1024;
            size_t ci = 0, pi = 0;
            if(M > 16)
                Dqip16bGemmV0_GemmLast1xMx2<term, type, N, 0>(A0, p, a, K, update, B0,
                    aZero, bSums, aScale, bScale, bias, params, sum0, NULL, buf2, C, tailN), ci += 32;
            else
                Dqip16bGemmV0_GemmLast1xMx1<term, type, N, 0>(A0, p, a, K, update, B0,
                    aZero, bSums, aScale, bScale, bias, params, sum0, NULL, buf2, C, tailN), ci += 16;
            for (; ci < M; pi = ci, ci += 32)
            {
                Swap(buf1, buf2);
                if (ci + 16 >= M)
                    Dqip16bGemmV0_GemmLast1xMx1<term, type, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        aZero + pi, bSums, aScale + pi, bScale, bias, params, sum0 + ci * dS, buf1, buf2, C + pi * dC, tailN);
                else
                    Dqip16bGemmV0_GemmLast1xMx2<term, type, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        aZero + pi, bSums, aScale + pi, bScale, bias, params, sum0 + ci * dS, buf1, buf2, C + pi * dC, tailN);
            }
            C += pi * dC;
            aZero += pi;
            aScale += pi;
            M -= pi;
            size_t i = 0, M8 = M & (~7);
            for (; i < M8; i += 8)
                ApplyMxN<term, type, N, 8>(C + i * dC, dC, buf2 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN);
            for (; i < M; ++i)
                ApplyMxN<term, type, N, 1>(C + i * dC, dC, buf2 + i * DF, aZero + i, bSums, aScale + i, bScale, bias, params, tailN);
        }

        //--------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply> void Dqip16bGemmV0_GemmLast(const uint8_t* A, 
            const DqipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C)
        {
            size_t n = 32;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dC = p.N * a.eC, dA = a.aK;
            __m512i _bSums[2];
            __m512 _bScale[2], _bias[2], _params[2];
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
                _bSums[0] = _mm512_loadu_si512((__m512i*)(bSums + j) + 0);
                _bSums[1] = _mm512_loadu_si512((__m512i*)(bSums + j) + 1);
                _bScale[0] = _mm512_loadu_ps(bScale + j + 0);
                _bScale[1] = _mm512_loadu_ps(bScale + j + F);
                _bias[0] = _mm512_loadu_ps(bias + j + 0);
                _bias[1] = _mm512_loadu_ps(bias + j + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + j + 0);
                    _params[1] = _mm512_loadu_ps(params + j + F);
                }
                size_t i = 0;
                if (dN > F)
                    Dqip16bGemmV0_GemmLastNxMx2<term, type, 2, apply>(A + i * dA, p, a, M, K, update, B, 
                        aZero + i, _bSums, aScale + i, _bScale, _bias, _params, sum + i * dS, buf, C + i * dC, tailN);
                else
                    Dqip16bGemmV0_GemmLastNxMx2<term, type, 1, apply>(A + i * dA, p, a, M, K, update, B, 
                        aZero + i, _bSums, aScale + i, _bScale, _bias, _params, sum + i * dS, buf, C + i * dC, tailN);
                B += a.aK * DF;
                sum += DF;
                C += DF * a.eC;
            }
        }

        template<Term16bType term, int apply> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationIdentity, apply>; break;
            case SimdConvolutionActivationRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRelu, apply>; break;
            case SimdConvolutionActivationLeakyRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationLeakyRelu, apply>; break;
            case SimdConvolutionActivationRestrictRange: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRestrictRange, apply>; break;
            case SimdConvolutionActivationPrelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationPrelu, apply>; break;
            case SimdConvolutionActivationElu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationElu, apply>; break;
            case SimdConvolutionActivationHswish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHswish, apply>; break;
            case SimdConvolutionActivationMish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationMish, apply>; break;
            case SimdConvolutionActivationHardSigmoid: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHardSigmoid, apply>; break;
            case SimdConvolutionActivationSwish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationSwish, apply>; break;
            case SimdConvolutionActivationGelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationGelu, apply>; break;
            default:
                assert(0);
            }
        }

        template<int apply> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            if (p.typeC == SimdTensorData16b)
                SetGemmLast<Term16bLast16b, apply>(p, gemmLast);
            else if (p.typeC == SimdTensorData32f)
                SetGemmLast<Term16bLast32f, apply>(p, gemmLast);
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

        SynetDynamicQuantizedInnerProduct16bGemmV0::SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p)
            : Avx512vnni::SynetDynamicQuantizedInnerProduct16bGemmV0(p)
        {
            if (p.M == 1)
                return;
            SetAlgParam(F, F * 2, F * 2, 64, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5f), Base::AlgCacheL3());
            _alg.overflow = 0;
            _alg.reorder = 1;
            _alg.cN = AlignHi(Max(_alg.aN, _alg.macroN), DF);
            _sizeC = _alg.aM * _alg.cN;
            if (p.typeA == SimdTensorData32f)
                _convertA = Dqip16bGemmV0_ConvertAf;
            else if (p.typeA == SimdTensorData16b)
                _convertA = Dqip16bGemmV0_ConvertAb;
            _gemmBody = Dqip16bGemmV0_GemmBody;
            SetGemmLast(p, _alg, _gemmLast);
        }
    }
#endif
}
