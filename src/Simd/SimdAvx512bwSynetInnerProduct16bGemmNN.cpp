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
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Avx512bw
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
                        Avx512bw::Float32ToBFloat16(src + k, dst + k);
                    if(tail)
                        Avx512bw::Float32ToBFloat16(src + k, dst + k, tail);
                    src += p.K;
                    dst += a.aK;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ReorderA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t KDF = Simd::AlignLo(p.K, DF);
            __mmask32 tail = TailMask32(p.K - KDF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KDF; k += DF)
                    Avx512bw::Copy(src + k, dst + k);
                if (tail)
                    Avx512bw::Copy(src + k, dst + k, tail);
                src += p.K;
                dst += a.aK;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBn(const float* src, size_t stride, uint16_t* dst)
        {
            __m512i d0 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(_mm512_loadu_ps(src + 0 * stride)), Bf16::ROUND), Base::Bf16::SHIFT);
            __m512i d1 = _mm512_and_si512(_mm512_add_epi32(_mm512_castps_si512(_mm512_loadu_ps(src + 1 * stride)), Bf16::ROUND), Bf16::MASK);
            _mm512_storeu_si512((__m512i*)dst, _mm512_or_si512(d0, d1));
        }

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ConvertBn(ps + f, p.N, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        //SIMD_INLINE void ConvertBt(const float* src, size_t stride, uint16_t* dst)
        //{
        //    __m256i d01 = Float32ToBFloat16(Load(src + 0 * stride, src + 1 * stride, src + 4 * stride, src + 5 * stride));
        //    __m256i d23 = Float32ToBFloat16(Load(src + 2 * stride, src + 3 * stride, src + 6 * stride, src + 7 * stride));
        //    _mm256_storeu_si256((__m256i*)dst, _mm256_packus_epi32(d01, d23));
        //}

        //static void InnerProduct16bGemmNN_ConvertBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        //{
        //    const float* src = (float*)src8;
        //    size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0;
        //    for (; j < Nf; j += a.F)
        //    {
        //        size_t k = 0;
        //        for (; k < Kl; k += 2)
        //        {
        //            const float* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; f += F, dst += DF)
        //                ConvertBt(ps + f * p.K, p.K, dst);
        //        }
        //        for (; k < Kh; k += 2)
        //        {
        //            const float* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; ++f)
        //            {
        //                for (size_t i = 0; i < 2; ++i)
        //                {
        //                    if (j + f < p.N && k + i < p.K)
        //                        *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
        //                    else
        //                        *(dst++) = 0;
        //                }
        //            }
        //        }
        //    }
        //    for (; j < N; j += a.F)
        //    {
        //        for (size_t k = 0; k < Kh; k += 2)
        //        {
        //            const float* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; ++f)
        //            {
        //                for (size_t i = 0; i < 2; ++i)
        //                {
        //                    if (j + f < p.N && k + i < p.K)
        //                        *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
        //                    else
        //                        *(dst++) = 0;
        //                }
        //            }
        //        }
        //    }
        //}

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderBn(const uint16_t* src, size_t stride, uint16_t* dst)
        {
            static const __m512i PERM_IDX = _mm512_set_epi16(
                0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
                0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
            __m512i s01 = Load<false>((__m256i*)(src + 0 * stride), (__m256i*)(src + 1 * stride));
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, s01));
        }

        static void InnerProduct16bGemmNN_ReorderBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ReorderBn(ps + f, p.N, dst);
                }
                for (; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        //SIMD_INLINE void ReorderBt(const uint16_t* src, size_t stride, uint16_t* dst)
        //{
        //    *(uint32_t*)(dst + 0x0) = *(uint32_t*)(src + 0 * stride);
        //    *(uint32_t*)(dst + 0x2) = *(uint32_t*)(src + 1 * stride);
        //    *(uint32_t*)(dst + 0x4) = *(uint32_t*)(src + 2 * stride);
        //    *(uint32_t*)(dst + 0x6) = *(uint32_t*)(src + 3 * stride);
        //    *(uint32_t*)(dst + 0x8) = *(uint32_t*)(src + 4 * stride);
        //    *(uint32_t*)(dst + 0xa) = *(uint32_t*)(src + 5 * stride);
        //    *(uint32_t*)(dst + 0xc) = *(uint32_t*)(src + 6 * stride);
        //    *(uint32_t*)(dst + 0xe) = *(uint32_t*)(src + 7 * stride);
        //}

        //static void InnerProduct16bGemmNN_ReorderBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        //{
        //    const uint16_t* src = (uint16_t*)src8;
        //    size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0;
        //    for (; j < Nf; j += a.F)
        //    {
        //        size_t k = 0;
        //        for (; k < Kl; k += 2)
        //        {
        //            const uint16_t* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; f += F, dst += DF)
        //                ReorderBt(ps + f * p.K, p.K, dst);
        //        }
        //        for (; k < Kh; k += 2)
        //        {
        //            const uint16_t* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; ++f)
        //            {
        //                for (size_t i = 0; i < 2; ++i)
        //                {
        //                    if (j + f < p.N && k + i < p.K)
        //                        *(dst++) = ps[f * p.K + i];
        //                    else
        //                        *(dst++) = 0;
        //                }
        //            }
        //        }
        //    }
        //    for (; j < N; j += a.F)
        //    {
        //        for (size_t k = 0; k < Kh; k += 2)
        //        {
        //            const uint16_t* ps = src + j * p.K + k;
        //            for (size_t f = 0; f < a.F; ++f)
        //            {
        //                for (size_t i = 0; i < 2; ++i)
        //                {
        //                    if (j + f < p.N && k + i < p.K)
        //                        *(dst++) = ps[f * p.K + i];
        //                    else
        //                        *(dst++) = 0;
        //                }
        //            }
        //        }
        //    }
        //}

        //-----------------------------------------------------------------------------------------

        template<int M> void InnerProduct16bGemmNN_2xM(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, 
            size_t N, size_t K, int update, const uint16_t* B0, float* C)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71,
                c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b00, b01, b10, b11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dC = a.cN, dA = a.aK;
            const uint16_t* B1 = B0 + a.aK * F;
            const uint16_t* A1 = A0 + 1 * dA;
            const uint16_t* A2 = A0 + 2 * dA;
            const uint16_t* A3 = A0 + 3 * dA;
            const uint16_t* A4 = A0 + 4 * dA;
            const uint16_t* A5 = A0 + 5 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_ps(C + 0x0 * dC + 0), c01 = _mm512_loadu_ps(C + 0x0 * dC + F);
                    if (M > 0x1) c10 = _mm512_loadu_ps(C + 0x1 * dC + 0), c11 = _mm512_loadu_ps(C + 0x1 * dC + F);
                    if (M > 0x2) c20 = _mm512_loadu_ps(C + 0x2 * dC + 0), c21 = _mm512_loadu_ps(C + 0x2 * dC + F);
                    if (M > 0x3) c30 = _mm512_loadu_ps(C + 0x3 * dC + 0), c31 = _mm512_loadu_ps(C + 0x3 * dC + F);
                    if (M > 0x4) c40 = _mm512_loadu_ps(C + 0x4 * dC + 0), c41 = _mm512_loadu_ps(C + 0x4 * dC + F);
                    if (M > 0x5) c50 = _mm512_loadu_ps(C + 0x5 * dC + 0), c51 = _mm512_loadu_ps(C + 0x5 * dC + F);
                    if (M > 0x6) c60 = _mm512_loadu_ps(C + 0x6 * dC + 0), c61 = _mm512_loadu_ps(C + 0x6 * dC + F);
                    if (M > 0x7) c70 = _mm512_loadu_ps(C + 0x7 * dC + 0), c71 = _mm512_loadu_ps(C + 0x7 * dC + F);
                    if (M > 0x8) c80 = _mm512_loadu_ps(C + 0x8 * dC + 0), c81 = _mm512_loadu_ps(C + 0x8 * dC + F);
                    if (M > 0x9) c90 = _mm512_loadu_ps(C + 0x9 * dC + 0), c91 = _mm512_loadu_ps(C + 0x9 * dC + F);
                    if (M > 0xa) ca0 = _mm512_loadu_ps(C + 0xa * dC + 0), ca1 = _mm512_loadu_ps(C + 0xa * dC + F);
                    if (M > 0xb) cb0 = _mm512_loadu_ps(C + 0xb * dC + 0), cb1 = _mm512_loadu_ps(C + 0xb * dC + F);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                    if (M > 0x1) c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                    if (M > 0x2) c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                    if (M > 0x3) c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                    if (M > 0x4) c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                    if (M > 0x5) c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
                    if (M > 0x6) c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
                    if (M > 0x7) c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
                    if (M > 0x8) c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
                    if (M > 0x9) c90 = _mm512_setzero_ps(), c91 = _mm512_setzero_ps();
                    if (M > 0xa) ca0 = _mm512_setzero_ps(), ca1 = _mm512_setzero_ps();
                    if (M > 0xb) cb0 = _mm512_setzero_ps(), cb1 = _mm512_setzero_ps();
                }
                for (size_t k0 = 0, k6 = k0 + 6 * dA; k0 < K; k0 += 2, k6 += 2)
                {
                    b01 = _mm512_loadu_ps((float*)B0);
                    b00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b01), Base::Bf16::SHIFT));
                    b01 = _mm512_and_ps(b01, m);
                    b11 = _mm512_loadu_ps((float*)B1);
                    b10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b11), Base::Bf16::SHIFT));
                    b11 = _mm512_and_ps(b11, m);
                    if (M > 0x0)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k0 - 1)), m);
                        c00 = _mm512_fmadd_ps(a0, b00, c00);
                        c01 = _mm512_fmadd_ps(a0, b10, c01);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k0 - 0)), m);
                        c00 = _mm512_fmadd_ps(a0, b01, c00);
                        c01 = _mm512_fmadd_ps(a0, b11, c01);
                    }
                    if (M > 0x1)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k0 - 1)), m);
                        c10 = _mm512_fmadd_ps(a0, b00, c10);
                        c11 = _mm512_fmadd_ps(a0, b10, c11);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k0 - 0)), m);
                        c10 = _mm512_fmadd_ps(a0, b01, c10);
                        c11 = _mm512_fmadd_ps(a0, b11, c11);
                    }
                    if (M > 0x2)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k0 - 1)), m);
                        c20 = _mm512_fmadd_ps(a0, b00, c20);
                        c21 = _mm512_fmadd_ps(a0, b10, c21);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k0 - 0)), m);
                        c20 = _mm512_fmadd_ps(a0, b01, c20);
                        c21 = _mm512_fmadd_ps(a0, b11, c21);
                    }
                    if (M > 0x3)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k0 - 1)), m);
                        c30 = _mm512_fmadd_ps(a0, b00, c30);
                        c31 = _mm512_fmadd_ps(a0, b10, c31);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k0 - 0)), m);
                        c30 = _mm512_fmadd_ps(a0, b01, c30);
                        c31 = _mm512_fmadd_ps(a0, b11, c31);
                    }
                    if (M > 0x4)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k0 - 1)), m);
                        c40 = _mm512_fmadd_ps(a0, b00, c40);
                        c41 = _mm512_fmadd_ps(a0, b10, c41);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k0 - 0)), m);
                        c40 = _mm512_fmadd_ps(a0, b01, c40);
                        c41 = _mm512_fmadd_ps(a0, b11, c41);
                    }
                    if (M > 0x5)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k0 - 1)), m);
                        c50 = _mm512_fmadd_ps(a0, b00, c50);
                        c51 = _mm512_fmadd_ps(a0, b10, c51);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k0 - 0)), m);
                        c50 = _mm512_fmadd_ps(a0, b01, c50);
                        c51 = _mm512_fmadd_ps(a0, b11, c51);
                    }
                    if (M > 0x6)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k6 - 1)), m);
                        c60 = _mm512_fmadd_ps(a0, b00, c60);
                        c61 = _mm512_fmadd_ps(a0, b10, c61);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k6 - 0)), m);
                        c60 = _mm512_fmadd_ps(a0, b01, c60);
                        c61 = _mm512_fmadd_ps(a0, b11, c61);
                    }
                    if (M > 0x7)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k6 - 1)), m);
                        c70 = _mm512_fmadd_ps(a0, b00, c70);
                        c71 = _mm512_fmadd_ps(a0, b10, c71);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k6 - 0)), m);
                        c70 = _mm512_fmadd_ps(a0, b01, c70);
                        c71 = _mm512_fmadd_ps(a0, b11, c71);
                    }
                    if (M > 0x8)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k6 - 1)), m);
                        c80 = _mm512_fmadd_ps(a0, b00, c80);
                        c81 = _mm512_fmadd_ps(a0, b10, c81);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k6 - 0)), m);
                        c80 = _mm512_fmadd_ps(a0, b01, c80);
                        c81 = _mm512_fmadd_ps(a0, b11, c81);
                    }
                    if (M > 0x9)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k6 - 1)), m);
                        c90 = _mm512_fmadd_ps(a0, b00, c90);
                        c91 = _mm512_fmadd_ps(a0, b10, c91);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k6 - 0)), m);
                        c90 = _mm512_fmadd_ps(a0, b01, c90);
                        c91 = _mm512_fmadd_ps(a0, b11, c91);
                    }
                    if (M > 0xa)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k6 - 1)), m);
                        ca0 = _mm512_fmadd_ps(a0, b00, ca0);
                        ca1 = _mm512_fmadd_ps(a0, b10, ca1);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k6 - 0)), m);
                        ca0 = _mm512_fmadd_ps(a0, b01, ca0);
                        ca1 = _mm512_fmadd_ps(a0, b11, ca1);
                    }
                    if (M > 0xb)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k6 - 1)), m);
                        cb0 = _mm512_fmadd_ps(a0, b00, cb0);
                        cb1 = _mm512_fmadd_ps(a0, b10, cb1);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k6 - 0)), m);
                        cb0 = _mm512_fmadd_ps(a0, b01, cb0);
                        cb1 = _mm512_fmadd_ps(a0, b11, cb1);
                    }
                    B0 += DF;
                    B1 += DF;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0x0) Save2(C, c00, c01, tail), C += dC;
                if (M > 0x1) Save2(C, c10, c11, tail), C += dC;
                if (M > 0x2) Save2(C, c20, c21, tail), C += dC;
                if (M > 0x3) Save2(C, c30, c31, tail), C += dC;
                if (M > 0x4) Save2(C, c40, c41, tail), C += dC;
                if (M > 0x5) Save2(C, c50, c51, tail), C += dC;
                if (M > 0x6) Save2(C, c60, c61, tail), C += dC;
                if (M > 0x7) Save2(C, c70, c71, tail), C += dC;
                if (M > 0x8) Save2(C, c80, c81, tail), C += dC;
                if (M > 0x9) Save2(C, c90, c91, tail), C += dC;
                if (M > 0xa) Save2(C, ca0, ca1, tail), C += dC;
                if (M > 0xb) Save2(C, cb0, cb1, tail), C += dC;
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_ps(C + 0x0 * dC + 0);
                    if (M > 0x1) c10 = _mm512_loadu_ps(C + 0x1 * dC + 0);
                    if (M > 0x2) c20 = _mm512_loadu_ps(C + 0x2 * dC + 0);
                    if (M > 0x3) c30 = _mm512_loadu_ps(C + 0x3 * dC + 0);
                    if (M > 0x4) c40 = _mm512_loadu_ps(C + 0x4 * dC + 0);
                    if (M > 0x5) c50 = _mm512_loadu_ps(C + 0x5 * dC + 0);
                    if (M > 0x6) c60 = _mm512_loadu_ps(C + 0x6 * dC + 0);
                    if (M > 0x7) c70 = _mm512_loadu_ps(C + 0x7 * dC + 0);
                    if (M > 0x8) c80 = _mm512_loadu_ps(C + 0x8 * dC + 0);
                    if (M > 0x9) c90 = _mm512_loadu_ps(C + 0x9 * dC + 0);
                    if (M > 0xa) ca0 = _mm512_loadu_ps(C + 0xa * dC + 0);
                    if (M > 0xb) cb0 = _mm512_loadu_ps(C + 0xb * dC + 0);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_ps();
                    if (M > 0x1) c10 = _mm512_setzero_ps();
                    if (M > 0x2) c20 = _mm512_setzero_ps();
                    if (M > 0x3) c30 = _mm512_setzero_ps();
                    if (M > 0x4) c40 = _mm512_setzero_ps();
                    if (M > 0x5) c50 = _mm512_setzero_ps();
                    if (M > 0x6) c60 = _mm512_setzero_ps();
                    if (M > 0x7) c70 = _mm512_setzero_ps();
                    if (M > 0x8) c80 = _mm512_setzero_ps();
                    if (M > 0x9) c90 = _mm512_setzero_ps();
                    if (M > 0xa) ca0 = _mm512_setzero_ps();
                    if (M > 0xb) cb0 = _mm512_setzero_ps();
                }
                for (size_t k0 = 0, k6 = k0 + 6 * dA; k0 < K; k0 += 2, k6 += 2)
                {
                    b01 = _mm512_loadu_ps((float*)B0);
                    b00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b01), Base::Bf16::SHIFT));
                    b01 = _mm512_and_ps(b01, m);
                    if (M > 0x0)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k0 - 1)), m);
                        c00 = _mm512_fmadd_ps(a0, b00, c00);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k0 - 0)), m);
                        c00 = _mm512_fmadd_ps(a0, b01, c00);
                    }
                    if (M > 0x1)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k0 - 1)), m);
                        c10 = _mm512_fmadd_ps(a0, b00, c10);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k0 - 0)), m);
                        c10 = _mm512_fmadd_ps(a0, b01, c10);
                    }
                    if (M > 0x2)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k0 - 1)), m);
                        c20 = _mm512_fmadd_ps(a0, b00, c20);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k0 - 0)), m);
                        c20 = _mm512_fmadd_ps(a0, b01, c20);
                    }
                    if (M > 0x3)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k0 - 1)), m);
                        c30 = _mm512_fmadd_ps(a0, b00, c30);
                        c31 = _mm512_fmadd_ps(a0, b10, c31);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k0 - 0)), m);
                        c30 = _mm512_fmadd_ps(a0, b01, c30);
                        c31 = _mm512_fmadd_ps(a0, b11, c31);
                    }
                    if (M > 0x4)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k0 - 1)), m);
                        c40 = _mm512_fmadd_ps(a0, b00, c40);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k0 - 0)), m);
                        c40 = _mm512_fmadd_ps(a0, b01, c40);
                    }
                    if (M > 0x5)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k0 - 1)), m);
                        c50 = _mm512_fmadd_ps(a0, b00, c50);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k0 - 0)), m);
                        c50 = _mm512_fmadd_ps(a0, b01, c50);
                    }
                    if (M > 0x6)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k6 - 1)), m);
                        c60 = _mm512_fmadd_ps(a0, b00, c60);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A0 + k6 - 0)), m);
                        c60 = _mm512_fmadd_ps(a0, b01, c60);
                    }
                    if (M > 0x7)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k6 - 1)), m);
                        c70 = _mm512_fmadd_ps(a0, b00, c70);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A1 + k6 - 0)), m);
                        c70 = _mm512_fmadd_ps(a0, b01, c70);
                    }
                    if (M > 0x8)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k6 - 1)), m);
                        c80 = _mm512_fmadd_ps(a0, b00, c80);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A2 + k6 - 0)), m);
                        c80 = _mm512_fmadd_ps(a0, b01, c80);
                    }
                    if (M > 0x9)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k6 - 1)), m);
                        c90 = _mm512_fmadd_ps(a0, b00, c90);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A3 + k6 - 0)), m);
                        c90 = _mm512_fmadd_ps(a0, b01, c90);
                    }
                    if (M > 0xa)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k6 - 1)), m);
                        ca0 = _mm512_fmadd_ps(a0, b00, ca0);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A4 + k6 - 0)), m);
                        ca0 = _mm512_fmadd_ps(a0, b01, ca0);
                    }
                    if (M > 0xb)
                    {
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k6 - 1)), m);
                        cb0 = _mm512_fmadd_ps(a0, b00, cb0);
                        a0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(A5 + k6 - 0)), m);
                        cb0 = _mm512_fmadd_ps(a0, b01, cb0);
                    }
                    B0 += DF;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0x0) Save1(C, c00, tail), C += dC;
                if (M > 0x1) Save1(C, c10, tail), C += dC;
                if (M > 0x2) Save1(C, c20, tail), C += dC;
                if (M > 0x3) Save1(C, c30, tail), C += dC;
                if (M > 0x4) Save1(C, c40, tail), C += dC;
                if (M > 0x5) Save1(C, c50, tail), C += dC;
                if (M > 0x6) Save1(C, c60, tail), C += dC;
                if (M > 0x7) Save1(C, c70, tail), C += dC;
                if (M > 0x8) Save1(C, c80, tail), C += dC;
                if (M > 0x9) Save1(C, c90, tail), C += dC;
                if (M > 0xa) Save1(C, ca0, tail), C += dC;
                if (M > 0xb) Save1(C, cb0, tail), C += dC;
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, int update, const uint16_t* B0, float* C);

        static GemmNN_2xM_Ptr GetInnerProduct16bGemmNN_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return InnerProduct16bGemmNN_2xM<0x1>;
            case 0x2: return InnerProduct16bGemmNN_2xM<0x2>;
            case 0x3: return InnerProduct16bGemmNN_2xM<0x3>;
            case 0x4: return InnerProduct16bGemmNN_2xM<0x4>;
            case 0x5: return InnerProduct16bGemmNN_2xM<0x5>;
            case 0x6: return InnerProduct16bGemmNN_2xM<0x6>;
            case 0x7: return InnerProduct16bGemmNN_2xM<0x7>;
            case 0x8: return InnerProduct16bGemmNN_2xM<0x8>;
            case 0x9: return InnerProduct16bGemmNN_2xM<0x9>;
            case 0xa: return InnerProduct16bGemmNN_2xM<0xa>;
            case 0xb: return InnerProduct16bGemmNN_2xM<0xb>;
            case 0xc: return InnerProduct16bGemmNN_2xM<0xc>;
            }
            assert(0);
            return NULL;
        }

        static void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C)
        {
            size_t m1 = M, m = 12;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.aK * DF, dC = a.cN;
            GemmNN_2xM_Ptr gemm_2xM = GetInnerProduct16bGemmNN_2xM(m);
            GemmNN_2xM_Ptr gemm_2xT = GetInnerProduct16bGemmNN_2xM(t);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC);
                B += dB;
                C += dN;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void InnerProduct16bGemmNN_Post32f(const float* src, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, const float* bias, uint8_t* dst8)
        {
            float* dst = (float*)dst8;
            size_t NF = Simd::AlignLo(N, F);
            __mmask16 tail = TailMask16(N - NF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t j = 0;
                for (; j < NF; j += F)
                    _mm512_storeu_ps(dst + j, _mm512_add_ps(_mm512_loadu_ps(src + j), _mm512_loadu_ps(bias + j)));
                if (j < N)
                    _mm512_mask_storeu_ps(dst + j, tail, _mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + j), _mm512_maskz_loadu_ps(tail, bias + j)));
                src += a.cN;
                dst += p.N;
            }
        }

        //-----------------------------------------------------------------------------------------

        void InnerProduct16bGemmNN_Post16b(const float* src, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t N, const float* bias, uint8_t* dst8)
        {
            uint16_t* dst = (uint16_t*)dst8;
            size_t NF = Simd::AlignLo(N, F);
            __mmask16 tail = TailMask16(N - NF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t j = 0;
                for (; j < NF; j += F)
                {
                    __m512i d = Float32ToBFloat16((_mm512_add_ps(_mm512_loadu_ps(src + j), _mm512_loadu_ps(bias + j))));
                    _mm256_storeu_si256((__m256i*)(dst + j), _mm512_cvtepi32_epi16(d));
                }
                if (j < N)
                {
                    __m512i d = Float32ToBFloat16((_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + j), _mm512_maskz_loadu_ps(tail, bias + j))));
                    _mm256_mask_storeu_epi16(dst + j, tail, _mm512_cvtepi32_epi16(d));
                }
                src += a.cN;
                dst += p.N;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Avx2::SynetInnerProduct16bGemmNN(p)
        {
            SetAlgParam(F, F * 2, 12, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_sizeA)
            {
                if (p.typeA == SimdTensorData16b)
                    _prepA = InnerProduct16bGemmNN_ReorderA;
                else
                    _prepA = InnerProduct16bGemmNN_ConvertA;
            }
            if (p.typeB == SimdTensorData32f || p.constB)
            {
                if (p.transB)
                    ;// _prepB = InnerProduct16bGemmNN_ConvertBt;
                else
                    _prepB = InnerProduct16bGemmNN_ConvertBn;
            }
            else
            {
                if (p.transB)
                    ;// _prepB = InnerProduct16bGemmNN_ReorderBt;
                else
                    _prepB = InnerProduct16bGemmNN_ReorderBn;
            }
            _gemm = InnerProduct16bGemmNN_Gemm2;
            if (_sizeC || p.bias)
            {
                if (p.typeC == SimdTensorData16b)
                    _post = InnerProduct16bGemmNN_Post16b;
                else
                    _post = InnerProduct16bGemmNN_Post32f;
            }
        }
    }
#endif
}
