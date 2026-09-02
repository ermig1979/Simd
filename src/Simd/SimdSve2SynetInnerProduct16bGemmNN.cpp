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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sve2
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svbfloat16_t BroadcastBf16x2(const uint16_t* src)
        {
            return svreinterpret_bf16_u32(svdup_n_u32(uint32_t(src[0]) | (uint32_t(src[1]) << 16)));
        }

        SIMD_INLINE svbfloat16_t LoadBf16x2(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_bf16_u32(svld1_u32(mask, (const uint32_t*)src));
        }

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
                for (size_t i = 0; i < M; ++i)
                {
                    Float32ToBFloat16(src, p.K, dst);
                    for (size_t k = p.K; k < a.aK; ++k)
                        dst[k] = 0;
                    src += p.K;
                    dst += a.aK;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ReorderA(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t M, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            for (size_t i = 0; i < M; ++i)
            {
                memcpy(dst, src, p.K * sizeof(uint16_t));
                for (size_t k = p.K; k < a.aK; ++k)
                    dst[k] = 0;
                src += p.K;
                dst += a.aK;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBn(const float* src, size_t stride, uint16_t* dst, const svbool_t& mask)
        {
            svfloat32_t s0 = svld1_f32(mask, src + 0 * stride);
            svfloat32_t s1 = svld1_f32(mask, src + 1 * stride);
            svuint32_t d0 = Float32ToBFloat16(s0, mask);
            svuint32_t d1 = svlsl_n_u32_x(mask, Float32ToBFloat16(s1, mask), Base::Bf16::SHIFT);
            svst1_u32(mask, (uint32_t*)dst, svorr_u32_x(mask, d0, d1));
        }

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    ConvertBn(ps, p.N, dst, body);
                    dst += F * 2;
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
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
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBt(const float* src, size_t stride, uint16_t* dst, size_t F)
        {
            for (size_t f = 0; f < F; ++f)
            {
                dst[0] = Base::Float32ToBFloat16(src[f * stride + 0]);
                dst[1] = Base::Float32ToBFloat16(src[f * stride + 1]);
                dst += 2;
            }
        }

        static void InnerProduct16bGemmNN_ConvertBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            const size_t F = a.F;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    ConvertBt(ps, p.K, dst, F);
                    dst += F * 2;
                }
                for (; k < Kh; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = Base::Float32ToBFloat16(ps[f * p.K + i]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderBn(const uint16_t* src, size_t stride, uint16_t* dst, const svbool_t& mask)
        {
            svuint32_t d0 = svld1uh_u32(mask, src + 0 * stride);
            svuint32_t d1 = svld1uh_u32(mask, src + 1 * stride);
            svst1_u32(mask, (uint32_t*)dst, svorr_u32_x(mask, d0, svlsl_n_u32_x(mask, d1, Base::Bf16::SHIFT)));
        }

        static void InnerProduct16bGemmNN_ReorderBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    ReorderBn(ps, p.N, dst, body);
                    dst += F * 2;
                }
                for (; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
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
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[i * p.N + f];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderBt(const uint16_t* src, size_t stride, uint16_t* dst, size_t F)
        {
            for (size_t f = 0; f < F; ++f)
            {
                ((uint32_t*)dst)[f] = ((uint32_t*)(src + f * stride))[0];
            }
        }

        static void InnerProduct16bGemmNN_ReorderBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            const size_t F = a.F;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    ReorderBt(ps, p.K, dst, F);
                    dst += F * 2;
                }
                for (; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[f * p.K + i];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
            for (; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < N && k + i < K)
                                *(dst++) = ps[f * p.K + i];
                            else
                                *(dst++) = 0;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(
            uint8_t* dst, float* buf, svfloat32_t val, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, size_t index, const svbool_t& mask)
        {
            if (term == Term16bInterim)
            {
                svst1_f32(mask, buf, val);
            }
            else
            {
                svfloat32_t f32 = Activate<type>(svadd_f32_x(mask, val, bias), param0, param1, index, mask);
                if (term == Term16bLast16b)
                    svst1h_u32(mask, (uint16_t*)dst, Float32ToBFloat16(f32, mask));
                else
                    svst1_f32(mask, (float*)dst, f32);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(
            uint8_t* dst, float* buf, svfloat32_t val0, svfloat32_t val1,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1,
            const svbool_t& mask0, const svbool_t& mask1, size_t F)
        {
            Save1<term, type>(dst, buf, val0, bias0, param0, param1, 0, mask0);
            if (term == Term16bInterim)
                Save1<term, type>(dst, buf + F, val1, bias1, param0, param1, 1, mask1);
            else if (term == Term16bLast16b)
                Save1<term, type>(dst + F * sizeof(uint16_t), buf, val1, bias1, param0, param1, 1, mask1);
            else
                Save1<term, type>(dst + F * sizeof(float), buf, val1, bias1, param0, param1, 1, mask1);
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void InnerProduct16bGemmNN_2xM(
            const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t N, size_t K, int update, const uint16_t* B0, float* C,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41;
            svbfloat16_t a0, b0, b1;
            size_t dC = a.cN, dA = a.aK, dD = p.N * a.eC;
            const uint16_t* B1 = B0 + a.bK * F;
            const uint16_t* A1 = A0 + 1 * dA;
            const uint16_t* A2 = A0 + 2 * dA;
            const uint16_t* A3 = A0 + 3 * dA;
            const uint16_t* A4 = A0 + 4 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0) c00 = svld1_f32(body, C + 0 * dC + 0), c01 = svld1_f32(body, C + 0 * dC + F);
                    if (M > 1) c10 = svld1_f32(body, C + 1 * dC + 0), c11 = svld1_f32(body, C + 1 * dC + F);
                    if (M > 2) c20 = svld1_f32(body, C + 2 * dC + 0), c21 = svld1_f32(body, C + 2 * dC + F);
                    if (M > 3) c30 = svld1_f32(body, C + 3 * dC + 0), c31 = svld1_f32(body, C + 3 * dC + F);
                    if (M > 4) c40 = svld1_f32(body, C + 4 * dC + 0), c41 = svld1_f32(body, C + 4 * dC + F);
                }
                else
                {
                    if (M > 0) c00 = svdup_n_f32(0.0f), c01 = svdup_n_f32(0.0f);
                    if (M > 1) c10 = svdup_n_f32(0.0f), c11 = svdup_n_f32(0.0f);
                    if (M > 2) c20 = svdup_n_f32(0.0f), c21 = svdup_n_f32(0.0f);
                    if (M > 3) c30 = svdup_n_f32(0.0f), c31 = svdup_n_f32(0.0f);
                    if (M > 4) c40 = svdup_n_f32(0.0f), c41 = svdup_n_f32(0.0f);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b0 = LoadBf16x2(B0, body);
                    b1 = LoadBf16x2(B1, body);
                    if (M > 0)
                    {
                        a0 = BroadcastBf16x2(A0 + k);
                        c00 = svbfdot_f32(c00, a0, b0);
                        c01 = svbfdot_f32(c01, a0, b1);
                    }
                    if (M > 1)
                    {
                        a0 = BroadcastBf16x2(A1 + k);
                        c10 = svbfdot_f32(c10, a0, b0);
                        c11 = svbfdot_f32(c11, a0, b1);
                    }
                    if (M > 2)
                    {
                        a0 = BroadcastBf16x2(A2 + k);
                        c20 = svbfdot_f32(c20, a0, b0);
                        c21 = svbfdot_f32(c21, a0, b1);
                    }
                    if (M > 3)
                    {
                        a0 = BroadcastBf16x2(A3 + k);
                        c30 = svbfdot_f32(c30, a0, b0);
                        c31 = svbfdot_f32(c31, a0, b1);
                    }
                    if (M > 4)
                    {
                        a0 = BroadcastBf16x2(A4 + k);
                        c40 = svbfdot_f32(c40, a0, b0);
                        c41 = svbfdot_f32(c41, a0, b1);
                    }
                    B0 += DF;
                    B1 += DF;
                }
                svbool_t mask1 = (N == DF) ? body : svwhilelt_b32((size_t)0, N - F);
                if (M > 0) Save2<term, type>(dst, C + 0 * dC, c00, c01, bias0, bias1, param0, param1, body, mask1, F), C += dC, dst += dD;
                if (M > 1) Save2<term, type>(dst, C + 0 * dC, c10, c11, bias0, bias1, param0, param1, body, mask1, F), C += dC, dst += dD;
                if (M > 2) Save2<term, type>(dst, C + 0 * dC, c20, c21, bias0, bias1, param0, param1, body, mask1, F), C += dC, dst += dD;
                if (M > 3) Save2<term, type>(dst, C + 0 * dC, c30, c31, bias0, bias1, param0, param1, body, mask1, F), C += dC, dst += dD;
                if (M > 4) Save2<term, type>(dst, C + 0 * dC, c40, c41, bias0, bias1, param0, param1, body, mask1, F), C += dC, dst += dD;
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = svld1_f32(body, C + 0 * dC + 0);
                    if (M > 1) c10 = svld1_f32(body, C + 1 * dC + 0);
                    if (M > 2) c20 = svld1_f32(body, C + 2 * dC + 0);
                    if (M > 3) c30 = svld1_f32(body, C + 3 * dC + 0);
                    if (M > 4) c40 = svld1_f32(body, C + 4 * dC + 0);
                }
                else
                {
                    if (M > 0) c00 = svdup_n_f32(0.0f);
                    if (M > 1) c10 = svdup_n_f32(0.0f);
                    if (M > 2) c20 = svdup_n_f32(0.0f);
                    if (M > 3) c30 = svdup_n_f32(0.0f);
                    if (M > 4) c40 = svdup_n_f32(0.0f);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    b0 = LoadBf16x2(B0, body);
                    if (M > 0)
                    {
                        a0 = BroadcastBf16x2(A0 + k);
                        c00 = svbfdot_f32(c00, a0, b0);
                    }
                    if (M > 1)
                    {
                        a0 = BroadcastBf16x2(A1 + k);
                        c10 = svbfdot_f32(c10, a0, b0);
                    }
                    if (M > 2)
                    {
                        a0 = BroadcastBf16x2(A2 + k);
                        c20 = svbfdot_f32(c20, a0, b0);
                    }
                    if (M > 3)
                    {
                        a0 = BroadcastBf16x2(A3 + k);
                        c30 = svbfdot_f32(c30, a0, b0);
                    }
                    if (M > 4)
                    {
                        a0 = BroadcastBf16x2(A4 + k);
                        c40 = svbfdot_f32(c40, a0, b0);
                    }
                    B0 += DF;
                }
                svbool_t mask0 = (N == F) ? body : svwhilelt_b32((size_t)0, N);
                if (M > 0) Save1<term, type>(dst, C + 0 * dC, c00, bias0, param0, param1, 0, mask0), C += dC, dst += dD;
                if (M > 1) Save1<term, type>(dst, C + 0 * dC, c10, bias0, param0, param1, 0, mask0), C += dC, dst += dD;
                if (M > 2) Save1<term, type>(dst, C + 0 * dC, c20, bias0, param0, param1, 0, mask0), C += dC, dst += dD;
                if (M > 3) Save1<term, type>(dst, C + 0 * dC, c30, bias0, param0, param1, 0, mask0), C += dC, dst += dD;
                if (M > 4) Save1<term, type>(dst, C + 0 * dC, c40, bias0, param0, param1, 0, mask0), C += dC, dst += dD;
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t N, size_t K, int update, const uint16_t* B0, float* C,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> GemmNN_2xM_Ptr GetGemmNN_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InnerProduct16bGemmNN_2xM<term, type, 1>;
            case 2: return InnerProduct16bGemmNN_2xM<term, type, 2>;
            case 3: return InnerProduct16bGemmNN_2xM<term, type, 3>;
            case 4: return InnerProduct16bGemmNN_2xM<term, type, 4>;
            case 5: return InnerProduct16bGemmNN_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void InnerProduct16bGemmNN_Gemm2(
            const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C, int post,
            const float* bias, const float* params, float* sum, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t m1 = M, m = 5;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.bK * DF, dC = a.cN, dD = p.N * a.eC;
            GemmNN_2xM_Ptr gemm_2xM = post ? GetGemmNN_2xM<term, type>(m) : GetGemmNN_2xM<Term16bInterim, type>(m);
            GemmNN_2xM_Ptr gemm_2xT = post ? GetGemmNN_2xM<term, type>(t) : GetGemmNN_2xM<Term16bInterim, type>(t);

            svfloat32_t _param0 = svdup_n_f32(params[0]);
            svfloat32_t _param1 = svdup_n_f32(params[1]);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                svfloat32_t _bias0 = svld1_f32(body, bias + j + 0);
                svfloat32_t _bias1 = svld1_f32(body, bias + j + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _param0 = svld1_f32(body, params + j + 0);
                    _param1 = svld1_f32(body, params + j + F);
                }

                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias0, _bias1, _param0, _param1, dst + i * dD);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias0, _bias1, _param0, _param1, dst + i * dD);
                B += dB;
                C += dN;
                dst += DF * a.eC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetGemm(const InnerProductParam16b& p, GemmPtr& gemm)
        {
            if (p.typeC == SimdTensorData16b)
                gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast16b, type>;
            else
                gemm = InnerProduct16bGemmNN_Gemm2<Term16bLast32f, type>;
        }

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : Base::SynetInnerProduct16bGemmNN(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, 5, F * 2, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
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
                    _prepB = InnerProduct16bGemmNN_ConvertBt;
                else
                    _prepB = InnerProduct16bGemmNN_ConvertBn;
            }
            else
            {
                if (p.transB)
                    _prepB = InnerProduct16bGemmNN_ReorderBt;
                else
                    _prepB = InnerProduct16bGemmNN_ReorderBn;
            }
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationRelu: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationLeakyRelu: SetGemm<SimdConvolutionActivationPrelu>(p, _gemm); break;
            case SimdConvolutionActivationRestrictRange: SetGemm<SimdConvolutionActivationRestrictRange>(p, _gemm); break;
            case SimdConvolutionActivationPrelu: SetGemm<SimdConvolutionActivationPrelu>(p, _gemm); break;
            case SimdConvolutionActivationElu: SetGemm<SimdConvolutionActivationElu>(p, _gemm); break;
            case SimdConvolutionActivationHswish: SetGemm<SimdConvolutionActivationHswish>(p, _gemm); break;
            case SimdConvolutionActivationMish: SetGemm<SimdConvolutionActivationMish>(p, _gemm); break;
            case SimdConvolutionActivationHardSigmoid: SetGemm<SimdConvolutionActivationHardSigmoid>(p, _gemm); break;
            case SimdConvolutionActivationSwish: SetGemm<SimdConvolutionActivationSwish>(p, _gemm); break;
            case SimdConvolutionActivationGelu: SetGemm<SimdConvolutionActivationGelu>(p, _gemm); break;
            default: assert(0);
            }
        }
    }
#endif
}
