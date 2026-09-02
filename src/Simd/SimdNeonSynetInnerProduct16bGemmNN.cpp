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
#include "Simd/SimdNeon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Neon
    {
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;
        typedef Base::SynetInnerProduct16bGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t BroadcastBf16(uint16_t value)
        {
            return vreinterpretq_f32_u32(vdupq_n_u32(uint32_t(value) << Base::Bf16::SHIFT));
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
                size_t KDF = Simd::AlignLo(p.K, DF);
                size_t KF = Simd::AlignLo(p.K, F);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t k = 0;
                    for (; k < KDF; k += DF)
                    {
                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + k + 0));
                        uint32x4_t d1 = Float32ToBFloat16(Load<false>(src + k + F));
                        Store<false>(dst + k, PackU32(d0, d1));
                    }
                    for (; k < KF; k += F)
                    {
                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + k));
                        Store<false>(dst + k, vmovn_u32(d0));
                    }
                    for (; k < p.K; ++k)
                        dst[k] = Base::Float32ToBFloat16(src[k]);
                    for (; k < a.aK; ++k)
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
            size_t KDF = Simd::AlignLo(p.K, DF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KDF; k += DF)
                    Store<false>(dst + k, Load<false>(src + k));
                for (; k < p.K; ++k)
                    dst[k] = src[k];
                for (; k < a.aK; ++k)
                    dst[k] = 0;
                src += p.K;
                dst += a.aK;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ConvertBn(const float* src, size_t stride, uint16_t* dst)
        {
            uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + 0 * stride));
            uint32x4_t d1 = vshlq_n_u32(Float32ToBFloat16(Load<false>(src + 1 * stride)), Base::Bf16::SHIFT);
            Store<false>((uint32_t*)dst, vorrq_u32(d0, d1));
        }

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
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

        SIMD_INLINE void ConvertBt(const float* src, size_t stride, uint16_t* dst)
        {
            uint32x4_t d01 = Float32ToBFloat16(Load(src + 0 * stride, src + 1 * stride));
            uint32x4_t d23 = Float32ToBFloat16(Load(src + 2 * stride, src + 3 * stride));
            Store<false>(dst, PackU32(d01, d23));
        }

        static void InnerProduct16bGemmNN_ConvertBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ConvertBt(ps + f * p.K, p.K, dst);
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

        SIMD_INLINE void ReorderBn(const uint16_t* src, size_t stride, uint16_t* dst)
        {
            uint16x4_t d0 = LoadHalf<false>(src + 0 * stride);
            uint16x4_t d1 = LoadHalf<false>(src + 1 * stride);
            uint16x4x2_t z = vzip_u16(d0, d1);
            Store<false>(dst, vcombine_u16(z.val[0], z.val[1]));
        }

        static void InnerProduct16bGemmNN_ReorderBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
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

        SIMD_INLINE void ReorderBt(const uint16_t* src, size_t stride, uint16_t* dst)
        {
            *(uint32_t*)(dst + 0) = *(uint32_t*)(src + 0 * stride);
            *(uint32_t*)(dst + 2) = *(uint32_t*)(src + 1 * stride);
            *(uint32_t*)(dst + 4) = *(uint32_t*)(src + 2 * stride);
            *(uint32_t*)(dst + 6) = *(uint32_t*)(src + 3 * stride);
        }

        static void InnerProduct16bGemmNN_ReorderBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t Kl = AlignLo(K, a.microK), Kh = AlignHi(K, a.microK), Nf = AlignLo(N, a.F), j = 0, gap = (a.bK - Kh) * a.F;
            for (; j < Nf; j += a.F)
            {
                size_t k = 0;
                for (; k < Kl; k += 2)
                {
                    const uint16_t* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; f += F, dst += DF)
                        ReorderBt(ps + f * p.K, p.K, dst);
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
            uint8_t* ptr, float* buf, float32x4_t val, const float32x4_t* bias, const float32x4_t* params, size_t index)
        {
            if (term == Term16bInterim)
            {
                Store<false>(buf + index * F, val);
            }
            else
            {
                float32x4_t f32 = Activate<type>(vaddq_f32(val, bias[index]), params, index);
                if (term == Term16bLast16b)
                    Store<false>((uint16_t*)(ptr + index * F * sizeof(uint16_t)), vmovn_u32(Float32ToBFloat16(f32)));
                else
                    Store<false>((float*)ptr + index * F, f32);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(
            uint8_t* ptr, float* buf, float32x4_t val, const float32x4_t* bias, const float32x4_t* params, size_t index, size_t tail)
        {
            if (term == Term16bInterim)
            {
                float tmp[F];
                Store<false>(tmp, val);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
            else
            {
                float32x4_t f32 = Activate<type>(vaddq_f32(val, bias[index]), params, index);
                if (term == Term16bLast16b)
                {
                    uint16_t tmp[F];
                    Store<false>(tmp, vmovn_u32(Float32ToBFloat16(f32)));
                    for (size_t i = 0; i < tail; ++i)
                        ((uint16_t*)ptr)[index * F + i] = tmp[i];
                }
                else
                {
                    float tmp[F];
                    Store<false>(tmp, f32);
                    for (size_t i = 0; i < tail; ++i)
                        ((float*)ptr)[index * F + i] = tmp[i];
                }
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(
            uint8_t* ptr, float* buf, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params)
        {
            Save1<term, type>(ptr, buf, val0, bias, params, 0);
            Save1<term, type>(ptr, buf, val1, bias, params, 1);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(
            uint8_t* ptr, float* buf, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Save1<term, type>(ptr, buf, val0, bias, params, 0);
            Save1<term, type>(ptr, buf, val1, bias, params, 1, tail);
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void InnerProduct16bGemmNN_2xM(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a,
            size_t N, size_t K, int update, const uint16_t* B0, float* C, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst)
        {
            float32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, a0, b00, b01, b10, b11;
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
                    if (M > 0) c00 = Load<false>(C + 0 * dC + 0), c01 = Load<false>(C + 0 * dC + F);
                    if (M > 1) c10 = Load<false>(C + 1 * dC + 0), c11 = Load<false>(C + 1 * dC + F);
                    if (M > 2) c20 = Load<false>(C + 2 * dC + 0), c21 = Load<false>(C + 2 * dC + F);
                    if (M > 3) c30 = Load<false>(C + 3 * dC + 0), c31 = Load<false>(C + 3 * dC + F);
                    if (M > 4) c40 = Load<false>(C + 4 * dC + 0), c41 = Load<false>(C + 4 * dC + F);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
                    if (M > 1) c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
                    if (M > 2) c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
                    if (M > 3) c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
                    if (M > 4) c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    uint32x4_t b0u = Load<false>((uint32_t*)B0);
                    b00 = vreinterpretq_f32_u32(vshlq_n_u32(b0u, Base::Bf16::SHIFT));
                    b01 = vreinterpretq_f32_u32(vandq_u32(b0u, Bf16::MASK));
                    uint32x4_t b1u = Load<false>((uint32_t*)B1);
                    b10 = vreinterpretq_f32_u32(vshlq_n_u32(b1u, Base::Bf16::SHIFT));
                    b11 = vreinterpretq_f32_u32(vandq_u32(b1u, Bf16::MASK));
                    if (M > 0)
                    {
                        a0 = BroadcastBf16(A0[k + 0]);
                        c00 = vmlaq_f32(c00, a0, b00);
                        c01 = vmlaq_f32(c01, a0, b10);
                        a0 = BroadcastBf16(A0[k + 1]);
                        c00 = vmlaq_f32(c00, a0, b01);
                        c01 = vmlaq_f32(c01, a0, b11);
                    }
                    if (M > 1)
                    {
                        a0 = BroadcastBf16(A1[k + 0]);
                        c10 = vmlaq_f32(c10, a0, b00);
                        c11 = vmlaq_f32(c11, a0, b10);
                        a0 = BroadcastBf16(A1[k + 1]);
                        c10 = vmlaq_f32(c10, a0, b01);
                        c11 = vmlaq_f32(c11, a0, b11);
                    }
                    if (M > 2)
                    {
                        a0 = BroadcastBf16(A2[k + 0]);
                        c20 = vmlaq_f32(c20, a0, b00);
                        c21 = vmlaq_f32(c21, a0, b10);
                        a0 = BroadcastBf16(A2[k + 1]);
                        c20 = vmlaq_f32(c20, a0, b01);
                        c21 = vmlaq_f32(c21, a0, b11);
                    }
                    if (M > 3)
                    {
                        a0 = BroadcastBf16(A3[k + 0]);
                        c30 = vmlaq_f32(c30, a0, b00);
                        c31 = vmlaq_f32(c31, a0, b10);
                        a0 = BroadcastBf16(A3[k + 1]);
                        c30 = vmlaq_f32(c30, a0, b01);
                        c31 = vmlaq_f32(c31, a0, b11);
                    }
                    if (M > 4)
                    {
                        a0 = BroadcastBf16(A4[k + 0]);
                        c40 = vmlaq_f32(c40, a0, b00);
                        c41 = vmlaq_f32(c41, a0, b10);
                        a0 = BroadcastBf16(A4[k + 1]);
                        c40 = vmlaq_f32(c40, a0, b01);
                        c41 = vmlaq_f32(c41, a0, b11);
                    }
                    B0 += DF;
                    B1 += DF;
                }
                if (N == DF)
                {
                    if (M > 0) Save2<term, type>(dst, C, c00, c01, bias, params), C += dC, dst += dD;
                    if (M > 1) Save2<term, type>(dst, C, c10, c11, bias, params), C += dC, dst += dD;
                    if (M > 2) Save2<term, type>(dst, C, c20, c21, bias, params), C += dC, dst += dD;
                    if (M > 3) Save2<term, type>(dst, C, c30, c31, bias, params), C += dC, dst += dD;
                    if (M > 4) Save2<term, type>(dst, C, c40, c41, bias, params), C += dC, dst += dD;
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0) Save2<term, type>(dst, C, c00, c01, bias, params, tail), C += dC, dst += dD;
                    if (M > 1) Save2<term, type>(dst, C, c10, c11, bias, params, tail), C += dC, dst += dD;
                    if (M > 2) Save2<term, type>(dst, C, c20, c21, bias, params, tail), C += dC, dst += dD;
                    if (M > 3) Save2<term, type>(dst, C, c30, c31, bias, params, tail), C += dC, dst += dD;
                    if (M > 4) Save2<term, type>(dst, C, c40, c41, bias, params, tail), C += dC, dst += dD;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) c00 = Load<false>(C + 0 * dC + 0);
                    if (M > 1) c10 = Load<false>(C + 1 * dC + 0);
                    if (M > 2) c20 = Load<false>(C + 2 * dC + 0);
                    if (M > 3) c30 = Load<false>(C + 3 * dC + 0);
                    if (M > 4) c40 = Load<false>(C + 4 * dC + 0);
                }
                else
                {
                    if (M > 0) c00 = vdupq_n_f32(0.0f);
                    if (M > 1) c10 = vdupq_n_f32(0.0f);
                    if (M > 2) c20 = vdupq_n_f32(0.0f);
                    if (M > 3) c30 = vdupq_n_f32(0.0f);
                    if (M > 4) c40 = vdupq_n_f32(0.0f);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    uint32x4_t b0u = Load<false>((uint32_t*)B0);
                    b00 = vreinterpretq_f32_u32(vshlq_n_u32(b0u, Base::Bf16::SHIFT));
                    b01 = vreinterpretq_f32_u32(vandq_u32(b0u, Bf16::MASK));
                    if (M > 0)
                    {
                        a0 = BroadcastBf16(A0[k + 0]);
                        c00 = vmlaq_f32(c00, a0, b00);
                        a0 = BroadcastBf16(A0[k + 1]);
                        c00 = vmlaq_f32(c00, a0, b01);
                    }
                    if (M > 1)
                    {
                        a0 = BroadcastBf16(A1[k + 0]);
                        c10 = vmlaq_f32(c10, a0, b00);
                        a0 = BroadcastBf16(A1[k + 1]);
                        c10 = vmlaq_f32(c10, a0, b01);
                    }
                    if (M > 2)
                    {
                        a0 = BroadcastBf16(A2[k + 0]);
                        c20 = vmlaq_f32(c20, a0, b00);
                        a0 = BroadcastBf16(A2[k + 1]);
                        c20 = vmlaq_f32(c20, a0, b01);
                    }
                    if (M > 3)
                    {
                        a0 = BroadcastBf16(A3[k + 0]);
                        c30 = vmlaq_f32(c30, a0, b00);
                        a0 = BroadcastBf16(A3[k + 1]);
                        c30 = vmlaq_f32(c30, a0, b01);
                    }
                    if (M > 4)
                    {
                        a0 = BroadcastBf16(A4[k + 0]);
                        c40 = vmlaq_f32(c40, a0, b00);
                        a0 = BroadcastBf16(A4[k + 1]);
                        c40 = vmlaq_f32(c40, a0, b01);
                    }
                    B0 += DF;
                }
                if (N == F)
                {
                    if (M > 0) Save1<term, type>(dst, C, c00, bias, params, 0), C += dC, dst += dD;
                    if (M > 1) Save1<term, type>(dst, C, c10, bias, params, 0), C += dC, dst += dD;
                    if (M > 2) Save1<term, type>(dst, C, c20, bias, params, 0), C += dC, dst += dD;
                    if (M > 3) Save1<term, type>(dst, C, c30, bias, params, 0), C += dC, dst += dD;
                    if (M > 4) Save1<term, type>(dst, C, c40, bias, params, 0), C += dC, dst += dD;
                }
                else
                {
                    size_t tail = N;
                    if (M > 0) Save1<term, type>(dst, C, c00, bias, params, 0, tail), C += dC, dst += dD;
                    if (M > 1) Save1<term, type>(dst, C, c10, bias, params, 0, tail), C += dC, dst += dD;
                    if (M > 2) Save1<term, type>(dst, C, c20, bias, params, 0, tail), C += dC, dst += dD;
                    if (M > 3) Save1<term, type>(dst, C, c30, bias, params, 0, tail), C += dC, dst += dD;
                    if (M > 4) Save1<term, type>(dst, C, c40, bias, params, 0, tail), C += dC, dst += dD;
                }
            }
        }

        typedef void(*GemmNN_2xM_Ptr)(const uint16_t* A0, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, int update, const uint16_t* B0, float* C, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst);

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

        template<Term16bType term, SimdConvolutionActivationType type> void InnerProduct16bGemmNN_Gemm2(const uint16_t* A, const InnerProductParam16b& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const uint16_t* B, float* C, int post, const float* bias, const float* params, float* sum, uint8_t* dst)
        {
            size_t m1 = M, m = 5;
            size_t mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dA = a.aK, dB = a.bK * DF, dC = a.cN, dD = p.N * a.eC;
            GemmNN_2xM_Ptr gemm_2xM = post ? GetGemmNN_2xM<term, type>(m) : GetGemmNN_2xM<Term16bInterim, type>(m);
            GemmNN_2xM_Ptr gemm_2xT = post ? GetGemmNN_2xM<term, type>(t) : GetGemmNN_2xM<Term16bInterim, type>(t);

            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bias[0] = Load<false>(bias + j + 0);
                _bias[1] = Load<false>(bias + j + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + j + 0);
                    _params[1] = Load<false>(params + j + F);
                }

                size_t i = 0;
                for (; i < mm; i += m)
                    gemm_2xM(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, _params, dst + i * dD);
                for (; i < m1; i += t)
                    gemm_2xT(A + i * dA, p, a, dN, K, update, B, C + i * dC, _bias, _params, dst + i * dD);
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
