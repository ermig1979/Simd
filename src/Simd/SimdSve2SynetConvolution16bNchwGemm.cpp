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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        typedef Base::SynetConvolution16bNchwGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNchwGemm::ConvolutionPtr Convolution;

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

        SIMD_INLINE void ConvertF(const float* src, size_t stride, uint16_t* dst, const svbool_t& mask)
        {
            svfloat32_t s0 = svld1_f32(mask, src + 0 * stride);
            svfloat32_t s1 = svld1_f32(mask, src + 1 * stride);
            svuint32_t d0 = Float32ToBFloat16(s0, mask);
            svuint32_t d1 = svlsl_n_u32_x(mask, Float32ToBFloat16(s1, mask), Base::Bf16::SHIFT);
            svst1_u32(mask, (uint32_t*)dst, svorr_u32_x(mask, d0, d1));
        }

        static void Convert16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const float* src = ((float*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (j = 0; j < NF; j += a.F)
            {
                for (k = 0; k < K2; k += 2)
                {
                    const float* src0 = src + k * dS;
                    ConvertF(src0, dS, dst, body);
                    dst += F * 2;
                }
                for (; k < K; k += 2)
                {
                    const float* src0 = src + k * dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = Base::Float32ToBFloat16(src0[f]);
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                src += a.F;
            }
            if (j < N)
            {
                size_t tail = N - j, f;
                for (k = 0; k < K2; k += 2)
                {
                    const float* src0 = src + k * dS, * src1 = src0 + dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = Base::Float32ToBFloat16(src0[f]);
                        *dst++ = Base::Float32ToBFloat16(src1[f]);
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < K; k += 2)
                {
                    const float* src0 = src + k * dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = Base::Float32ToBFloat16(src0[f]);
                        *dst++ = 0;
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
            }
        }

        SIMD_INLINE void ReorderF(const uint16_t* src, size_t stride, uint16_t* dst, const svbool_t& mask)
        {
            svuint32_t d0 = svld1uh_u32(mask, src + 0 * stride);
            svuint32_t d1 = svld1uh_u32(mask, src + 1 * stride);
            svst1_u32(mask, (uint32_t*)dst, svorr_u32_x(mask, d0, svlsl_n_u32_x(mask, d1, Base::Bf16::SHIFT)));
        }

        static void Reorder16bNchwGemm1x1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint16_t* dst)
        {
            const uint16_t* src = ((uint16_t*)src8) + (cBeg * p.srcH + yBeg) * p.srcW;
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), j = 0, dS = p.srcH * p.srcW;
            size_t K = Min(cEnd, a.K) - cBeg, K2 = AlignLo(K, 2), KH = AlignHi(K, a.microK), k;
            for (; j < NF; j += a.F)
            {
                for (k = 0; k < K2; k += 2)
                {
                    const uint16_t* src0 = src + k * dS;
                    ReorderF(src0, dS, dst, body);
                    dst += F * 2;
                }
                for (; k < K; k += 2)
                {
                    const uint16_t* src0 = src + k * dS;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                src += a.F;
            }
            if (j < N)
            {
                size_t tail = N - j, f;
                for (k = 0; k < K2; k += 2)
                {
                    const uint16_t* src0 = src + k * dS, * src1 = src0 + dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = src1[f];
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < K; k += 2)
                {
                    const uint16_t* src0 = src + k * dS;
                    for (f = 0; f < tail; ++f)
                    {
                        *dst++ = src0[f];
                        *dst++ = 0;
                    }
                    for (; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
                for (; k < KH; k += 2)
                {
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        *dst++ = 0;
                        *dst++ = 0;
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(
            uint8_t* ptr, float* buf, svfloat32_t val, const float* bias, const float* params, size_t offset, const svbool_t& mask)
        {
            if (term == Term16bInterim)
            {
                svst1_f32(mask, buf, val);
            }
            else
            {
                svfloat32_t f32 = ActivateNchw<type>(svadd_n_f32_x(mask, val, bias[offset]), params, offset, mask);
                if (term == Term16bLast16b)
                    svst1h_u32(mask, (uint16_t*)ptr, Float32ToBFloat16(f32, mask));
                else
                    svst1_f32(mask, (float*)ptr, f32);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(
            uint8_t* ptr, float* buf, svfloat32_t val0, svfloat32_t val1, const float* bias, const float* params, size_t offset,
            const svbool_t& mask0, const svbool_t& mask1, size_t F)
        {
            Save1<term, type>(ptr, buf, val0, bias, params, offset, mask0);
            if (term == Term16bInterim)
                Save1<term, type>(ptr, buf + F, val1, bias, params, offset, mask1);
            else if (term == Term16bLast16b)
                Save1<term, type>(ptr + F * sizeof(uint16_t), buf, val1, bias, params, offset, mask1);
            else
                Save1<term, type>(ptr + F * sizeof(float), buf, val1, bias, params, offset, mask1);
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNchwGemm_2xM(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t w0, s0, s1;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem;
            const uint16_t* src1 = src0 + K * F;
            const uint16_t* weight1 = weight0 + 1 * K;
            const uint16_t* weight2 = weight0 + 2 * K;
            const uint16_t* weight3 = weight0 + 3 * K;
            const uint16_t* weight4 = weight0 + 4 * K;
            if (dstS > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                    if (M > 1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                    if (M > 2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                    if (M > 3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                    if (M > 4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = svld1_f32(body, buf + 0 * dB + 0), d01 = svld1_f32(body, buf + 0 * dB + F);
                    if (M > 1) d10 = svld1_f32(body, buf + 1 * dB + 0), d11 = svld1_f32(body, buf + 1 * dB + F);
                    if (M > 2) d20 = svld1_f32(body, buf + 2 * dB + 0), d21 = svld1_f32(body, buf + 2 * dB + F);
                    if (M > 3) d30 = svld1_f32(body, buf + 3 * dB + 0), d31 = svld1_f32(body, buf + 3 * dB + F);
                    if (M > 4) d40 = svld1_f32(body, buf + 4 * dB + 0), d41 = svld1_f32(body, buf + 4 * dB + F);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    s0 = LoadBf16x2(src0, body);
                    s1 = LoadBf16x2(src1, body);
                    if (M > 0)
                    {
                        w0 = BroadcastBf16x2(weight0 + k);
                        d00 = svbfdot_f32(d00, w0, s0);
                        d01 = svbfdot_f32(d01, w0, s1);
                    }
                    if (M > 1)
                    {
                        w0 = BroadcastBf16x2(weight1 + k);
                        d10 = svbfdot_f32(d10, w0, s0);
                        d11 = svbfdot_f32(d11, w0, s1);
                    }
                    if (M > 2)
                    {
                        w0 = BroadcastBf16x2(weight2 + k);
                        d20 = svbfdot_f32(d20, w0, s0);
                        d21 = svbfdot_f32(d21, w0, s1);
                    }
                    if (M > 3)
                    {
                        w0 = BroadcastBf16x2(weight3 + k);
                        d30 = svbfdot_f32(d30, w0, s0);
                        d31 = svbfdot_f32(d31, w0, s1);
                    }
                    if (M > 4)
                    {
                        w0 = BroadcastBf16x2(weight4 + k);
                        d40 = svbfdot_f32(d40, w0, s0);
                        d41 = svbfdot_f32(d41, w0, s1);
                    }
                    src0 += DF;
                    src1 += DF;
                }
                svbool_t mask1 = (dstS == DF) ? body : svwhilelt_b32((size_t)0, dstS - F);
                if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias, params, 0, body, mask1, F), dst += dD, buf += dB;
                if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias, params, 1, body, mask1, F), dst += dD, buf += dB;
                if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias, params, 2, body, mask1, F), dst += dD, buf += dB;
                if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias, params, 3, body, mask1, F), dst += dD, buf += dB;
                if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias, params, 4, body, mask1, F), dst += dD, buf += dB;
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = svdup_n_f32(0.0f);
                    if (M > 1) d10 = svdup_n_f32(0.0f);
                    if (M > 2) d20 = svdup_n_f32(0.0f);
                    if (M > 3) d30 = svdup_n_f32(0.0f);
                    if (M > 4) d40 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = svld1_f32(body, buf + 0 * dB + 0);
                    if (M > 1) d10 = svld1_f32(body, buf + 1 * dB + 0);
                    if (M > 2) d20 = svld1_f32(body, buf + 2 * dB + 0);
                    if (M > 3) d30 = svld1_f32(body, buf + 3 * dB + 0);
                    if (M > 4) d40 = svld1_f32(body, buf + 4 * dB + 0);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    s0 = LoadBf16x2(src0, body);
                    if (M > 0)
                    {
                        w0 = BroadcastBf16x2(weight0 + k);
                        d00 = svbfdot_f32(d00, w0, s0);
                    }
                    if (M > 1)
                    {
                        w0 = BroadcastBf16x2(weight1 + k);
                        d10 = svbfdot_f32(d10, w0, s0);
                    }
                    if (M > 2)
                    {
                        w0 = BroadcastBf16x2(weight2 + k);
                        d20 = svbfdot_f32(d20, w0, s0);
                    }
                    if (M > 3)
                    {
                        w0 = BroadcastBf16x2(weight3 + k);
                        d30 = svbfdot_f32(d30, w0, s0);
                    }
                    if (M > 4)
                    {
                        w0 = BroadcastBf16x2(weight4 + k);
                        d40 = svbfdot_f32(d40, w0, s0);
                    }
                    src0 += DF;
                }
                svbool_t mask0 = (dstS == F) ? body : svwhilelt_b32((size_t)0, dstS);
                if (M > 0) Save1<term, type>(dst, buf, d00, bias, params, 0, mask0), dst += dD, buf += dB;
                if (M > 1) Save1<term, type>(dst, buf, d10, bias, params, 1, mask0), dst += dD, buf += dB;
                if (M > 2) Save1<term, type>(dst, buf, d20, bias, params, 2, mask0), dst += dD, buf += dB;
                if (M > 3) Save1<term, type>(dst, buf, d30, bias, params, 3, mask0), dst += dD, buf += dB;
                if (M > 4) Save1<term, type>(dst, buf, d40, bias, params, 4, mask0), dst += dD, buf += dB;
            }
        }

        typedef void(*Convolution16bNchwGemm_2xM_Ptr)(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNchwGemm_2xM_Ptr GetConvolution16bNchwGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNchwGemm_2xM<term, type, 1>;
            case 2: return Convolution16bNchwGemm_2xM<term, type, 2>;
            case 3: return Convolution16bNchwGemm_2xM<term, type, 3>;
            case 4: return Convolution16bNchwGemm_2xM<term, type, 4>;
            case 5: return Convolution16bNchwGemm_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNchwGemm_2(const uint16_t* weight, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* src, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            Convolution16bNchwGemm_2xM_Ptr convolution_2xN = GetConvolution16bNchwGemm_2xM<term, type>(n);
            Convolution16bNchwGemm_2xM_Ptr convolution_2xM = GetConvolution16bNchwGemm_2xM<term, type>(m);

            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const uint16_t* w = weight;
                float* b = buf + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                    convolution_2xN(w, p, a, K, dS, zero, src, bias + i, params + i * dp, b, d);
                for (; i < n1; i += m, w += m * dW, b += m * dB, d += m * dD)
                    convolution_2xM(w, p, a, K, dS, zero, src, bias + i, params + i * dp, b, d);
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNchwGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if (p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNchwGemm::SynetConvolution16bNchwGemm(const ConvParam& p)
            : Base::SynetConvolution16bNchwGemm(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                if (_is1x1)
                    _convert = Reorder16bNchwGemm1x1;
            }
            else
            {
                if (_is1x1)
                    _convert = Convert16bNchwGemm1x1;
            }
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, _alg, _convolutions); break;
            default: assert(0);
            }
        }
    }
#endif
}
