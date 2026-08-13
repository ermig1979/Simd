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
        typedef Base::SynetConvolution16bNhwcGemmV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcGemmV0::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svbfloat16_t BroadcastBf16x2(const uint16_t* src)
        {
            return svreinterpret_bf16_u32(svdup_n_u32(uint32_t(src[0]) | (uint32_t(src[1]) << 16)));
        }

        SIMD_INLINE svbfloat16_t LoadBf16x2(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_bf16_u32(svld1_u32(mask, (const uint32_t*)src));
        }

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const float* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    Float32ToBFloat16(ps, p.srcC, row);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        static void Reorder16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint16_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    memcpy(row, ps, p.srcC * 2);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0,
            svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias0, param0, param1, mask);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0,
            svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias0, param0, param1, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, svfloat32_t val0, svfloat32_t val1,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask0, const svbool_t& mask1)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias0, param0, param1, mask0);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias1, param0, param1, mask1);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, svfloat32_t val0, svfloat32_t val1,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias0, param0, param1, svptrue_b32());
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias1, param0, param1, svwhilelt_b32((size_t)0, tail));
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNhwcGemm_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* buf, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t s0, w0, w1;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const uint16_t* weight1 = weight0 + a.bufK * F;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            if (dstC > F)
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
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w0 = LoadBf16x2(weight0, body);
                    w1 = LoadBf16x2(weight1, body);
                    if (M > 0)
                    {
                        s0 = BroadcastBf16x2(src0 + offs);
                        d00 = svbfdot_f32(d00, s0, w0);
                        d01 = svbfdot_f32(d01, s0, w1);
                    }
                    if (M > 1)
                    {
                        s0 = BroadcastBf16x2(src1 + offs);
                        d10 = svbfdot_f32(d10, s0, w0);
                        d11 = svbfdot_f32(d11, s0, w1);
                    }
                    if (M > 2)
                    {
                        s0 = BroadcastBf16x2(src2 + offs);
                        d20 = svbfdot_f32(d20, s0, w0);
                        d21 = svbfdot_f32(d21, s0, w1);
                    }
                    if (M > 3)
                    {
                        s0 = BroadcastBf16x2(src3 + offs);
                        d30 = svbfdot_f32(d30, s0, w0);
                        d31 = svbfdot_f32(d31, s0, w1);
                    }
                    if (M > 4)
                    {
                        s0 = BroadcastBf16x2(src4 + offs);
                        d40 = svbfdot_f32(d40, s0, w0);
                        d41 = svbfdot_f32(d41, s0, w1);
                    }
                    weight0 += DF;
                    weight1 += DF;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias0, bias1, param0, param1, body, body), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias0, bias1, param0, param1, body, body), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias0, bias1, param0, param1, body, body), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias0, bias1, param0, param1, body, body), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias0, bias1, param0, param1, body, body), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias0, bias1, param0, param1, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias0, bias1, param0, param1, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias0, bias1, param0, param1, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias0, bias1, param0, param1, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias0, bias1, param0, param1, dstC - F), dst += dD, buf += dB;
                }
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
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w0 = LoadBf16x2(weight0, body);
                    if (M > 0)
                    {
                        s0 = BroadcastBf16x2(src0 + offs);
                        d00 = svbfdot_f32(d00, s0, w0);
                    }
                    if (M > 1)
                    {
                        s0 = BroadcastBf16x2(src1 + offs);
                        d10 = svbfdot_f32(d10, s0, w0);
                    }
                    if (M > 2)
                    {
                        s0 = BroadcastBf16x2(src2 + offs);
                        d20 = svbfdot_f32(d20, s0, w0);
                    }
                    if (M > 3)
                    {
                        s0 = BroadcastBf16x2(src3 + offs);
                        d30 = svbfdot_f32(d30, s0, w0);
                    }
                    if (M > 4)
                    {
                        s0 = BroadcastBf16x2(src4 + offs);
                        d40 = svbfdot_f32(d40, s0, w0);
                    }
                    weight0 += DF;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias0, param0, param1, body), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias0, param0, param1, body), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias0, param0, param1, body), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias0, param0, param1, body), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias0, param0, param1, body), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias0, param0, param1, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias0, param0, param1, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias0, param0, param1, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias0, param0, param1, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias0, param0, param1, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*Convolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNhwcGemm_2xM_Ptr GetConvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNhwcGemm_2xM<term, type, 1>;
            case 2: return Convolution16bNhwcGemm_2xM<term, type, 2>;
            case 3: return Convolution16bNhwcGemm_2xM<term, type, 3>;
            case 4: return Convolution16bNhwcGemm_2xM<term, type, 4>;
            case 5: return Convolution16bNhwcGemm_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xN = GetConvolution16bNhwcGemm_2xM<term, type>(n);
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xM = GetConvolution16bNhwcGemm_2xM<term, type>(m);

            svfloat32_t param0 = svdup_n_f32(params[0]);
            svfloat32_t param1 = svdup_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                svfloat32_t bias0 = svld1_f32(body, bias + dc + 0);
                svfloat32_t bias1 = svld1_f32(body, bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(body, params + dc + 0);
                    param1 = svld1_f32(body, params + dc + F);
                }
                const uint16_t* s = src;
                float* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_2xN(s, p, a, srcC, dC, zero, weight, bias0, bias1, param0, param1, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_2xM(s, p, a, srcC, dC, zero, weight, bias0, bias1, param0, param1, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNhwcGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if (p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNhwcGemmV0::SynetConvolution16bNhwcGemmV0(const ConvParam& p)
            : Base::SynetConvolution16bNhwcGemmV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = Reorder16bNhwcGemm;
            }
            else
                _convert = Convert16bNhwcGemm;
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
