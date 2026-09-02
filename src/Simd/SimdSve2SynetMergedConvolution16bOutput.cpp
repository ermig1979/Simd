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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution16b::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias0, param0, param1, mask);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, size_t tail)
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

        //---------------------------------------------------------------------

        SIMD_INLINE svbfloat16_t BroadcastBf16x2(const uint16_t* src)
        {
            return svreinterpret_bf16_u32(svdup_n_u32(uint32_t(src[0]) | (uint32_t(src[1]) << 16)));
        }

        SIMD_INLINE svbfloat16_t LoadBf16x2(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_bf16_u32(svld1_u32(mask, (const uint32_t*)src));
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t s0, w0, w1;
            size_t dS = a.maC * p.strideX, dB = p.dstC, dD = p.dstC * a.elem[1];
            const uint16_t* weight1 = weight0 + AlignHi(srcC, 2) * F;
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
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias0, bias1, param0, param1, body, body), buf += dB, dst += dD;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias0, bias1, param0, param1, body, body), buf += dB, dst += dD;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias0, bias1, param0, param1, body, body), buf += dB, dst += dD;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias0, bias1, param0, param1, body, body), buf += dB, dst += dD;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias0, bias1, param0, param1, body, body), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias0, bias1, param0, param1, dstC - F), buf += dB, dst += dD;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias0, bias1, param0, param1, dstC - F), buf += dB, dst += dD;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias0, bias1, param0, param1, dstC - F), buf += dB, dst += dD;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias0, bias1, param0, param1, dstC - F), buf += dB, dst += dD;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias0, bias1, param0, param1, dstC - F), buf += dB, dst += dD;
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
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias0, param0, param1, body), buf += dB, dst += dD;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias0, param0, param1, body), buf += dB, dst += dD;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias0, param0, param1, body), buf += dB, dst += dD;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias0, param0, param1, body), buf += dB, dst += dD;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias0, param0, param1, body), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias0, param0, param1, dstC), buf += dB, dst += dD;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias0, param0, param1, dstC), buf += dB, dst += dD;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias0, param0, param1, dstC), buf += dB, dst += dD;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias0, param0, param1, dstC), buf += dB, dst += dD;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias0, param0, param1, dstC), buf += dB, dst += dD;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return OutputConvolution1x1_2xM<term, type, 1>;
            case 2: return OutputConvolution1x1_2xM<term, type, 2>;
            case 3: return OutputConvolution1x1_2xM<term, type, 3>;
            case 4: return OutputConvolution1x1_2xM<term, type, 4>;
            case 5: return OutputConvolution1x1_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, float*, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            svfloat32_t param0 = svdup_n_f32(params[0]);
            svfloat32_t param1 = svdup_n_f32(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                svfloat32_t bias0 = svld1_f32(body, bias + dc + 0);
                svfloat32_t bias1 = svld1_f32(body, bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(body, params + dc + 0);
                    param1 = svld1_f32(body, params + dc + F);
                }
                const uint16_t* s = src;
                float* b = buf + dc + yBeg * p.dstW * p.dstC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += p.dstC * n, d += p.dstC * a.elem[1] * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, zero, weight, bias0, bias1, param0, param1, b, d);
                for (; i < n1; i += m, s += a.maC * m, b += p.dstC * m, d += p.dstC * a.elem[1] * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, zero, weight, bias0, bias1, param0, param1, b, d);
                weight += AlignHi(maC, 2) * DF;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            if (p.dstT == SimdTensorData16b)
                output[0] = OutputConvolution1x1_2<Term16bLast16b, type>;
            else
                output[0] = OutputConvolution1x1_2<Term16bLast32f, type>;
            output[1] = OutputConvolution1x1_2<Term16bInterim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, output); break;
            case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, output); break;
            case SimdConvolutionActivationMish: SetOutput<SimdConvolutionActivationMish>(p, output); break;
            case SimdConvolutionActivationHardSigmoid: SetOutput<SimdConvolutionActivationHardSigmoid>(p, output); break;
            case SimdConvolutionActivationSwish: SetOutput<SimdConvolutionActivationSwish>(p, output); break;
            case SimdConvolutionActivationGelu: SetOutput<SimdConvolutionActivationGelu>(p, output); break;
            }
        }
    }
#endif
}
