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
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using InputPtr = Base::SynetMergedConvolution16b::InputConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svbfloat16_t BroadcastBf16x2(const uint16_t* src)
        {
            return svreinterpret_bf16_u32(svdup_n_u32(uint32_t(src[0]) | (uint32_t(src[1]) << 16)));
        }

        SIMD_INLINE svbfloat16_t LoadBf16x2(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_bf16_u32(svld1_u32(mask, (const uint32_t*)src));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveInput1(float* dst, svfloat32_t sum, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, size_t index, const svbool_t& mask)
        {
            svst1_f32(mask, dst, Activate<type>(svadd_f32_x(mask, sum, bias), param0, param1, index, mask));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveInput2(float* dst0, float* dst1, svfloat32_t sum0, svfloat32_t sum1,
            svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask0, const svbool_t& mask1)
        {
            SaveInput1<type>(dst0, sum0, bias0, param0, param1, 0, mask0);
            SaveInput1<type>(dst1, sum1, bias1, param0, param1, 1, mask1);
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint16_t* src0, const ConvParam& p,
            const AlgParam& a, size_t dstC, const uint16_t* weight0, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t s0, w0, w1;
            size_t srcC = AlignHi(p.srcC, a.miK);
            const uint16_t* weight1 = weight0 + srcC * F;
            const uint16_t* src1 = src0 + 1 * srcC;
            const uint16_t* src2 = src0 + 2 * srcC;
            const uint16_t* src3 = src0 + 3 * srcC;
            const uint16_t* src4 = src0 + 4 * srcC;
            if (dstC > F)
            {
                if (M > 0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                if (M > 1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                if (M > 2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                if (M > 3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                if (M > 4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                for (size_t offs = 0, end = srcC; offs < end; offs += 2)
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
                svbool_t mask1 = (dstC == DF) ? body : svwhilelt_b32((size_t)0, dstC - F);
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, bias0, bias1, param0, param1, body, mask1);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, bias0, bias1, param0, param1, body, mask1);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, bias0, bias1, param0, param1, body, mask1);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, bias0, bias1, param0, param1, body, mask1);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, bias0, bias1, param0, param1, body, mask1);
            }
            else
            {
                if (M > 0) d00 = svdup_n_f32(0.0f);
                if (M > 1) d10 = svdup_n_f32(0.0f);
                if (M > 2) d20 = svdup_n_f32(0.0f);
                if (M > 3) d30 = svdup_n_f32(0.0f);
                if (M > 4) d40 = svdup_n_f32(0.0f);
                for (size_t offs = 0, end = srcC; offs < end; offs += 2)
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
                svbool_t mask0 = (dstC == F) ? body : svwhilelt_b32((size_t)0, dstC);
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, bias0, param0, param1, 0, mask0);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, bias0, param0, param1, 0, mask0);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, bias0, param0, param1, 0, mask0);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, bias0, param0, param1, 0, mask0);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, bias0, param0, param1, 0, mask0);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstC,
            const uint16_t* weight0, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InputConvolution1x1_2xM<type, 1>;
            case 2: return InputConvolution1x1_2xM<type, 2>;
            case 3: return InputConvolution1x1_2xM<type, 3>;
            case 4: return InputConvolution1x1_2xM<type, 4>;
            case 5: return InputConvolution1x1_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint16_t* src, const ConvParam& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* sum, float* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcC = AlignHi(p.srcC, a.miK), y0 = a.bufH[0] ? yBeg : 0;
            svfloat32_t param0 = svdup_n_f32(params[0]);
            svfloat32_t param1 = svdup_n_f32(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 5;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                svfloat32_t bias0 = svld1_f32(body, bias + dc + 0);
                svfloat32_t bias1 = svld1_f32(body, bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(body, params + dc + 0);
                    param1 = svld1_f32(body, params + dc + F);
                }
                if (yInt > yBeg)
                {
                    const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                    float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        inputConvolution1x1_2xN(src0, p, a, dC, weight, bias0, bias1, param0, param1, dst0, dst1);
                    if (in < i1)
                        inputConvolution1x1_2xI(src0, p, a, dC, weight, bias0, bias1, param0, param1, dst0, dst1);
                }
                if (yEnd > yInt)
                {
                    const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                    float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        inputConvolution1x1_2xN(src0, p, a, dC, weight, bias0, bias1, param0, param1, dst0, dst1);
                    if (en < e1)
                        inputConvolution1x1_2xE(src0, p, a, dC, weight, bias0, bias1, param0, param1, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += srcC * DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam& p, InputPtr& input)
        {
            if (Is1x1(p))
                input = InputConvolution1x1_2<type>;
            else
                assert(0);
        }

        void SetInput(const ConvParam& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInput<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            case SimdConvolutionActivationGelu: SetInput<SimdConvolutionActivationGelu>(p, input); break;
            }
        }
    }
#endif
}
