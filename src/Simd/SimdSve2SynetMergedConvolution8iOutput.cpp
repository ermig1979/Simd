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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using OutputConvolutionPtr = Base::SynetMergedConvolution8i::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight,
            const svfloat32_t& norm0, const svfloat32_t& norm1, const svfloat32_t& bias0, const svfloat32_t& bias1,
            const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& scale0, const svfloat32_t& scale1,
            const svfloat32_t& shift0, const svfloat32_t& shift1, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = svcntw(), A = F * 4, DA = 2 * A;
            size_t dS = a.maC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const svbool_t body8 = svptrue_b8();
            const svbool_t tail0 = svwhilelt_b32((size_t)0, Simd::Min(F, dstC));
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svuint8_t s0;
            svint8_t w0, w1;
            if (dstC > F)
            {
                const svbool_t tail1 = svwhilelt_b32((size_t)0, dstC - F);
                if (first)
                {
                    if (M > 0) d00 = svdup_n_s32(0), d01 = svdup_n_s32(0);
                    if (M > 1) d10 = svdup_n_s32(0), d11 = svdup_n_s32(0);
                    if (M > 2) d20 = svdup_n_s32(0), d21 = svdup_n_s32(0);
                    if (M > 3) d30 = svdup_n_s32(0), d31 = svdup_n_s32(0);
                    if (M > 4) d40 = svdup_n_s32(0), d41 = svdup_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = svld1_s32(tail0, buf + 0 * dB + 0), d01 = svld1_s32(tail1, buf + 0 * dB + F);
                    if (M > 1) d10 = svld1_s32(tail0, buf + 1 * dB + 0), d11 = svld1_s32(tail1, buf + 1 * dB + F);
                    if (M > 2) d20 = svld1_s32(tail0, buf + 2 * dB + 0), d21 = svld1_s32(tail1, buf + 2 * dB + F);
                    if (M > 3) d30 = svld1_s32(tail0, buf + 3 * dB + 0), d31 = svld1_s32(tail1, buf + 3 * dB + F);
                    if (M > 4) d40 = svld1_s32(tail0, buf + 4 * dB + 0), d41 = svld1_s32(tail1, buf + 4 * dB + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = svld1_s8(body8, weight + 0);
                    w1 = svld1_s8(body8, weight + A);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                    weight += DA;
                }
                if (dstC == 2 * F)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, a.upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = svdup_n_s32(0);
                    if (M > 1) d10 = svdup_n_s32(0);
                    if (M > 2) d20 = svdup_n_s32(0);
                    if (M > 3) d30 = svdup_n_s32(0);
                    if (M > 4) d40 = svdup_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = svld1_s32(tail0, buf + 0 * dB + 0);
                    if (M > 1) d10 = svld1_s32(tail0, buf + 1 * dB + 0);
                    if (M > 2) d20 = svld1_s32(tail0, buf + 2 * dB + 0);
                    if (M > 3) d30 = svld1_s32(tail0, buf + 3 * dB + 0);
                    if (M > 4) d40 = svld1_s32(tail0, buf + 4 * dB + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = svld1_s8(body8, weight + 0);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0);
                    weight += DA;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm0, bias0, param0, param1, scale0, shift0, a.upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm0, bias0, param0, param1, scale0, shift0, a.upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm0, bias0, param0, param1, scale0, shift0, a.upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm0, bias0, param0, param1, scale0, shift0, a.upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm0, bias0, param0, param1, scale0, shift0, a.upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm0, bias0, param0, param1, scale0, shift0, a.upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm0, bias0, param0, param1, scale0, shift0, a.upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm0, bias0, param0, param1, scale0, shift0, a.upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm0, bias0, param0, param1, scale0, shift0, a.upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm0, bias0, param0, param1, scale0, shift0, a.upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const svfloat32_t& norm0, const svfloat32_t& norm1, const svfloat32_t& bias0, const svfloat32_t& bias1,
            const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& scale0, const svfloat32_t& scale1,
            const svfloat32_t& shift0, const svfloat32_t& shift1, int32_t* buf, uint8_t* dst, int first);

        template<Term8iType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(const ConvParam& p, size_t M)
        {
            if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
            {
                switch (M)
                {
                case 0: return NULL;
                case 1: return OutputConvolution1x1_2xM<true, term, type, 1>;
                case 2: return OutputConvolution1x1_2xM<true, term, type, 2>;
                case 3: return OutputConvolution1x1_2xM<true, term, type, 3>;
                case 4: return OutputConvolution1x1_2xM<true, term, type, 4>;
                case 5: return OutputConvolution1x1_2xM<true, term, type, 5>;
                }
            }
            else
            {
                switch (M)
                {
                case 0: return NULL;
                case 1: return OutputConvolution1x1_2xM<false, term, type, 1>;
                case 2: return OutputConvolution1x1_2xM<false, term, type, 2>;
                case 3: return OutputConvolution1x1_2xM<false, term, type, 3>;
                case 4: return OutputConvolution1x1_2xM<false, term, type, 4>;
                case 5: return OutputConvolution1x1_2xM<false, term, type, 5>;
                }
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint8_t* src,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = svcntw(), A = F * 4, DA = 2 * A, DF = 2 * F;
            const svbool_t body = svptrue_b32();
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(p, n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(p, m);
            svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                svfloat32_t norm0 = svld1_f32(body, norm + dc + 0);
                svfloat32_t norm1 = svld1_f32(body, norm + dc + F);
                svfloat32_t bias0 = svld1_f32(body, bias + dc + 0);
                svfloat32_t bias1 = svld1_f32(body, bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(body, params + dc + 0);
                    param1 = svld1_f32(body, params + dc + F);
                }
                svfloat32_t scale0 = svld1_f32(body, scale + dc + 0);
                svfloat32_t scale1 = svld1_f32(body, scale + dc + F);
                svfloat32_t shift0 = svld1_f32(body, shift + dc + 0);
                svfloat32_t shift1 = svld1_f32(body, shift + dc + F);
                const uint8_t* s = src;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, weight, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, b, d, first);
                for (; i < n1; i += m, s += a.maC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, weight, norm0, norm1, bias0, bias1, param0, param1, scale0, scale1, shift0, shift1, b, d, first);
                weight += DivHi(maC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam& p, OutputConvolutionPtr* output)
        {
            output[0] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iLast32f, type> : OutputConvolution1x1_2<Term8iLast8u, type>;
            output[1] = OutputConvolution1x1_2<Term8iInterim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam& p, OutputConvolutionPtr* output)
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
