/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fBf16Common.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bf16
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<TermBf16Type term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst, int zero)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61,
                d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1;
            __m512bh s0, w0, w1;
            size_t dS = a.maC * p.strideX, dD = p.dstC;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            const uint16_t* src6 = src0 + 6 * dS;
            if (dstC > F)
            {
                __mmask16 tail = TailMask16(dstC - F);
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_loadu_ps(dst + 0x0 * dD + 0), d01 = _mm512_maskz_loadu_ps(tail, dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_ps(dst + 0x1 * dD + 0), d11 = _mm512_maskz_loadu_ps(tail, dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_ps(dst + 0x2 * dD + 0), d21 = _mm512_maskz_loadu_ps(tail, dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_ps(dst + 0x3 * dD + 0), d31 = _mm512_maskz_loadu_ps(tail, dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_ps(dst + 0x4 * dD + 0), d41 = _mm512_maskz_loadu_ps(tail, dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_ps(dst + 0x5 * dD + 0), d51 = _mm512_maskz_loadu_ps(tail, dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_ps(dst + 0x6 * dD + 0), d61 = _mm512_maskz_loadu_ps(tail, dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_ps(dst + 0x7 * dD + 0), d71 = _mm512_maskz_loadu_ps(tail, dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_ps(dst + 0x8 * dD + 0), d81 = _mm512_maskz_loadu_ps(tail, dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_ps(dst + 0x9 * dD + 0), d91 = _mm512_maskz_loadu_ps(tail, dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = _mm512_loadu_ps(dst + 0xa * dD + 0), da1 = _mm512_maskz_loadu_ps(tail, dst + 0xa * dD + F);
                    if (M > 0xb) db0 = _mm512_loadu_ps(dst + 0xb * dD + 0), db1 = _mm512_maskz_loadu_ps(tail, dst + 0xb * dD + F);
                    if (M > 0xc) dc0 = _mm512_loadu_ps(dst + 0xc * dD + 0), dc1 = _mm512_maskz_loadu_ps(tail, dst + 0xc * dD + F);
                    if (M > 0xd) dd0 = _mm512_loadu_ps(dst + 0xd * dD + 0), dd1 = _mm512_maskz_loadu_ps(tail, dst + 0xd * dD + F);
                }
                for (size_t offs0 = 0, offs7 = 7 * dS; offs0 < srcC; offs0 += 2, offs7 += 2)
                {
                    w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                    w1 = (__m512bh)_mm512_loadu_si512(weight + 1 * DF);
                    if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0), d01 = _mm512_dpbf16_ps(d01, s0, w1);
                    if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0), d11 = _mm512_dpbf16_ps(d11, s0, w1);
                    if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0), d21 = _mm512_dpbf16_ps(d21, s0, w1);
                    if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0), d31 = _mm512_dpbf16_ps(d31, s0, w1);
                    if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0), d41 = _mm512_dpbf16_ps(d41, s0, w1);
                    if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0), d51 = _mm512_dpbf16_ps(d51, s0, w1);
                    if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0), d61 = _mm512_dpbf16_ps(d61, s0, w1);
                    if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0), d71 = _mm512_dpbf16_ps(d71, s0, w1);
                    if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0), d81 = _mm512_dpbf16_ps(d81, s0, w1);
                    if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0), d91 = _mm512_dpbf16_ps(d91, s0, w1);
                    if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0), da1 = _mm512_dpbf16_ps(da1, s0, w1);
                    if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0), db1 = _mm512_dpbf16_ps(db1, s0, w1);
                    if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0), dc1 = _mm512_dpbf16_ps(dc1, s0, w1);
                    if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0), dd1 = _mm512_dpbf16_ps(dd1, s0, w1);
                    weight += QF;
                }
                if (M > 0x0) Save2<term, type>((uint16_t*)dst, d00, d01, bias, params, tail), dst += dD;
                if (M > 0x1) Save2<term, type>((uint16_t*)dst, d10, d11, bias, params, tail), dst += dD;
                if (M > 0x2) Save2<term, type>((uint16_t*)dst, d20, d21, bias, params, tail), dst += dD;
                if (M > 0x3) Save2<term, type>((uint16_t*)dst, d30, d31, bias, params, tail), dst += dD;
                if (M > 0x4) Save2<term, type>((uint16_t*)dst, d40, d41, bias, params, tail), dst += dD;
                if (M > 0x5) Save2<term, type>((uint16_t*)dst, d50, d51, bias, params, tail), dst += dD;
                if (M > 0x6) Save2<term, type>((uint16_t*)dst, d60, d61, bias, params, tail), dst += dD;
                if (M > 0x7) Save2<term, type>((uint16_t*)dst, d70, d71, bias, params, tail), dst += dD;
                if (M > 0x8) Save2<term, type>((uint16_t*)dst, d80, d81, bias, params, tail), dst += dD;
                if (M > 0x9) Save2<term, type>((uint16_t*)dst, d90, d91, bias, params, tail), dst += dD;
                if (M > 0xa) Save2<term, type>((uint16_t*)dst, da0, da1, bias, params, tail), dst += dD;
                if (M > 0xb) Save2<term, type>((uint16_t*)dst, db0, db1, bias, params, tail), dst += dD;
                if (M > 0xc) Save2<term, type>((uint16_t*)dst, dc0, dc1, bias, params, tail), dst += dD;
                if (M > 0xd) Save2<term, type>((uint16_t*)dst, dd0, dd1, bias, params, tail), dst += dD;
            }
            else
            {
                __mmask16 tail = TailMask16(dstC);
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                    if (M > 0xc) dc0 = _mm512_setzero_ps();
                    if (M > 0xd) dd0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_maskz_loadu_ps(tail, dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_maskz_loadu_ps(tail, dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_maskz_loadu_ps(tail, dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_maskz_loadu_ps(tail, dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_maskz_loadu_ps(tail, dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_maskz_loadu_ps(tail, dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_maskz_loadu_ps(tail, dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_maskz_loadu_ps(tail, dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_maskz_loadu_ps(tail, dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_maskz_loadu_ps(tail, dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = _mm512_maskz_loadu_ps(tail, dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = _mm512_maskz_loadu_ps(tail, dst + 0xb * dD + 0);
                    if (M > 0xc) dc0 = _mm512_maskz_loadu_ps(tail, dst + 0xc * dD + 0);
                    if (M > 0xd) dd0 = _mm512_maskz_loadu_ps(tail, dst + 0xd * dD + 0);
                }
                for (size_t offs0 = 0, offs7 = 7 * dS; offs0 < srcC; offs0 += 2, offs7 += 2)
                {
                    w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                    if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0);
                    if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0);
                    if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0);
                    if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0);
                    if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0);
                    if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0);
                    if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0);
                    if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0);
                    if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0);
                    if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0);
                    if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0);
                    if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0);
                    if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0);
                    if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0);
                    weight += QF;
                }
                if (M > 0x0) Save1<term, type>((uint16_t*)dst, d00, bias, params, tail), dst += dD;
                if (M > 0x1) Save1<term, type>((uint16_t*)dst, d10, bias, params, tail), dst += dD;
                if (M > 0x2) Save1<term, type>((uint16_t*)dst, d20, bias, params, tail), dst += dD;
                if (M > 0x3) Save1<term, type>((uint16_t*)dst, d30, bias, params, tail), dst += dD;
                if (M > 0x4) Save1<term, type>((uint16_t*)dst, d40, bias, params, tail), dst += dD;
                if (M > 0x5) Save1<term, type>((uint16_t*)dst, d50, bias, params, tail), dst += dD;
                if (M > 0x6) Save1<term, type>((uint16_t*)dst, d60, bias, params, tail), dst += dD;
                if (M > 0x7) Save1<term, type>((uint16_t*)dst, d70, bias, params, tail), dst += dD;
                if (M > 0x8) Save1<term, type>((uint16_t*)dst, d80, bias, params, tail), dst += dD;
                if (M > 0x9) Save1<term, type>((uint16_t*)dst, d90, bias, params, tail), dst += dD;
                if (M > 0xa) Save1<term, type>((uint16_t*)dst, da0, bias, params, tail), dst += dD;
                if (M > 0xb) Save1<term, type>((uint16_t*)dst, db0, bias, params, tail), dst += dD;
                if (M > 0xc) Save1<term, type>((uint16_t*)dst, dc0, bias, params, tail), dst += dD;
                if (M > 0xd) Save1<term, type>((uint16_t*)dst, dd0, bias, params, tail), dst += dD;
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC,
            const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst, int zero);

        template<TermBf16Type term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return OutputConvolution1x1_2xM<term, type, 0x1>;
            case 0x2: return OutputConvolution1x1_2xM<term, type, 0x2>;
            case 0x3: return OutputConvolution1x1_2xM<term, type, 0x3>;
            case 0x4: return OutputConvolution1x1_2xM<term, type, 0x4>;
            case 0x5: return OutputConvolution1x1_2xM<term, type, 0x5>;
            case 0x6: return OutputConvolution1x1_2xM<term, type, 0x6>;
            case 0x7: return OutputConvolution1x1_2xM<term, type, 0x7>;
            case 0x8: return OutputConvolution1x1_2xM<term, type, 0x8>;
            case 0x9: return OutputConvolution1x1_2xM<term, type, 0x9>;
            case 0xa: return OutputConvolution1x1_2xM<term, type, 0xa>;
            case 0xb: return OutputConvolution1x1_2xM<term, type, 0xb>;
            case 0xc: return OutputConvolution1x1_2xM<term, type, 0xc>;
            case 0xd: return OutputConvolution1x1_2xM<term, type, 0xd>;
            case 0xe: return OutputConvolution1x1_2xM<term, type, 0xe>;
            }
            assert(0);
            return NULL;
        }

        template<TermBf16Type term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src,
            const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight,
            const float* bias, const float* params, float* dst, int zero)
        {
            size_t n = 14, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                const uint16_t* s = src;
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, d += p.dstC * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, weight, _bias, _params, d, zero);
                for (; i < n1; i += m, s += a.maC * m, d += p.dstC * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, weight, _bias, _params, d, zero);
                weight += DivHi(maC, 2) * QF;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam32f& p, OutputPtr* output)
        {
            output[0] = OutputConvolution1x1_2<TermBf16Last32f, type>;
            output[1] = OutputConvolution1x1_2<TermBf16Interim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam32f& p, OutputPtr* output)
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
            }
        }
    }
#endif
}
