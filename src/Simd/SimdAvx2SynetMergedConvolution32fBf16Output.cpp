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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<TermBf16Type term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC,
            const uint16_t* weight, const __m256* bias, const __m256* params, float* dst, int zero)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm256_castsi256_ps(Bf16::MASK);
            size_t dS = a.maC * p.strideX, dD = p.dstC;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0), d01 = _mm256_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0), d11 = _mm256_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0), d21 = _mm256_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0), d31 = _mm256_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0), d41 = _mm256_loadu_ps(dst + 4 * dD + F);
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight + 0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        w11 = _mm256_loadu_ps((float*)weight + F);
                        w10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w11), Base::Bf16::SHIFT));
                        w11 = _mm256_and_ps(w11, m);
                        if (M > 0)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                            d00 = Fmadd<true>(s0, w00, d00); d01 = Fmadd<true>(s0, w10, d01);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                            d00 = Fmadd<true>(s0, w01, d00); d01 = Fmadd<true>(s0, w11, d01);
                        }
                        if (M > 1)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                            d10 = Fmadd<true>(s0, w00, d10); d11 = Fmadd<true>(s0, w10, d11);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                            d10 = Fmadd<true>(s0, w01, d10); d11 = Fmadd<true>(s0, w11, d11);
                        }
                        if (M > 2)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                            d20 = Fmadd<true>(s0, w00, d20); d21 = Fmadd<true>(s0, w10, d21);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                            d20 = Fmadd<true>(s0, w01, d20); d21 = Fmadd<true>(s0, w11, d21);
                        }
                        if (M > 3)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                            d30 = Fmadd<true>(s0, w00, d30); d31 = Fmadd<true>(s0, w10, d31);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                            d30 = Fmadd<true>(s0, w01, d30); d31 = Fmadd<true>(s0, w11, d31);
                        }
                        if (M > 4)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                            d40 = Fmadd<true>(s0, w00, d40); d41 = Fmadd<true>(s0, w10, d41);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                            d40 = Fmadd<true>(s0, w01, d40); d41 = Fmadd<true>(s0, w11, d41);
                        }
                        weight += QF;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight + 0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        w11 = _mm256_loadu_ps((float*)weight + F);
                        w10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w11), Base::Bf16::SHIFT));
                        w11 = _mm256_and_ps(w11, m);
                        if (M > 0)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                            d00 = Fmadd<false>(s0, w00, d00); d01 = Fmadd<false>(s0, w10, d01);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                            d00 = Fmadd<false>(s0, w01, d00); d01 = Fmadd<false>(s0, w11, d01);
                        }
                        if (M > 1)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                            d10 = Fmadd<false>(s0, w00, d10); d11 = Fmadd<false>(s0, w10, d11);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                            d10 = Fmadd<false>(s0, w01, d10); d11 = Fmadd<false>(s0, w11, d11);
                        }
                        if (M > 2)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                            d20 = Fmadd<false>(s0, w00, d20); d21 = Fmadd<false>(s0, w10, d21);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                            d20 = Fmadd<false>(s0, w01, d20); d21 = Fmadd<false>(s0, w11, d21);
                        }
                        if (M > 3)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                            d30 = Fmadd<false>(s0, w00, d30); d31 = Fmadd<false>(s0, w10, d31);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                            d30 = Fmadd<false>(s0, w01, d30); d31 = Fmadd<false>(s0, w11, d31);
                        }
                        if (M > 4)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                            d40 = Fmadd<false>(s0, w00, d40); d41 = Fmadd<false>(s0, w10, d41);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                            d40 = Fmadd<false>(s0, w01, d40); d41 = Fmadd<false>(s0, w11, d41);
                        }
                        weight += QF;
                    }
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>((uint16_t*)dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>((uint16_t*)dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>((uint16_t*)dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>((uint16_t*)dst, d30, d31, bias, params), dst += dD;
                    if (M > 4) Save2<term, type>((uint16_t*)dst, d40, d41, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save2<term, type>((uint16_t*)dst, d00, d01, bias, params, dstC - F), dst += dD;
                    if (M > 1) Save2<term, type>((uint16_t*)dst, d10, d11, bias, params, dstC - F), dst += dD;
                    if (M > 2) Save2<term, type>((uint16_t*)dst, d20, d21, bias, params, dstC - F), dst += dD;
                    if (M > 3) Save2<term, type>((uint16_t*)dst, d30, d31, bias, params, dstC - F), dst += dD;
                    if (M > 4) Save2<term, type>((uint16_t*)dst, d40, d41, bias, params, dstC - F), dst += dD;
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0);
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight + 0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        if (M > 0)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                            d00 = Fmadd<true>(s0, w00, d00);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                            d00 = Fmadd<true>(s0, w01, d00);
                        }
                        if (M > 1)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                            d10 = Fmadd<true>(s0, w00, d10);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                            d10 = Fmadd<true>(s0, w01, d10);
                        }
                        if (M > 2)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                            d20 = Fmadd<true>(s0, w00, d20);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                            d20 = Fmadd<true>(s0, w01, d20);
                        }
                        if (M > 3)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                            d30 = Fmadd<true>(s0, w00, d30);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                            d30 = Fmadd<true>(s0, w01, d30);
                        }
                        if (M > 4)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                            d40 = Fmadd<true>(s0, w00, d40);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                            d40 = Fmadd<true>(s0, w01, d40);
                        }
                        weight += QF;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight + 0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        if (M > 0)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                            d00 = Fmadd<false>(s0, w00, d00);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                            d00 = Fmadd<false>(s0, w01, d00);
                        }
                        if (M > 1)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                            d10 = Fmadd<false>(s0, w00, d10);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                            d10 = Fmadd<false>(s0, w01, d10);
                        }
                        if (M > 2)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                            d20 = Fmadd<false>(s0, w00, d20);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                            d20 = Fmadd<false>(s0, w01, d20);
                        }
                        if (M > 3)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                            d30 = Fmadd<false>(s0, w00, d30);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                            d30 = Fmadd<false>(s0, w01, d30);
                        }
                        if (M > 4)
                        {
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                            d40 = Fmadd<false>(s0, w00, d40);
                            s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                            d40 = Fmadd<false>(s0, w01, d40);
                        }
                        weight += QF;
                    }
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>((uint16_t*)dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>((uint16_t*)dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>((uint16_t*)dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>((uint16_t*)dst, d30, bias, params), dst += dD;
                    if (M > 4) Save1<term, type>((uint16_t*)dst, d40, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>((uint16_t*)dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>((uint16_t*)dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>((uint16_t*)dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>((uint16_t*)dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>((uint16_t*)dst, d40, bias, params, dstC), dst += dD;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC,
            const uint16_t* weight0, const __m256* bias, const __m256* params, float* dst, int zero);

        template<TermBf16Type term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
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

        template<TermBf16Type term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src,
            const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight,
            const float* bias, const float* params, float* dst, int zero)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m256 _bias[2], _params[2];
            _params[0] = _mm256_set1_ps(params[0]);
            _params[1] = _mm256_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
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
