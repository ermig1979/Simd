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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using InputPtr = Base::SynetMergedConvolution16b::InputConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> 
        SIMD_INLINE void SaveInput1(float* dst, __m256 sum, const __m256* bias, const __m256* params)
        {
            _mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(sum, bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m256 sum0, __m256 sum1, const __m256* bias, const __m256* params)
        {
            _mm256_storeu_ps(dst0, Activate<type>(_mm256_add_ps(sum0, bias[0]), params, 0));
            _mm256_storeu_ps(dst1, Activate<type>(_mm256_add_ps(sum1, bias[1]), params, 1));
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint16_t* src0, const ConvParam& p,
            const AlgParam& a, size_t dstC, const uint16_t* weight0, const __m256* bias, const __m256* params, float* dst0, float* dst1)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm256_castsi256_ps(Bf16::MASK);
            size_t srcC = AlignHi(p.srcC, a.miK);
            const uint16_t* weight1 = weight0 + srcC * F;
            const uint16_t* src1 = src0 + 1 * srcC;
            const uint16_t* src2 = src0 + 2 * srcC;
            const uint16_t* src3 = src0 + 3 * srcC;
            const uint16_t* src4 = src0 + 4 * srcC;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs = 0, end = srcC; offs < end; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        w11 = _mm256_loadu_ps((float*)weight1);
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
                        weight0 += DF;
                        weight1 += DF;
                    }
                }
                else
                {
                    for (size_t offs = 0, end = srcC; offs < end; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight0);
                        w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                        w01 = _mm256_and_ps(w01, m);
                        w11 = _mm256_loadu_ps((float*)weight1);
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
                        weight0 += DF;
                        weight1 += DF;
                    }
                }
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, bias, params);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, bias, params);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, bias, params);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, bias, params);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, bias, params);
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps();
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (size_t offs = 0, end = srcC; offs < end; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight0);
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
                        weight0 += DF;
                    }
                }
                else
                {
                    for (size_t offs = 0, end = srcC; offs < end; offs += 2)
                    {
                        w01 = _mm256_loadu_ps((float*)weight0);
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
                        weight0 += DF;
                    }
                }
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, bias, params);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, bias, params);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, bias, params);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, bias, params);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstC,
            const uint16_t* weight0, const __m256* bias, const __m256* params, float* dst0, float* dst1);

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
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcC = AlignHi(p.srcC, a.miK), y0 = a.bufH[0] ? yBeg : 0;
            __m256 _bias[2], _params[2];
            _params[0] = _mm256_set1_ps(params[0]);
            _params[1] = _mm256_set1_ps(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 5;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
            InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
                }
                if (yInt > yBeg)
                {
                    const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                    float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    if (in < i1)
                        inputConvolution1x1_2xI(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                }
                if (yEnd > yInt)
                {
                    const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                    float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    if (en < e1)
                        inputConvolution1x1_2xE(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
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
