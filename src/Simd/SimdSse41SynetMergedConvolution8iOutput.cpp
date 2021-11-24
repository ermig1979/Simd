/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSse2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using OutputConvolutionPtr = Base::SynetMergedConvolution8i::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst, int first)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = a.maC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                    if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                    if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                    if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                    if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB + 0)), d01 = _mm_loadu_si128((__m128i*)(buf + 0 * dB + F));
                    if (M > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB + 0)), d11 = _mm_loadu_si128((__m128i*)(buf + 1 * dB + F));
                    if (M > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB + 0)), d21 = _mm_loadu_si128((__m128i*)(buf + 2 * dB + F));
                    if (M > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB + 0)), d31 = _mm_loadu_si128((__m128i*)(buf + 3 * dB + F));
                    if (M > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB + 0)), d41 = _mm_loadu_si128((__m128i*)(buf + 4 * dB + F));
                }
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        weight += DA;
                    }
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = _mm_setzero_si128();
                    if (M > 1) d10 = _mm_setzero_si128();
                    if (M > 2) d20 = _mm_setzero_si128();
                    if (M > 3) d30 = _mm_setzero_si128();
                    if (M > 4) d40 = _mm_setzero_si128();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB + 0));
                    if (M > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB + 0));
                    if (M > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB + 0));
                    if (M > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB + 0));
                    if (M > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB + 0));
                }
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                        weight += DA;
                    }
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst, int first);

        template<Term8iType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
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

        template<Term8iType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m128 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                _scale[0] = _mm_loadu_ps(scale + dc + 0);
                _scale[1] = _mm_loadu_ps(scale + dc + F);
                _shift[0] = _mm_loadu_ps(shift + dc + 0);
                _shift[1] = _mm_loadu_ps(shift + dc + F);
                const uint8_t* s = src;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                for (; i < n1; i += m, s += a.maC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                weight += DivHi(maC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam8i& p, OutputConvolutionPtr* output)
        {
            output[0] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iLast32f, type> : OutputConvolution1x1_2<Term8iLast8u, type>;
            output[1] = OutputConvolution1x1_2<Term8iInterim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam8i& p, OutputConvolutionPtr* output)
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
