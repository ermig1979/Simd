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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512vnni
    {
        typedef Base::SynetQuantizedConvolutionNhwcGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedConvolutionNhwcGemm_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const int8_t* weight1 = weight0 + a.bufK * F;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm512_loadu_si512(buf + 0x0 * dB + 0), d01 = _mm512_loadu_si512(buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = _mm512_loadu_si512(buf + 0x1 * dB + 0), d11 = _mm512_loadu_si512(buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = _mm512_loadu_si512(buf + 0x2 * dB + 0), d21 = _mm512_loadu_si512(buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = _mm512_loadu_si512(buf + 0x3 * dB + 0), d31 = _mm512_loadu_si512(buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = _mm512_loadu_si512(buf + 0x4 * dB + 0), d41 = _mm512_loadu_si512(buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = _mm512_loadu_si512(buf + 0x5 * dB + 0), d51 = _mm512_loadu_si512(buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = _mm512_loadu_si512(buf + 0x6 * dB + 0), d61 = _mm512_loadu_si512(buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = _mm512_loadu_si512(buf + 0x7 * dB + 0), d71 = _mm512_loadu_si512(buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = _mm512_loadu_si512(buf + 0x8 * dB + 0), d81 = _mm512_loadu_si512(buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = _mm512_loadu_si512(buf + 0x9 * dB + 0), d91 = _mm512_loadu_si512(buf + 0x9 * dB + F);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(buf + 0xA * dB + 0), dA1 = _mm512_loadu_si512(buf + 0xA * dB + F);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(buf + 0xB * dB + 0), dB1 = _mm512_loadu_si512(buf + 0xB * dB + F);
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                    if (M > 0x1) d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                    if (M > 0x2) d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                    if (M > 0x3) d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                    if (M > 0x4) d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                    if (M > 0x5) d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                    if (M > 0x6) d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                    if (M > 0x7) d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                    if (M > 0x8) d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                    if (M > 0x9) d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                    if (M > 0xA) dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                    if (M > 0xB) dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    w1 = _mm512_loadu_si512((__m512i*)weight1);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0), Madd4<false>(d51, s0, w1);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0), Madd4<false>(d61, s0, w1);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0), Madd4<false>(d71, s0, w1);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0), Madd4<false>(d81, s0, w1);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0), Madd4<false>(d91, s0, w1);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0), Madd4<false>(dA1, s0, w1);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0), Madd4<false>(dB1, s0, w1);
                    weight0 += A, weight1 += A;
                }
                __mmask16 tail = TailMask16(dstC - F);
                if (M > 0x0) Save2<term>(dst, buf, d00, d01, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x1) Save2<term>(dst, buf, d10, d11, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x2) Save2<term>(dst, buf, d20, d21, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x3) Save2<term>(dst, buf, d30, d31, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x4) Save2<term>(dst, buf, d40, d41, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x5) Save2<term>(dst, buf, d50, d51, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x6) Save2<term>(dst, buf, d60, d61, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x7) Save2<term>(dst, buf, d70, d71, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x8) Save2<term>(dst, buf, d80, d81, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x9) Save2<term>(dst, buf, d90, d91, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0xA) Save2<term>(dst, buf, dA0, dA1, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0xB) Save2<term>(dst, buf, dB0, dB1, bias, norm, zero, tail), dst += dD, buf += dB;
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm512_loadu_si512(buf + 0x0 * dB + 0);
                    if (M > 0x1) d10 = _mm512_loadu_si512(buf + 0x1 * dB + 0);
                    if (M > 0x2) d20 = _mm512_loadu_si512(buf + 0x2 * dB + 0);
                    if (M > 0x3) d30 = _mm512_loadu_si512(buf + 0x3 * dB + 0);
                    if (M > 0x4) d40 = _mm512_loadu_si512(buf + 0x4 * dB + 0);
                    if (M > 0x5) d50 = _mm512_loadu_si512(buf + 0x5 * dB + 0);
                    if (M > 0x6) d60 = _mm512_loadu_si512(buf + 0x6 * dB + 0);
                    if (M > 0x7) d70 = _mm512_loadu_si512(buf + 0x7 * dB + 0);
                    if (M > 0x8) d80 = _mm512_loadu_si512(buf + 0x8 * dB + 0);
                    if (M > 0x9) d90 = _mm512_loadu_si512(buf + 0x9 * dB + 0);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(buf + 0xA * dB + 0);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(buf + 0xB * dB + 0);
                }
                else
                {
                    if (M > 0x0) d00 = _mm512_setzero_si512();
                    if (M > 0x1) d10 = _mm512_setzero_si512();
                    if (M > 0x2) d20 = _mm512_setzero_si512();
                    if (M > 0x3) d30 = _mm512_setzero_si512();
                    if (M > 0x4) d40 = _mm512_setzero_si512();
                    if (M > 0x5) d50 = _mm512_setzero_si512();
                    if (M > 0x6) d60 = _mm512_setzero_si512();
                    if (M > 0x7) d70 = _mm512_setzero_si512();
                    if (M > 0x8) d80 = _mm512_setzero_si512();
                    if (M > 0x9) d90 = _mm512_setzero_si512();
                    if (M > 0xA) dA0 = _mm512_setzero_si512();
                    if (M > 0xB) dB0 = _mm512_setzero_si512();
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0);
                    weight0 += A;
                }
                __mmask16 tail = TailMask16(dstC);
                if (M > 0x0) Save1<term>(dst, buf, d00, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x1) Save1<term>(dst, buf, d10, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x2) Save1<term>(dst, buf, d20, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x3) Save1<term>(dst, buf, d30, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x4) Save1<term>(dst, buf, d40, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x5) Save1<term>(dst, buf, d50, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x6) Save1<term>(dst, buf, d60, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x7) Save1<term>(dst, buf, d70, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x8) Save1<term>(dst, buf, d80, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0x9) Save1<term>(dst, buf, d90, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0xA) Save1<term>(dst, buf, dA0, bias, norm, zero, tail), dst += dD, buf += dB;
                if (M > 0xB) Save1<term>(dst, buf, dB0, bias, norm, zero, tail), dst += dD, buf += dB;
            }
        }

        typedef void(*QuantizedConvolutionNhwcGemm_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> QuantizedConvolutionNhwcGemm_2xM_Ptr GetQuantizedConvolutionNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedConvolutionNhwcGemm_2xM<term, 0x1>;
            case 0x2: return QuantizedConvolutionNhwcGemm_2xM<term, 0x2>;
            case 0x3: return QuantizedConvolutionNhwcGemm_2xM<term, 0x3>;
            case 0x4: return QuantizedConvolutionNhwcGemm_2xM<term, 0x4>;
            case 0x5: return QuantizedConvolutionNhwcGemm_2xM<term, 0x5>;
            case 0x6: return QuantizedConvolutionNhwcGemm_2xM<term, 0x6>;
            case 0x7: return QuantizedConvolutionNhwcGemm_2xM<term, 0x7>;
            case 0x8: return QuantizedConvolutionNhwcGemm_2xM<term, 0x8>;
            case 0x9: return QuantizedConvolutionNhwcGemm_2xM<term, 0x9>;
            case 0xA: return QuantizedConvolutionNhwcGemm_2xM<term, 0xA>;
            case 0xB: return QuantizedConvolutionNhwcGemm_2xM<term, 0xB>;
            case 0xC: return QuantizedConvolutionNhwcGemm_2xM<term, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedConvolutionNhwcGemm_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH,
            size_t srcC, int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            QuantizedConvolutionNhwcGemm_2xM_Ptr convolution_2xN = GetQuantizedConvolutionNhwcGemm_2xM<term>(n);
            QuantizedConvolutionNhwcGemm_2xM_Ptr convolution_2xM = GetQuantizedConvolutionNhwcGemm_2xM<term>(m);

            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                _norm[1] = _mm512_loadu_ps(norm + dc + F);
                const uint8_t* s = src;
                int32_t* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_2xN(s, p, a, srcC, dC, update, weight, _bias, _norm, _zero, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_2xM(s, p, a, srcC, dC, update, weight, _bias, _norm, _zero, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_2<Term8iInterim>;
            if (p.dstT == SimdTensorData8u)
                convolutions[1] = QuantizedConvolutionNhwcGemm_2<Term8iLast8u>;
            else
                convolutions[1] = NULL;// QuantizedConvolutionNhwcGemm_2<Term8iLast32f>;
        }

        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : Avx512bw::SynetQuantizedConvolutionNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            Set(p, _alg, _convolutions);
        }
    }
#endif
}
