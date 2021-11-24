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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx512bw
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (first)
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
                else
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
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight0);
                        w1 = _mm512_loadu_si512((__m512i*)weight1);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0), Madd4<true>(d51, s0, w1);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0), Madd4<true>(d61, s0, w1);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0), Madd4<true>(d71, s0, w1);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0), Madd4<true>(d81, s0, w1);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0), Madd4<true>(d91, s0, w1);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s0, w1);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s0, w1);
                        weight0 += A, weight1 += A;
                    }
                }
                else
                {
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
                }
                __mmask16 tail = TailMask16(dstC - F);
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) Save2<term, type, true>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x1) Save2<term, type, true>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x2) Save2<term, type, true>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x3) Save2<term, type, true>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x4) Save2<term, type, true>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x5) Save2<term, type, true>(dst, buf, d50, d51, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x6) Save2<term, type, true>(dst, buf, d60, d61, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x7) Save2<term, type, true>(dst, buf, d70, d71, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x8) Save2<term, type, true>(dst, buf, d80, d81, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x9) Save2<term, type, true>(dst, buf, d90, d91, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xA) Save2<term, type, true>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xB) Save2<term, type, true>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0x0) Save2<term, type, false>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x1) Save2<term, type, false>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x2) Save2<term, type, false>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x3) Save2<term, type, false>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x4) Save2<term, type, false>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x5) Save2<term, type, false>(dst, buf, d50, d51, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x6) Save2<term, type, false>(dst, buf, d60, d61, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x7) Save2<term, type, false>(dst, buf, d70, d71, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x8) Save2<term, type, false>(dst, buf, d80, d81, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x9) Save2<term, type, false>(dst, buf, d90, d91, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xA) Save2<term, type, false>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xB) Save2<term, type, false>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                }
            }
            else
            {
                if (first)
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
                else
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
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight0);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0);
                        weight0 += A;
                    }
                }
                else
                {
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
                }
                __mmask16 tail = TailMask16(dstC);
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) Save1<term, type, true>(dst, buf, d00, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x1) Save1<term, type, true>(dst, buf, d10, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x2) Save1<term, type, true>(dst, buf, d20, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x3) Save1<term, type, true>(dst, buf, d30, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x4) Save1<term, type, true>(dst, buf, d40, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x5) Save1<term, type, true>(dst, buf, d50, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x6) Save1<term, type, true>(dst, buf, d60, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x7) Save1<term, type, true>(dst, buf, d70, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x8) Save1<term, type, true>(dst, buf, d80, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x9) Save1<term, type, true>(dst, buf, d90, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xA) Save1<term, type, true>(dst, buf, dA0, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xB) Save1<term, type, true>(dst, buf, dB0, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0x0) Save1<term, type, false>(dst, buf, d00, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x1) Save1<term, type, false>(dst, buf, d10, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x2) Save1<term, type, false>(dst, buf, d20, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x3) Save1<term, type, false>(dst, buf, d30, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x4) Save1<term, type, false>(dst, buf, d40, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x5) Save1<term, type, false>(dst, buf, d50, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x6) Save1<term, type, false>(dst, buf, d60, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x7) Save1<term, type, false>(dst, buf, d70, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x8) Save1<term, type, false>(dst, buf, d80, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0x9) Save1<term, type, false>(dst, buf, d90, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xA) Save1<term, type, false>(dst, buf, dA0, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                    if (M > 0xB) Save1<term, type, false>(dst, buf, dB0, norm, bias, params, scale, shift, upper, tail), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first);

        template<Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect1x1_2xM< term, type, 0x9>;
            case 0xA: return ConvolutionNhwcDirect1x1_2xM< term, type, 0xA>;
            case 0xB: return ConvolutionNhwcDirect1x1_2xM< term, type, 0xB>;
            case 0xC: return ConvolutionNhwcDirect1x1_2xM< term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t n = 12, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<term, type>(n);
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);
            __m512 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                _norm[1] = _mm512_loadu_ps(norm + dc + F);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                _scale[0] = _mm512_loadu_ps(scale + dc + 0);
                _scale[1] = _mm512_loadu_ps(scale + dc + F);
                _shift[0] = _mm512_loadu_ps(shift + dc + 0);
                _shift[1] = _mm512_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    convolutionNhwcDirect1x1_2xN(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    convolutionNhwcDirect1x1_2xM(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                weight += DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            assert(a.microD == 2 * F && p.Is1x1() == true);
            d[term] = ConvolutionNhwcDirect1x1_2<term, activation>;
        }

        template<SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            SetDirect1x1<Term8iLast8u, activation>(p, a, d);
            SetDirect1x1<Term8iLast32f, activation>(p, a, d);
            SetDirect1x1<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: SetDirect1x1<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: SetDirect1x1<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: SetDirect1x1<SimdConvolutionActivationMish>(p, a, d); break;
            case SimdConvolutionActivationHardSigmoid: SetDirect1x1<SimdConvolutionActivationHardSigmoid>(p, a, d); break;
            case SimdConvolutionActivationSwish: SetDirect1x1<SimdConvolutionActivationSwish>(p, a, d); break;
            default: assert(0);
            }
        }
    }
#endif
}
