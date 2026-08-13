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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynetQuantizedAddCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //------------------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedMergedConvolutionOutputConvolution_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const svint32_t& bias0, const svint32_t& bias1, const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), A = F * 4, DF = F * 2;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body = svptrue_b32();
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1;
            svuint8_t s0;
            svint8_t w0, w1;
            size_t dS = a.maC * p.strideX, dB = a.owStep, dD = p.dstC;
            const int8_t* weight1 = weight0 + AlignHi(srcC, 4) * F;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0x0) d00 = svld1_s32(body, buf + 0x0 * dB + 0), d01 = svld1_s32(body, buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = svld1_s32(body, buf + 0x1 * dB + 0), d11 = svld1_s32(body, buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = svld1_s32(body, buf + 0x2 * dB + 0), d21 = svld1_s32(body, buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = svld1_s32(body, buf + 0x3 * dB + 0), d31 = svld1_s32(body, buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = svld1_s32(body, buf + 0x4 * dB + 0), d41 = svld1_s32(body, buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = svld1_s32(body, buf + 0x5 * dB + 0), d51 = svld1_s32(body, buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = svld1_s32(body, buf + 0x6 * dB + 0), d61 = svld1_s32(body, buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = svld1_s32(body, buf + 0x7 * dB + 0), d71 = svld1_s32(body, buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = svld1_s32(body, buf + 0x8 * dB + 0), d81 = svld1_s32(body, buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = svld1_s32(body, buf + 0x9 * dB + 0), d91 = svld1_s32(body, buf + 0x9 * dB + F);
                    if (M > 0xA) dA0 = svld1_s32(body, buf + 0xA * dB + 0), dA1 = svld1_s32(body, buf + 0xA * dB + F);
                    if (M > 0xB) dB0 = svld1_s32(body, buf + 0xB * dB + 0), dB1 = svld1_s32(body, buf + 0xB * dB + F);
                }
                else
                {
                    if (M > 0x0) d00 = svdup_n_s32(0), d01 = svdup_n_s32(0);
                    if (M > 0x1) d10 = svdup_n_s32(0), d11 = svdup_n_s32(0);
                    if (M > 0x2) d20 = svdup_n_s32(0), d21 = svdup_n_s32(0);
                    if (M > 0x3) d30 = svdup_n_s32(0), d31 = svdup_n_s32(0);
                    if (M > 0x4) d40 = svdup_n_s32(0), d41 = svdup_n_s32(0);
                    if (M > 0x5) d50 = svdup_n_s32(0), d51 = svdup_n_s32(0);
                    if (M > 0x6) d60 = svdup_n_s32(0), d61 = svdup_n_s32(0);
                    if (M > 0x7) d70 = svdup_n_s32(0), d71 = svdup_n_s32(0);
                    if (M > 0x8) d80 = svdup_n_s32(0), d81 = svdup_n_s32(0);
                    if (M > 0x9) d90 = svdup_n_s32(0), d91 = svdup_n_s32(0);
                    if (M > 0xA) dA0 = svdup_n_s32(0), dA1 = svdup_n_s32(0);
                    if (M > 0xB) dB0 = svdup_n_s32(0), dB1 = svdup_n_s32(0);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = svld1_s8(body8, weight0);
                    w1 = svld1_s8(body8, weight1);
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
                    weight0 += A;
                    weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0x0) Save2<term>(dst, buf, d00, d01, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x1) Save2<term>(dst, buf, d10, d11, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x2) Save2<term>(dst, buf, d20, d21, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x3) Save2<term>(dst, buf, d30, d31, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x4) Save2<term>(dst, buf, d40, d41, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x5) Save2<term>(dst, buf, d50, d51, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x6) Save2<term>(dst, buf, d60, d61, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x7) Save2<term>(dst, buf, d70, d71, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x8) Save2<term>(dst, buf, d80, d81, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0x9) Save2<term>(dst, buf, d90, d91, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0xA) Save2<term>(dst, buf, dA0, dA1, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                    if (M > 0xB) Save2<term>(dst, buf, dB0, dB1, bias0, bias1, norm0, norm1, zero), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0x0) Save2<term>(dst, buf, d00, d01, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x1) Save2<term>(dst, buf, d10, d11, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x2) Save2<term>(dst, buf, d20, d21, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x3) Save2<term>(dst, buf, d30, d31, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x4) Save2<term>(dst, buf, d40, d41, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x5) Save2<term>(dst, buf, d50, d51, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x6) Save2<term>(dst, buf, d60, d61, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x7) Save2<term>(dst, buf, d70, d71, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x8) Save2<term>(dst, buf, d80, d81, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0x9) Save2<term>(dst, buf, d90, d91, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0xA) Save2<term>(dst, buf, dA0, dA1, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 0xB) Save2<term>(dst, buf, dB0, dB1, bias0, bias1, norm0, norm1, zero, dstC - F), buf += dB, dst += dD;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = svld1_s32(body, buf + 0x0 * dB);
                    if (M > 0x1) d10 = svld1_s32(body, buf + 0x1 * dB);
                    if (M > 0x2) d20 = svld1_s32(body, buf + 0x2 * dB);
                    if (M > 0x3) d30 = svld1_s32(body, buf + 0x3 * dB);
                    if (M > 0x4) d40 = svld1_s32(body, buf + 0x4 * dB);
                    if (M > 0x5) d50 = svld1_s32(body, buf + 0x5 * dB);
                    if (M > 0x6) d60 = svld1_s32(body, buf + 0x6 * dB);
                    if (M > 0x7) d70 = svld1_s32(body, buf + 0x7 * dB);
                    if (M > 0x8) d80 = svld1_s32(body, buf + 0x8 * dB);
                    if (M > 0x9) d90 = svld1_s32(body, buf + 0x9 * dB);
                    if (M > 0xA) dA0 = svld1_s32(body, buf + 0xA * dB);
                    if (M > 0xB) dB0 = svld1_s32(body, buf + 0xB * dB);
                }
                else
                {
                    if (M > 0x0) d00 = svdup_n_s32(0);
                    if (M > 0x1) d10 = svdup_n_s32(0);
                    if (M > 0x2) d20 = svdup_n_s32(0);
                    if (M > 0x3) d30 = svdup_n_s32(0);
                    if (M > 0x4) d40 = svdup_n_s32(0);
                    if (M > 0x5) d50 = svdup_n_s32(0);
                    if (M > 0x6) d60 = svdup_n_s32(0);
                    if (M > 0x7) d70 = svdup_n_s32(0);
                    if (M > 0x8) d80 = svdup_n_s32(0);
                    if (M > 0x9) d90 = svdup_n_s32(0);
                    if (M > 0xA) dA0 = svdup_n_s32(0);
                    if (M > 0xB) dB0 = svdup_n_s32(0);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = svld1_s8(body8, weight0);
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
                if (dstC == F)
                {
                    if (M > 0x0) Save1<term>(dst, buf, d00, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x1) Save1<term>(dst, buf, d10, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x2) Save1<term>(dst, buf, d20, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x3) Save1<term>(dst, buf, d30, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x4) Save1<term>(dst, buf, d40, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x5) Save1<term>(dst, buf, d50, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x6) Save1<term>(dst, buf, d60, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x7) Save1<term>(dst, buf, d70, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x8) Save1<term>(dst, buf, d80, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0x9) Save1<term>(dst, buf, d90, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0xA) Save1<term>(dst, buf, dA0, bias0, norm0, zero), buf += dB, dst += dD;
                    if (M > 0xB) Save1<term>(dst, buf, dB0, bias0, norm0, zero), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0x0) Save1<term>(dst, buf, d00, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x1) Save1<term>(dst, buf, d10, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x2) Save1<term>(dst, buf, d20, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x3) Save1<term>(dst, buf, d30, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x4) Save1<term>(dst, buf, d40, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x5) Save1<term>(dst, buf, d50, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x6) Save1<term>(dst, buf, d60, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x7) Save1<term>(dst, buf, d70, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x8) Save1<term>(dst, buf, d80, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0x9) Save1<term>(dst, buf, d90, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0xA) Save1<term>(dst, buf, dA0, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                    if (M > 0xB) Save1<term>(dst, buf, dB0, bias0, norm0, zero, dstC), buf += dB, dst += dD;
                }
            }
        }

        typedef void(*QuantizedMergedConvolutionOutputConvolution_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const svint32_t& bias0, const svint32_t& bias1, const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> QuantizedMergedConvolutionOutputConvolution_2xM_Ptr GetQuantizedMergedConvolutionOutputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x1>;
            case 0x2: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x2>;
            case 0x3: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x3>;
            case 0x4: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x4>;
            case 0x5: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x5>;
            case 0x6: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x6>;
            case 0x7: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x7>;
            case 0x8: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x8>;
            case 0x9: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0x9>;
            case 0xA: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0xA>;
            case 0xB: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0xB>;
            case 0xC: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedMergedConvolutionOutputConvolution_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n = 12, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            QuantizedMergedConvolutionOutputConvolution_2xM_Ptr outputConvolution1x1_2xN = GetQuantizedMergedConvolutionOutputConvolution_2xM<term>(n);
            QuantizedMergedConvolutionOutputConvolution_2xM_Ptr outputConvolution1x1_2xM = GetQuantizedMergedConvolutionOutputConvolution_2xM<term>(m);
            svint32_t _zero = svdup_n_s32(zero);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                svint32_t _bias0 = svld1_s32(body, bias + dc + 0);
                svint32_t _bias1 = svld1_s32(body, bias + dc + F);
                svfloat32_t _norm0 = svld1_f32(body, norm + dc + 0);
                svfloat32_t _norm1 = svld1_f32(body, norm + dc + F);
                const uint8_t* s = src;
                int32_t* b = buf + dc + yBeg * p.dstW * a.owStep;
                uint8_t* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += a.owStep * n, d += p.dstC * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, update, weight, _bias0, _bias1, _norm0, _norm1, _zero, b, d);
                for (; i < n1; i += m, s += a.maC * m, b += a.owStep * m, d += p.dstC * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, update, weight, _bias0, _bias1, _norm0, _norm1, _zero, b, d);
                weight += AlignHi(maC, 4) * DF;
            }
        }

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t Load8u(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE void Store8u(const svint32_t& value, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t lo = svmax_n_s32_x(mask, value, 0);
            svuint32_t u32 = svreinterpret_u32_s32(svmin_n_s32_x(mask, lo, 255));
            svst1b_u32(mask, dst, u32);
        }

        SIMD_INLINE void QuantizedAdd8u8u8u(const uint8_t* a, const svfloat32_t& aNorm, const uint8_t* b, const svfloat32_t& bNorm, const svfloat32_t& term, uint8_t* dst, const svbool_t& mask)
        {
            svfloat32_t value = svmla_f32_x(mask, term, svcvt_f32_s32_x(mask, Load8u(a, mask)), aNorm);
            value = svmla_f32_x(mask, value, svcvt_f32_s32_x(mask, Load8u(b, mask)), bNorm);
            Store8u(NearbyInt(value, mask), dst, mask);
        }

        void QuantizedMergedConvolutionAddInputToOutput(const uint8_t* a, float aNorm, const uint8_t* b, float bNorm, const ConvParam& p, size_t yBeg, size_t yEnd, float dBias, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            svfloat32_t _aNorm = svdup_n_f32(aNorm), _bNorm = svdup_n_f32(bNorm), _dBias = svdup_n_f32(dBias);
            size_t beg = yBeg * p.dstW * p.dstC, end = yEnd * p.dstW * p.dstC;
            size_t i = beg, endQF = beg + AlignLo(end - beg, QF);
            for (; i < endQF; i += QF)
            {
                QuantizedAdd8u8u8u(a + i + 0 * F, _aNorm, b + i + 0 * F, _bNorm, _dBias, dst + i + 0 * F, full);
                QuantizedAdd8u8u8u(a + i + 1 * F, _aNorm, b + i + 1 * F, _bNorm, _dBias, dst + i + 1 * F, full);
                QuantizedAdd8u8u8u(a + i + 2 * F, _aNorm, b + i + 2 * F, _bNorm, _dBias, dst + i + 2 * F, full);
                QuantizedAdd8u8u8u(a + i + 3 * F, _aNorm, b + i + 3 * F, _bNorm, _dBias, dst + i + 3 * F, full);
            }
            for (; i < end; i += F)
                QuantizedAdd8u8u8u(a + i, _aNorm, b + i, _bNorm, _dBias, dst + i, svwhilelt_b32(i, end));
        }

        //------------------------------------------------------------------------------------------------

        void SetOutputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::OutputConvolutionPtr* funcs)
        {
            funcs[0] = QuantizedMergedConvolutionOutputConvolution_2<Term8iInterim>;
            funcs[1] = QuantizedMergedConvolutionOutputConvolution_2<Term8iLast8u>;
        }

        void SetAddInputToOutput(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::AddInputToOutputPtr& func)
        {
            func = QuantizedMergedConvolutionAddInputToOutput;
        }
    }
#endif
}
