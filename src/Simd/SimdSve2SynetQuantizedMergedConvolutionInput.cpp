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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SaveInput1(uint8_t* dst, const svint32_t& sum, const svint32_t& bias, const svfloat32_t& norm, const svint32_t& zero)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, NULL, sum, bias, bias, norm, norm, zero, svptrue_b32());
        }

        SIMD_INLINE void SaveInput2(uint8_t* dst0, uint8_t* dst1, const svint32_t& sum0, const svint32_t& sum1,
            const svint32_t& bias0, const svint32_t& bias1, const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst0, NULL, sum0, bias0, bias0, norm0, norm0, zero, svptrue_b32());
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst1, NULL, sum1, bias1, bias1, norm1, norm1, zero, svptrue_b32());
        }

        //------------------------------------------------------------------------------------------------

        template<int M> void QuantizedMergedConvolutionInput_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstC, const int8_t* weight0, const svint32_t& bias0, const svint32_t& bias1, const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, uint8_t* dst0, uint8_t* dst1)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1;
            svuint8_t s0;
            svint8_t w0, w1;
            size_t srcC = a.isB ? a.iwStep : p.srcC;
            const int8_t* weight1 = weight0 + a.iwStep * F;
            const uint8_t* src1 = src0 + 1 * srcC;
            const uint8_t* src2 = src0 + 2 * srcC;
            const uint8_t* src3 = src0 + 3 * srcC;
            const uint8_t* src4 = src0 + 4 * srcC;
            const uint8_t* src5 = src0 + 5 * srcC;
            if (dstC > F)
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
                for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC; offs0 < srcC; offs0 += 4, offs6 += 4)
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
                    weight0 += A, weight1 += A;
                }
                if (M > 0x0) SaveInput2(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, bias0, bias1, norm0, norm1, zero);
                if (M > 0x1) SaveInput2(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, bias0, bias1, norm0, norm1, zero);
                if (M > 0x2) SaveInput2(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, bias0, bias1, norm0, norm1, zero);
                if (M > 0x3) SaveInput2(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, bias0, bias1, norm0, norm1, zero);
                if (M > 0x4) SaveInput2(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, bias0, bias1, norm0, norm1, zero);
                if (M > 0x5) SaveInput2(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, bias0, bias1, norm0, norm1, zero);
                if (M > 0x6) SaveInput2(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, bias0, bias1, norm0, norm1, zero);
                if (M > 0x7) SaveInput2(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, bias0, bias1, norm0, norm1, zero);
                if (M > 0x8) SaveInput2(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, bias0, bias1, norm0, norm1, zero);
                if (M > 0x9) SaveInput2(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, bias0, bias1, norm0, norm1, zero);
                if (M > 0xA) SaveInput2(dst0 + 0xA * F, dst1 + 0xA * F, dA0, dA1, bias0, bias1, norm0, norm1, zero);
                if (M > 0xB) SaveInput2(dst0 + 0xB * F, dst1 + 0xB * F, dB0, dB1, bias0, bias1, norm0, norm1, zero);
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
                for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC; offs0 < srcC; offs0 += 4, offs6 += 4)
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
                if (M > 0x0) SaveInput1(dst0 + 0x0 * F, d00, bias0, norm0, zero);
                if (M > 0x1) SaveInput1(dst0 + 0x1 * F, d10, bias0, norm0, zero);
                if (M > 0x2) SaveInput1(dst0 + 0x2 * F, d20, bias0, norm0, zero);
                if (M > 0x3) SaveInput1(dst0 + 0x3 * F, d30, bias0, norm0, zero);
                if (M > 0x4) SaveInput1(dst0 + 0x4 * F, d40, bias0, norm0, zero);
                if (M > 0x5) SaveInput1(dst0 + 0x5 * F, d50, bias0, norm0, zero);
                if (M > 0x6) SaveInput1(dst0 + 0x6 * F, d60, bias0, norm0, zero);
                if (M > 0x7) SaveInput1(dst0 + 0x7 * F, d70, bias0, norm0, zero);
                if (M > 0x8) SaveInput1(dst0 + 0x8 * F, d80, bias0, norm0, zero);
                if (M > 0x9) SaveInput1(dst0 + 0x9 * F, d90, bias0, norm0, zero);
                if (M > 0xA) SaveInput1(dst0 + 0xA * F, dA0, bias0, norm0, zero);
                if (M > 0xB) SaveInput1(dst0 + 0xB * F, dB0, bias0, norm0, zero);
            }
        }

        typedef void(*QuantizedMergedConvolutionInput_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstC, const int8_t* weight0, const svint32_t& bias0, const svint32_t& bias1, const svfloat32_t& norm0, const svfloat32_t& norm1, const svint32_t& zero, uint8_t* dst0, uint8_t* dst1);

        QuantizedMergedConvolutionInput_2xM_Ptr GetQuantizedMergedConvolutionInput_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedMergedConvolutionInput_2xM<0x1>;
            case 0x2: return QuantizedMergedConvolutionInput_2xM<0x2>;
            case 0x3: return QuantizedMergedConvolutionInput_2xM<0x3>;
            case 0x4: return QuantizedMergedConvolutionInput_2xM<0x4>;
            case 0x5: return QuantizedMergedConvolutionInput_2xM<0x5>;
            case 0x6: return QuantizedMergedConvolutionInput_2xM<0x6>;
            case 0x7: return QuantizedMergedConvolutionInput_2xM<0x7>;
            case 0x8: return QuantizedMergedConvolutionInput_2xM<0x8>;
            case 0x9: return QuantizedMergedConvolutionInput_2xM<0x9>;
            case 0xA: return QuantizedMergedConvolutionInput_2xM<0xA>;
            case 0xB: return QuantizedMergedConvolutionInput_2xM<0xB>;
            case 0xC: return QuantizedMergedConvolutionInput_2xM<0xC>;
            }
            assert(0);
            return NULL;
        }

        void QuantizedMergedConvolutionInput_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* sum, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t dstM = a.dsH - 1, dstS = a.dsH * p.dstW * F, srcC = a.isB ? a.iwStep : p.srcC, y0 = a.isB ? yBeg : 0;
            svint32_t _zero = svdup_n_s32(zero);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.dsH)), n = 12;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
            QuantizedMergedConvolutionInput_2xM_Ptr quantizedMergedConvolutionInput_2xN = GetQuantizedMergedConvolutionInput_2xM(n);
            QuantizedMergedConvolutionInput_2xM_Ptr quantizedMergedConvolutionInput_2xI = GetQuantizedMergedConvolutionInput_2xM(i);
            QuantizedMergedConvolutionInput_2xM_Ptr quantizedMergedConvolutionInput_2xE = GetQuantizedMergedConvolutionInput_2xM(e);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                svint32_t _bias0 = svld1_s32(body, bias + dc + 0);
                svint32_t _bias1 = svld1_s32(body, bias + dc + F);
                svfloat32_t _norm0 = svld1_f32(body, norm + dc + 0);
                svfloat32_t _norm1 = svld1_f32(body, norm + dc + F);
                if (yInt > yBeg)
                {
                    const uint8_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                    uint8_t* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        quantizedMergedConvolutionInput_2xN(src0, p, a, dC, weight, _bias0, _bias1, _norm0, _norm1, _zero, dst0, dst1);
                    if (in < i1)
                        quantizedMergedConvolutionInput_2xI(src0, p, a, dC, weight, _bias0, _bias1, _norm0, _norm1, _zero, dst0, dst1);
                }
                if (yEnd > yInt)
                {
                    const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                    uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                        quantizedMergedConvolutionInput_2xN(src0, p, a, dC, weight, _bias0, _bias1, _norm0, _norm1, _zero, dst0, dst1);
                    if (en < e1)
                        quantizedMergedConvolutionInput_2xE(src0, p, a, dC, weight, _bias0, _bias1, _norm0, _norm1, _zero, dst0, dst1);
                }
                dst += a.dsH * p.dstW * DF;
                weight += a.iwStep * DF;
            }
        }

        //------------------------------------------------------------------------------------------------

        void SetInputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputConvolutionPtr& func)
        {
            func = QuantizedMergedConvolutionInput_2;
        }
    }
#endif
}
