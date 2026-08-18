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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizedDepthwise.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::ConvolutionPtr Convolution;

        SIMD_INLINE void Copy(const uint8_t* src, size_t size, uint8_t* dst)
        {
            const size_t A = svcntb();
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = src[i];
            for (; i < A; ++i)
                dst[i] = 0;
        }

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcSpecV0Reorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            const size_t A = svcntb();
            assert(a.microC == A);
            const svbool_t body = svptrue_b8();
            svuint8_t _zero = svdup_n_u8(zero);
            size_t srcCA = Simd::AlignLo(p.srcC, A), tailC = p.srcC - srcCA;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        svst1_u8(body, dst + c * cD + s * sD, _zero);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            svst1_u8(body, dst + c * cD + s * sD, _zero);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCA; sc += A)
                        Sve2::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Copy(src + sc, tailC, dst + sc * cD);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        svst1_u8(body, dst + c * cD + s * sD, _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        svst1_u8(body, dst + c * cD + s * sD, _zero);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int M> void QuantizedConvolutionNhwcSpecV0_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int update, const int8_t* weight0, int32_t* dst)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1;
            svuint8_t s0;
            svint8_t w0, w1;
            size_t dD = a.macroD, dX = a.microC;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 1 * dX;
            const uint8_t* src2 = src0 + 2 * dX;
            const uint8_t* src3 = src0 + 3 * dX;
            const uint8_t* src4 = src0 + 4 * dX;
            const uint8_t* src5 = src0 + 5 * dX;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0x0) d00 = svld1_s32(body32, dst + 0x0 * dD + 0), d01 = svld1_s32(body32, dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = svld1_s32(body32, dst + 0x1 * dD + 0), d11 = svld1_s32(body32, dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = svld1_s32(body32, dst + 0x2 * dD + 0), d21 = svld1_s32(body32, dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = svld1_s32(body32, dst + 0x3 * dD + 0), d31 = svld1_s32(body32, dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = svld1_s32(body32, dst + 0x4 * dD + 0), d41 = svld1_s32(body32, dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = svld1_s32(body32, dst + 0x5 * dD + 0), d51 = svld1_s32(body32, dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = svld1_s32(body32, dst + 0x6 * dD + 0), d61 = svld1_s32(body32, dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = svld1_s32(body32, dst + 0x7 * dD + 0), d71 = svld1_s32(body32, dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = svld1_s32(body32, dst + 0x8 * dD + 0), d81 = svld1_s32(body32, dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = svld1_s32(body32, dst + 0x9 * dD + 0), d91 = svld1_s32(body32, dst + 0x9 * dD + F);
                    if (M > 0xA) dA0 = svld1_s32(body32, dst + 0xA * dD + 0), dA1 = svld1_s32(body32, dst + 0xA * dD + F);
                    if (M > 0xB) dB0 = svld1_s32(body32, dst + 0xB * dD + 0), dB1 = svld1_s32(body32, dst + 0xB * dD + F);
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
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + dX * 6; offs0 < end; offs0 += 4, offs6 += 4)
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
                }
                if (M > 0x0) svst1_s32(body32, dst + 0x0 * dD + 0, d00), svst1_s32(body32, dst + 0x0 * dD + F, d01);
                if (M > 0x1) svst1_s32(body32, dst + 0x1 * dD + 0, d10), svst1_s32(body32, dst + 0x1 * dD + F, d11);
                if (M > 0x2) svst1_s32(body32, dst + 0x2 * dD + 0, d20), svst1_s32(body32, dst + 0x2 * dD + F, d21);
                if (M > 0x3) svst1_s32(body32, dst + 0x3 * dD + 0, d30), svst1_s32(body32, dst + 0x3 * dD + F, d31);
                if (M > 0x4) svst1_s32(body32, dst + 0x4 * dD + 0, d40), svst1_s32(body32, dst + 0x4 * dD + F, d41);
                if (M > 0x5) svst1_s32(body32, dst + 0x5 * dD + 0, d50), svst1_s32(body32, dst + 0x5 * dD + F, d51);
                if (M > 0x6) svst1_s32(body32, dst + 0x6 * dD + 0, d60), svst1_s32(body32, dst + 0x6 * dD + F, d61);
                if (M > 0x7) svst1_s32(body32, dst + 0x7 * dD + 0, d70), svst1_s32(body32, dst + 0x7 * dD + F, d71);
                if (M > 0x8) svst1_s32(body32, dst + 0x8 * dD + 0, d80), svst1_s32(body32, dst + 0x8 * dD + F, d81);
                if (M > 0x9) svst1_s32(body32, dst + 0x9 * dD + 0, d90), svst1_s32(body32, dst + 0x9 * dD + F, d91);
                if (M > 0xA) svst1_s32(body32, dst + 0xA * dD + 0, dA0), svst1_s32(body32, dst + 0xA * dD + F, dA1);
                if (M > 0xB) svst1_s32(body32, dst + 0xB * dD + 0, dB0), svst1_s32(body32, dst + 0xB * dD + F, dB1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = svld1_s32(body32, dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = svld1_s32(body32, dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = svld1_s32(body32, dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = svld1_s32(body32, dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = svld1_s32(body32, dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = svld1_s32(body32, dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = svld1_s32(body32, dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = svld1_s32(body32, dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = svld1_s32(body32, dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = svld1_s32(body32, dst + 0x9 * dD + 0);
                    if (M > 0xA) dA0 = svld1_s32(body32, dst + 0xA * dD + 0);
                    if (M > 0xB) dB0 = svld1_s32(body32, dst + 0xB * dD + 0);
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
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + dX * 6; offs0 < end; offs0 += 4, offs6 += 4)
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
                }
                if (M > 0x0) svst1_s32(body32, dst + 0x0 * dD + 0, d00);
                if (M > 0x1) svst1_s32(body32, dst + 0x1 * dD + 0, d10);
                if (M > 0x2) svst1_s32(body32, dst + 0x2 * dD + 0, d20);
                if (M > 0x3) svst1_s32(body32, dst + 0x3 * dD + 0, d30);
                if (M > 0x4) svst1_s32(body32, dst + 0x4 * dD + 0, d40);
                if (M > 0x5) svst1_s32(body32, dst + 0x5 * dD + 0, d50);
                if (M > 0x6) svst1_s32(body32, dst + 0x6 * dD + 0, d60);
                if (M > 0x7) svst1_s32(body32, dst + 0x7 * dD + 0, d70);
                if (M > 0x8) svst1_s32(body32, dst + 0x8 * dD + 0, d80);
                if (M > 0x9) svst1_s32(body32, dst + 0x9 * dD + 0, d90);
                if (M > 0xA) svst1_s32(body32, dst + 0xA * dD + 0, dA0);
                if (M > 0xB) svst1_s32(body32, dst + 0xB * dD + 0, dB0);
            }
        }

        typedef void(*QuantizedConvolutionNhwcSpecV0_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, size_t dstC, int update, const int8_t* weight0, int32_t* dst);

        static QuantizedConvolutionNhwcSpecV0_2xM_Ptr GetQuantizedConvolutionNhwcSpecV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedConvolutionNhwcSpecV0_2xM<0x1>;
            case 0x2: return QuantizedConvolutionNhwcSpecV0_2xM<0x2>;
            case 0x3: return QuantizedConvolutionNhwcSpecV0_2xM<0x3>;
            case 0x4: return QuantizedConvolutionNhwcSpecV0_2xM<0x4>;
            case 0x5: return QuantizedConvolutionNhwcSpecV0_2xM<0x5>;
            case 0x6: return QuantizedConvolutionNhwcSpecV0_2xM<0x6>;
            case 0x7: return QuantizedConvolutionNhwcSpecV0_2xM<0x7>;
            case 0x8: return QuantizedConvolutionNhwcSpecV0_2xM<0x8>;
            case 0x9: return QuantizedConvolutionNhwcSpecV0_2xM<0x9>;
            case 0xA: return QuantizedConvolutionNhwcSpecV0_2xM<0xA>;
            case 0xB: return QuantizedConvolutionNhwcSpecV0_2xM<0xB>;
            case 0xC: return QuantizedConvolutionNhwcSpecV0_2xM<0xC>;
            }
            assert(0);
            return NULL;
        }

        static void QuantizedConvolutionNhwcSpecV0_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int update, const int8_t* weight, int32_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            size_t n1 = dstH * a.srcW - a.gapH, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            QuantizedConvolutionNhwcSpecV0_2xM_Ptr convolution_2xN = GetQuantizedConvolutionNhwcSpecV0_2xM(n);
            QuantizedConvolutionNhwcSpecV0_2xM_Ptr convolution_2xM = GetQuantizedConvolutionNhwcSpecV0_2xM(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                for (; i < nn; i += n)
                    convolution_2xN(src + i * dS, p, a, offs, nK, dC, update, weight, dst + i * dD);
                for (; i < n1; i += m)
                    convolution_2xM(src + i * dS, p, a, offs, nK, dC, update, weight, dst + i * dD);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void SynetQuantizedConvolutionNhwcSpecV0Postprocess(const int32_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dyBeg, size_t dyEnd, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t dstCF = AlignLo(dstC, F), tailD = dstC - dstCF;
            size_t rowGap = a.gapH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            svfloat32_t _sNorm, _iScale, _param0, _param1, _dNorm;
            svint32_t _src, _dZero = svdup_n_s32(dZero), _sBias, _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = svdup_n_s32(-iZero);
                _iHi = svdup_n_s32(255 - iZero);
                _iScale = svdup_n_f32(iScale);
                _dNorm = svdup_n_f32(dNorm);
                _param0 = svdup_n_f32(params[0]);
                _param1 = svdup_n_f32(params[1]);
            }
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                    {
                        _src = svld1_s32(body, src + dc);
                        _sBias = svld1_s32(body, sBias + dc);
                        _sNorm = svld1_f32(body, sNorm + dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _param0 = svld1_f32(body, params + dc);
                        Save1<Term8iLast8u, type>(dst + dc, _src, _sBias, _sNorm, _iLo, _iHi, _iScale, _param0, _param1, _dNorm, _dZero);
                    }
                    if (tailD)
                    {
                        svbool_t tail = svwhilelt_b32((size_t)0, tailD);
                        _src = svld1_s32(tail, src + dc);
                        _sBias = svld1_s32(tail, sBias + dc);
                        _sNorm = svld1_f32(tail, sNorm + dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _param0 = svld1_f32(tail, params + dc);
                        Save1<Term8iLast8u, type>(dst + dc, _src, _sBias, _sNorm, _iLo, _iHi, _iScale, _param0, _param1, _dNorm, _dZero, tailD);
                    }
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcSpecV0::SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcSpecV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, F * 2, 12, F * 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src8u)
            {
                _preprocess = QuantizedConvolutionNhwcSpecV0Reorder;
            }
            else
                assert(0);
            _convolution = QuantizedConvolutionNhwcSpecV0_2;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationGelu>; break;
            default:
                _postprocess = NULL;
                assert(0);
            }
        }
    }
#endif
}
