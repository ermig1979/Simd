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
        typedef Base::SynetQuantizedConvolutionNhwcGemmV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemmV0::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmV0_Reorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint8_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    memcpy(row, ps, p.srcC);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, zero, p.srcC);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, zero, p.kernelX * p.srcC);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void QuantizedConvolutionNhwcGemmV0_i2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC,
            int update, const int8_t* weight0, const svint32_t& sBias0, const svint32_t& sBias1, const svfloat32_t& sNorm0, const svfloat32_t& sNorm1,
            const svint32_t& iLo, const svint32_t& iHi, const svfloat32_t& iScale, const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& dNorm, const svint32_t& dZero, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), A = F * 4, DF = F * 2;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1;
            svuint8_t s0;
            svint8_t w0, w1;
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
                    if (M > 0x0) d00 = svld1_s32(body32, buf + 0x0 * dB + 0), d01 = svld1_s32(body32, buf + 0x0 * dB + F);
                    if (M > 0x1) d10 = svld1_s32(body32, buf + 0x1 * dB + 0), d11 = svld1_s32(body32, buf + 0x1 * dB + F);
                    if (M > 0x2) d20 = svld1_s32(body32, buf + 0x2 * dB + 0), d21 = svld1_s32(body32, buf + 0x2 * dB + F);
                    if (M > 0x3) d30 = svld1_s32(body32, buf + 0x3 * dB + 0), d31 = svld1_s32(body32, buf + 0x3 * dB + F);
                    if (M > 0x4) d40 = svld1_s32(body32, buf + 0x4 * dB + 0), d41 = svld1_s32(body32, buf + 0x4 * dB + F);
                    if (M > 0x5) d50 = svld1_s32(body32, buf + 0x5 * dB + 0), d51 = svld1_s32(body32, buf + 0x5 * dB + F);
                    if (M > 0x6) d60 = svld1_s32(body32, buf + 0x6 * dB + 0), d61 = svld1_s32(body32, buf + 0x6 * dB + F);
                    if (M > 0x7) d70 = svld1_s32(body32, buf + 0x7 * dB + 0), d71 = svld1_s32(body32, buf + 0x7 * dB + F);
                    if (M > 0x8) d80 = svld1_s32(body32, buf + 0x8 * dB + 0), d81 = svld1_s32(body32, buf + 0x8 * dB + F);
                    if (M > 0x9) d90 = svld1_s32(body32, buf + 0x9 * dB + 0), d91 = svld1_s32(body32, buf + 0x9 * dB + F);
                    if (M > 0xA) dA0 = svld1_s32(body32, buf + 0xA * dB + 0), dA1 = svld1_s32(body32, buf + 0xA * dB + F);
                    if (M > 0xB) dB0 = svld1_s32(body32, buf + 0xB * dB + 0), dB1 = svld1_s32(body32, buf + 0xB * dB + F);
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
                    weight0 += A, weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    dstC -= F;
                    if (M > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias0, sBias1, sNorm0, sNorm1, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = svld1_s32(body32, buf + 0x0 * dB);
                    if (M > 0x1) d10 = svld1_s32(body32, buf + 0x1 * dB);
                    if (M > 0x2) d20 = svld1_s32(body32, buf + 0x2 * dB);
                    if (M > 0x3) d30 = svld1_s32(body32, buf + 0x3 * dB);
                    if (M > 0x4) d40 = svld1_s32(body32, buf + 0x4 * dB);
                    if (M > 0x5) d50 = svld1_s32(body32, buf + 0x5 * dB);
                    if (M > 0x6) d60 = svld1_s32(body32, buf + 0x6 * dB);
                    if (M > 0x7) d70 = svld1_s32(body32, buf + 0x7 * dB);
                    if (M > 0x8) d80 = svld1_s32(body32, buf + 0x8 * dB);
                    if (M > 0x9) d90 = svld1_s32(body32, buf + 0x9 * dB);
                    if (M > 0xA) dA0 = svld1_s32(body32, buf + 0xA * dB);
                    if (M > 0xB) dB0 = svld1_s32(body32, buf + 0xB * dB);
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
                    if (M > 0x0) Save1<term, type>(dst, buf, d00, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x1) Save1<term, type>(dst, buf, d10, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x2) Save1<term, type>(dst, buf, d20, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x3) Save1<term, type>(dst, buf, d30, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x4) Save1<term, type>(dst, buf, d40, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x5) Save1<term, type>(dst, buf, d50, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x6) Save1<term, type>(dst, buf, d60, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x7) Save1<term, type>(dst, buf, d70, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x8) Save1<term, type>(dst, buf, d80, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0x9) Save1<term, type>(dst, buf, d90, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0xA) Save1<term, type>(dst, buf, dA0, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 0xB) Save1<term, type>(dst, buf, dB0, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0x0) Save1<term, type>(dst, buf, d00, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x1) Save1<term, type>(dst, buf, d10, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x2) Save1<term, type>(dst, buf, d20, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x3) Save1<term, type>(dst, buf, d30, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x4) Save1<term, type>(dst, buf, d40, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x5) Save1<term, type>(dst, buf, d50, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x6) Save1<term, type>(dst, buf, d60, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x7) Save1<term, type>(dst, buf, d70, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x8) Save1<term, type>(dst, buf, d80, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0x9) Save1<term, type>(dst, buf, d90, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0xA) Save1<term, type>(dst, buf, dA0, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 0xB) Save1<term, type>(dst, buf, dB0, sBias0, sNorm0, iLo, iHi, iScale, param0, param1, dNorm, dZero, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*QuantizedConvolutionNhwcGemmV0_i2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, int update, const int8_t* weight,
            const svint32_t& sBias0, const svint32_t& sBias1, const svfloat32_t& sNorm0, const svfloat32_t& sNorm1,
            const svint32_t& iLo, const svint32_t& iHi, const svfloat32_t& iScale, const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& dNorm, const svint32_t& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> QuantizedConvolutionNhwcGemmV0_i2xM_Ptr GetQuantizedConvolutionNhwcGemmV0_i2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x1>;
            case 0x2: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x2>;
            case 0x3: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x3>;
            case 0x4: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x4>;
            case 0x5: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x5>;
            case 0x6: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x6>;
            case 0x7: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x7>;
            case 0x8: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x8>;
            case 0x9: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0x9>;
            case 0xA: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0xA>;
            case 0xB: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0xB>;
            case 0xC: return QuantizedConvolutionNhwcGemmV0_i2xM<term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcGemmV0_i2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int update, const int8_t* weight,
            const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n1 = dstH * p.dstW, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            QuantizedConvolutionNhwcGemmV0_i2xM_Ptr convolution_i2xN = GetQuantizedConvolutionNhwcGemmV0_i2xM<term, type>(n);
            QuantizedConvolutionNhwcGemmV0_i2xM_Ptr convolution_i2xM = GetQuantizedConvolutionNhwcGemmV0_i2xM<term, type>(m);

            svfloat32_t _sNorm0, _sNorm1, _iScale, _param0, _param1, _dNorm;
            svint32_t _sBias0, _sBias1, _dZero = svdup_n_s32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = svdup_n_s32(-iZero);
                _iHi = svdup_n_s32(255 - iZero);
                _iScale = svdup_n_f32(iScale);
                _dNorm = svdup_n_f32(dNorm);
                _param0 = svdup_n_f32(params[0]);
                _param1 = svdup_n_f32(params[1]);
            }
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _sBias0 = svld1_s32(body, sBias + dc + 0);
                _sBias1 = svld1_s32(body, sBias + dc + F);
                _sNorm0 = svld1_f32(body, sNorm + dc + 0);
                _sNorm1 = svld1_f32(body, sNorm + dc + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _param0 = svld1_f32(body, params + dc + 0);
                    _param1 = svld1_f32(body, params + dc + F);
                }
                const uint8_t* s = src;
                int32_t* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_i2xN(s, p, a, srcC, dC, update, weight, _sBias0, _sBias1, _sNorm0, _sNorm1, _iLo, _iHi, _iScale, _param0, _param1, _dNorm, _dZero, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_i2xM(s, p, a, srcC, dC, update, weight, _sBias0, _sBias1, _sNorm0, _sNorm1, _iLo, _iHi, _iScale, _param0, _param1, _dNorm, _dZero, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemmV0_i2<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: convolutions[1] = QuantizedConvolutionNhwcGemmV0_i2<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                convolutions[1] = NULL;
            }
        }

        SynetQuantizedConvolutionNhwcGemmV0::SynetQuantizedConvolutionNhwcGemmV0(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcGemmV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, F * 2, 12, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src8u)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = QuantizedConvolutionNhwcGemmV0_Reorder;
            }
            else
                assert(0);
            Set(p, _alg, _convolutions);
        }
    }
#endif
}
