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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
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
        typedef Base::SynetQuantizedConvolutionNchwGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNchwGemm::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int N> void SynetQuantizedConvolutionNchwGemm_Gemm2xN(const int8_t* weight0, const ConvParam& p,
            const AlgParam& a, size_t K, size_t M, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const svint32_t& iLo, const svint32_t& iHi,
            const svfloat32_t& iScale, const float* params, const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& dNorm, const svint32_t& dZero, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), A = F * 4, DF = F * 2;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1;
            svuint8_t s0, s1;
            svint8_t w0;
            size_t dB = a.bufN, dD = a.N * a.elem;
            const uint8_t* src1 = src0 + K * F;
            const int8_t* weight1 = weight0 + 1 * K;
            const int8_t* weight2 = weight0 + 2 * K;
            const int8_t* weight3 = weight0 + 3 * K;
            const int8_t* weight4 = weight0 + 4 * K;
            const int8_t* weight5 = weight0 + 5 * K;
            if (M > F)
            {
                if (update)
                {
                    if (N > 0x0) d00 = svld1_s32(body32, buf + 0x0 * dB + 0), d01 = svld1_s32(body32, buf + 0x0 * dB + F);
                    if (N > 0x1) d10 = svld1_s32(body32, buf + 0x1 * dB + 0), d11 = svld1_s32(body32, buf + 0x1 * dB + F);
                    if (N > 0x2) d20 = svld1_s32(body32, buf + 0x2 * dB + 0), d21 = svld1_s32(body32, buf + 0x2 * dB + F);
                    if (N > 0x3) d30 = svld1_s32(body32, buf + 0x3 * dB + 0), d31 = svld1_s32(body32, buf + 0x3 * dB + F);
                    if (N > 0x4) d40 = svld1_s32(body32, buf + 0x4 * dB + 0), d41 = svld1_s32(body32, buf + 0x4 * dB + F);
                    if (N > 0x5) d50 = svld1_s32(body32, buf + 0x5 * dB + 0), d51 = svld1_s32(body32, buf + 0x5 * dB + F);
                    if (N > 0x6) d60 = svld1_s32(body32, buf + 0x6 * dB + 0), d61 = svld1_s32(body32, buf + 0x6 * dB + F);
                    if (N > 0x7) d70 = svld1_s32(body32, buf + 0x7 * dB + 0), d71 = svld1_s32(body32, buf + 0x7 * dB + F);
                    if (N > 0x8) d80 = svld1_s32(body32, buf + 0x8 * dB + 0), d81 = svld1_s32(body32, buf + 0x8 * dB + F);
                    if (N > 0x9) d90 = svld1_s32(body32, buf + 0x9 * dB + 0), d91 = svld1_s32(body32, buf + 0x9 * dB + F);
                    if (N > 0xA) dA0 = svld1_s32(body32, buf + 0xA * dB + 0), dA1 = svld1_s32(body32, buf + 0xA * dB + F);
                    if (N > 0xB) dB0 = svld1_s32(body32, buf + 0xB * dB + 0), dB1 = svld1_s32(body32, buf + 0xB * dB + F);
                }
                else
                {
                    if (N > 0x0) d00 = svdup_n_s32(0), d01 = svdup_n_s32(0);
                    if (N > 0x1) d10 = svdup_n_s32(0), d11 = svdup_n_s32(0);
                    if (N > 0x2) d20 = svdup_n_s32(0), d21 = svdup_n_s32(0);
                    if (N > 0x3) d30 = svdup_n_s32(0), d31 = svdup_n_s32(0);
                    if (N > 0x4) d40 = svdup_n_s32(0), d41 = svdup_n_s32(0);
                    if (N > 0x5) d50 = svdup_n_s32(0), d51 = svdup_n_s32(0);
                    if (N > 0x6) d60 = svdup_n_s32(0), d61 = svdup_n_s32(0);
                    if (N > 0x7) d70 = svdup_n_s32(0), d71 = svdup_n_s32(0);
                    if (N > 0x8) d80 = svdup_n_s32(0), d81 = svdup_n_s32(0);
                    if (N > 0x9) d90 = svdup_n_s32(0), d91 = svdup_n_s32(0);
                    if (N > 0xA) dA0 = svdup_n_s32(0), dA1 = svdup_n_s32(0);
                    if (N > 0xB) dB0 = svdup_n_s32(0), dB1 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * K; k0 < K; k0 += 4, k6 += 4)
                {
                    s0 = svld1_u8(body8, src0);
                    s1 = svld1_u8(body8, src1);
                    if (N > 0x0) w0 = Set4(weight0 + k0), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s1, w0);
                    if (N > 0x1) w0 = Set4(weight1 + k0), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s1, w0);
                    if (N > 0x2) w0 = Set4(weight2 + k0), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s1, w0);
                    if (N > 0x3) w0 = Set4(weight3 + k0), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s1, w0);
                    if (N > 0x4) w0 = Set4(weight4 + k0), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s1, w0);
                    if (N > 0x5) w0 = Set4(weight5 + k0), Madd4<true>(d50, s0, w0), Madd4<true>(d51, s1, w0);
                    if (N > 0x6) w0 = Set4(weight0 + k6), Madd4<true>(d60, s0, w0), Madd4<true>(d61, s1, w0);
                    if (N > 0x7) w0 = Set4(weight1 + k6), Madd4<true>(d70, s0, w0), Madd4<true>(d71, s1, w0);
                    if (N > 0x8) w0 = Set4(weight2 + k6), Madd4<true>(d80, s0, w0), Madd4<true>(d81, s1, w0);
                    if (N > 0x9) w0 = Set4(weight3 + k6), Madd4<true>(d90, s0, w0), Madd4<true>(d91, s1, w0);
                    if (N > 0xA) w0 = Set4(weight4 + k6), Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s1, w0);
                    if (N > 0xB) w0 = Set4(weight5 + k6), Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s1, w0);
                    src0 += A, src1 += A;
                }
                if (M == DF)
                {
                    if (N > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x0, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x1, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x2, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x3, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x4, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x5, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x6, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x7, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x8, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x9, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xA, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xB, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    M -= F;
                    if (N > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x0, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x1, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x2, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x3, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x4, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x5, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x6, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x7, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x8, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x9, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xA, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xB, dNorm, dZero, M), dst += dD, buf += dB;
                }
            }
            else
            {
                if (update)
                {
                    if (N > 0x0) d00 = svld1_s32(body32, buf + 0x0 * dB);
                    if (N > 0x1) d10 = svld1_s32(body32, buf + 0x1 * dB);
                    if (N > 0x2) d20 = svld1_s32(body32, buf + 0x2 * dB);
                    if (N > 0x3) d30 = svld1_s32(body32, buf + 0x3 * dB);
                    if (N > 0x4) d40 = svld1_s32(body32, buf + 0x4 * dB);
                    if (N > 0x5) d50 = svld1_s32(body32, buf + 0x5 * dB);
                    if (N > 0x6) d60 = svld1_s32(body32, buf + 0x6 * dB);
                    if (N > 0x7) d70 = svld1_s32(body32, buf + 0x7 * dB);
                    if (N > 0x8) d80 = svld1_s32(body32, buf + 0x8 * dB);
                    if (N > 0x9) d90 = svld1_s32(body32, buf + 0x9 * dB);
                    if (N > 0xA) dA0 = svld1_s32(body32, buf + 0xA * dB);
                    if (N > 0xB) dB0 = svld1_s32(body32, buf + 0xB * dB);
                }
                else
                {
                    if (N > 0x0) d00 = svdup_n_s32(0);
                    if (N > 0x1) d10 = svdup_n_s32(0);
                    if (N > 0x2) d20 = svdup_n_s32(0);
                    if (N > 0x3) d30 = svdup_n_s32(0);
                    if (N > 0x4) d40 = svdup_n_s32(0);
                    if (N > 0x5) d50 = svdup_n_s32(0);
                    if (N > 0x6) d60 = svdup_n_s32(0);
                    if (N > 0x7) d70 = svdup_n_s32(0);
                    if (N > 0x8) d80 = svdup_n_s32(0);
                    if (N > 0x9) d90 = svdup_n_s32(0);
                    if (N > 0xA) dA0 = svdup_n_s32(0);
                    if (N > 0xB) dB0 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * K; k0 < K; k0 += 4, k6 += 4)
                {
                    s0 = svld1_u8(body8, src0);
                    if (N > 0x0) w0 = Set4(weight0 + k0), Madd4<true>(d00, s0, w0);
                    if (N > 0x1) w0 = Set4(weight1 + k0), Madd4<true>(d10, s0, w0);
                    if (N > 0x2) w0 = Set4(weight2 + k0), Madd4<true>(d20, s0, w0);
                    if (N > 0x3) w0 = Set4(weight3 + k0), Madd4<true>(d30, s0, w0);
                    if (N > 0x4) w0 = Set4(weight4 + k0), Madd4<true>(d40, s0, w0);
                    if (N > 0x5) w0 = Set4(weight5 + k0), Madd4<true>(d50, s0, w0);
                    if (N > 0x6) w0 = Set4(weight0 + k6), Madd4<true>(d60, s0, w0);
                    if (N > 0x7) w0 = Set4(weight1 + k6), Madd4<true>(d70, s0, w0);
                    if (N > 0x8) w0 = Set4(weight2 + k6), Madd4<true>(d80, s0, w0);
                    if (N > 0x9) w0 = Set4(weight3 + k6), Madd4<true>(d90, s0, w0);
                    if (N > 0xA) w0 = Set4(weight4 + k6), Madd4<true>(dA0, s0, w0);
                    if (N > 0xB) w0 = Set4(weight5 + k6), Madd4<true>(dB0, s0, w0);
                    src0 += A;
                }
                if (M == F)
                {
                    if (N > 0x0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x0, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x1, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x2, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x3, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x4, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x5) Save1<term, type>(dst, buf, d50, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x5, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x6) Save1<term, type>(dst, buf, d60, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x6, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x7) Save1<term, type>(dst, buf, d70, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x7, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x8) Save1<term, type>(dst, buf, d80, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x8, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0x9) Save1<term, type>(dst, buf, d90, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x9, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0xA) Save1<term, type>(dst, buf, dA0, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xA, dNorm, dZero), dst += dD, buf += dB;
                    if (N > 0xB) Save1<term, type>(dst, buf, dB0, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xB, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    if (N > 0x0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x0, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x1, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x2, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x3, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x4, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x5) Save1<term, type>(dst, buf, d50, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x5, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x6) Save1<term, type>(dst, buf, d60, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x6, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x7) Save1<term, type>(dst, buf, d70, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x7, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x8) Save1<term, type>(dst, buf, d80, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x8, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0x9) Save1<term, type>(dst, buf, d90, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0x9, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0xA) Save1<term, type>(dst, buf, dA0, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xA, dNorm, dZero, M), dst += dD, buf += dB;
                    if (N > 0xB) Save1<term, type>(dst, buf, dB0, sBias, sNorm, iLo, iHi, iScale, param0, param1, params, 0xB, dNorm, dZero, M), dst += dD, buf += dB;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        typedef void(*SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr)(const int8_t* weight0, const ConvParam& p, const AlgParam& a, size_t K, size_t M, int update,
            const uint8_t* src, const int32_t* sBias, const float* sNorm, const svint32_t& iLo, const svint32_t& iHi, const svfloat32_t& iScale,
            const float* params, const svfloat32_t& param0, const svfloat32_t& param1, const svfloat32_t& dNorm, const svint32_t& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr GetSynetQuantizedConvolutionNchwGemm_Gemm2xN(size_t N)
        {
            switch (N)
            {
            case 0: return NULL;
            case 1: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 1>;
            case 2: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 2>;
            case 3: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 3>;
            case 4: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 4>;
            case 5: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 5>;
            case 6: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 6>;
            case 7: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 7>;
            case 8: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 8>;
            case 9: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 9>;
            case 10: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 10>;
            case 11: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 11>;
            case 12: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 12>;
            default: assert(0);  return NULL;
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static void SynetQuantizedConvolutionNchwGemm_Gemm(const int8_t* weight,
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t K, int update, const uint8_t* src, const int32_t* sBias,
            const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2;
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.bufN, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xN = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(n);
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xM = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(m);

            svfloat32_t _iScale, _param0, _param1, _dNorm;
            svint32_t _dZero = svdup_n_s32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = svdup_n_s32(-iZero);
                _iHi = svdup_n_s32(255 - iZero);
                _iScale = svdup_n_f32(iScale);
                _dNorm = svdup_n_f32(dNorm);
                _param0 = svdup_n_f32(params[0]);
                _param1 = svdup_n_f32(params[1]);
            }
            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const int8_t* w = weight;
                int32_t* b = sum + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                    gemm_2xN(w, p, a, K, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _param0, _param1, _dNorm, _dZero, b, d);
                for (; i < n1; i += m, w += m * dW, b += m * dB, d += m * dD)
                    gemm_2xM(w, p, a, K, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _param0, _param1, _dNorm, _dZero, b, d);
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                gemm[1] = NULL;
            }
        }

         //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNchwGemm::SynetQuantizedConvolutionNchwGemm(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNchwGemm(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, F * 2, 12, 4);
            SetGemm(p, _alg, _gemm);
        }
    }
#endif
}
