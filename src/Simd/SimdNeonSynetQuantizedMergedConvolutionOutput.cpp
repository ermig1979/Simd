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
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //------------------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedMergedConvolutionOutputConvolution_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const int32x4_t* bias, const float32x4_t* norm, const int32x4_t& zero, int32_t* buf, uint8_t* dst)
        {
            int32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            uint8x16_t s0;
            int8x16_t w0, w1;
            size_t dS = a.maC * p.strideX, dB = a.owStep, dD = p.dstC;
            const int8_t* weight1 = weight0 + AlignHi(srcC, 4) * F;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0) d00 = vld1q_s32(buf + 0 * dB + 0), d01 = vld1q_s32(buf + 0 * dB + F);
                    if (M > 1) d10 = vld1q_s32(buf + 1 * dB + 0), d11 = vld1q_s32(buf + 1 * dB + F);
                    if (M > 2) d20 = vld1q_s32(buf + 2 * dB + 0), d21 = vld1q_s32(buf + 2 * dB + F);
                    if (M > 3) d30 = vld1q_s32(buf + 3 * dB + 0), d31 = vld1q_s32(buf + 3 * dB + F);
                    if (M > 4) d40 = vld1q_s32(buf + 4 * dB + 0), d41 = vld1q_s32(buf + 4 * dB + F);
                }
                else
                {
                    if (M > 0) d00 = vdupq_n_s32(0), d01 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0), d11 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0), d21 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0), d31 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0), d41 = vdupq_n_s32(0);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = vld1q_s8(weight0);
                    w1 = vld1q_s8(weight1);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                    weight0 += A;
                    weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term>(dst, buf, d00, d01, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 1) Save2<term>(dst, buf, d10, d11, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 2) Save2<term>(dst, buf, d20, d21, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 3) Save2<term>(dst, buf, d30, d31, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 4) Save2<term>(dst, buf, d40, d41, bias, norm, zero), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0) Save2<term>(dst, buf, d00, d01, bias, norm, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 1) Save2<term>(dst, buf, d10, d11, bias, norm, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 2) Save2<term>(dst, buf, d20, d21, bias, norm, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 3) Save2<term>(dst, buf, d30, d31, bias, norm, zero, dstC - F), buf += dB, dst += dD;
                    if (M > 4) Save2<term>(dst, buf, d40, d41, bias, norm, zero, dstC - F), buf += dB, dst += dD;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) d00 = vld1q_s32(buf + 0 * dB);
                    if (M > 1) d10 = vld1q_s32(buf + 1 * dB);
                    if (M > 2) d20 = vld1q_s32(buf + 2 * dB);
                    if (M > 3) d30 = vld1q_s32(buf + 3 * dB);
                    if (M > 4) d40 = vld1q_s32(buf + 4 * dB);
                }
                else
                {
                    if (M > 0) d00 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = vld1q_s8(weight0);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                    weight0 += A;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term>(dst, buf, d00, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 1) Save1<term>(dst, buf, d10, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 2) Save1<term>(dst, buf, d20, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 3) Save1<term>(dst, buf, d30, bias, norm, zero), buf += dB, dst += dD;
                    if (M > 4) Save1<term>(dst, buf, d40, bias, norm, zero), buf += dB, dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term>(dst, buf, d00, bias, norm, zero, dstC), buf += dB, dst += dD;
                    if (M > 1) Save1<term>(dst, buf, d10, bias, norm, zero, dstC), buf += dB, dst += dD;
                    if (M > 2) Save1<term>(dst, buf, d20, bias, norm, zero, dstC), buf += dB, dst += dD;
                    if (M > 3) Save1<term>(dst, buf, d30, bias, norm, zero, dstC), buf += dB, dst += dD;
                    if (M > 4) Save1<term>(dst, buf, d40, bias, norm, zero, dstC), buf += dB, dst += dD;
                }
            }
        }

        typedef void(*QuantizedMergedConvolutionOutputConvolution_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const int32x4_t* bias, const float32x4_t* norm, const int32x4_t& zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> QuantizedMergedConvolutionOutputConvolution_2xM_Ptr GetQuantizedMergedConvolutionOutputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 1>;
            case 2: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 2>;
            case 3: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 3>;
            case 4: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 4>;
            case 5: return QuantizedMergedConvolutionOutputConvolution_2xM<term, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedMergedConvolutionOutputConvolution_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* buf, uint8_t* dst)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            QuantizedMergedConvolutionOutputConvolution_2xM_Ptr outputConvolution1x1_2xN = GetQuantizedMergedConvolutionOutputConvolution_2xM<term>(n);
            QuantizedMergedConvolutionOutputConvolution_2xM_Ptr outputConvolution1x1_2xM = GetQuantizedMergedConvolutionOutputConvolution_2xM<term>(m);
            float32x4_t _norm[2];
            int32x4_t _bias[2], _zero = vdupq_n_s32(zero);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _bias[0] = vld1q_s32(bias + dc + 0);
                _bias[1] = vld1q_s32(bias + dc + F);
                _norm[0] = vld1q_f32(norm + dc + 0);
                _norm[1] = vld1q_f32(norm + dc + F);
                const uint8_t* s = src;
                int32_t* b = buf + dc + yBeg * p.dstW * a.owStep;
                uint8_t* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += a.owStep * n, d += p.dstC * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, update, weight, _bias, _norm, _zero, b, d);
                for (; i < n1; i += m, s += a.maC * m, b += a.owStep * m, d += p.dstC * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, update, weight, _bias, _norm, _zero, b, d);
                weight += AlignHi(maC, 4) * DF;
            }
        }

        //------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionAddInputToOutput(const uint8_t* a, float aNorm, const uint8_t* b, float bNorm, const ConvParam& p, size_t yBeg, size_t yEnd, float dBias, uint8_t* dst)
        {
            float32x4_t _aNorm = vdupq_n_f32(aNorm), _bNorm = vdupq_n_f32(bNorm), _dBias = vdupq_n_f32(dBias);
            size_t beg = yBeg * p.dstW * p.dstC, end = yEnd * p.dstW * p.dstC;
            size_t i = beg, end4 = beg + AlignLo(end - beg, 4), end16 = beg + AlignLo(end - beg, 16);
            for (; i < end16; i += 16)
                QuantizedAdd8u8u8u16(a + i, _aNorm, b + i, _bNorm, _dBias, dst + i);
            for (; i < end4; i += 4)
                QuantizedAdd8u8u8u4(a + i, _aNorm, b + i, _bNorm, _dBias, dst + i);
            for (; i < end; i += 1)
                QuantizedAdd8u8u8u1(a + i, _aNorm, b + i, _bNorm, _dBias, dst + i);
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
