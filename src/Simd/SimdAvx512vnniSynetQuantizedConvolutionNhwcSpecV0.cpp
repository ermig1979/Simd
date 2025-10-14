/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512vnni
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        template<int M> void QuantizedConvolutionNhwcSpecV0_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int update, const int8_t* weight0, int32_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
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
                    if (M > 0x0) d00 = _mm512_loadu_si512(dst + 0x0 * dD + 0), d01 = _mm512_loadu_si512(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = _mm512_loadu_si512(dst + 0x1 * dD + 0), d11 = _mm512_loadu_si512(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = _mm512_loadu_si512(dst + 0x2 * dD + 0), d21 = _mm512_loadu_si512(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = _mm512_loadu_si512(dst + 0x3 * dD + 0), d31 = _mm512_loadu_si512(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = _mm512_loadu_si512(dst + 0x4 * dD + 0), d41 = _mm512_loadu_si512(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = _mm512_loadu_si512(dst + 0x5 * dD + 0), d51 = _mm512_loadu_si512(dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = _mm512_loadu_si512(dst + 0x6 * dD + 0), d61 = _mm512_loadu_si512(dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = _mm512_loadu_si512(dst + 0x7 * dD + 0), d71 = _mm512_loadu_si512(dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = _mm512_loadu_si512(dst + 0x8 * dD + 0), d81 = _mm512_loadu_si512(dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = _mm512_loadu_si512(dst + 0x9 * dD + 0), d91 = _mm512_loadu_si512(dst + 0x9 * dD + F);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(dst + 0xA * dD + 0), dA1 = _mm512_loadu_si512(dst + 0xA * dD + F);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(dst + 0xB * dD + 0), dB1 = _mm512_loadu_si512(dst + 0xB * dD + F);
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
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + dX * 6; offs0 < end; offs0 += 4, offs6 += 4)
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
                if (M > 0x0) _mm512_storeu_si512((__m512i*)(dst + 0x0 * dD + 0), d00), _mm512_storeu_si512((__m512i*)(dst + 0x0 * dD + F), d01);
                if (M > 0x1) _mm512_storeu_si512((__m512i*)(dst + 0x1 * dD + 0), d10), _mm512_storeu_si512((__m512i*)(dst + 0x1 * dD + F), d11);
                if (M > 0x2) _mm512_storeu_si512((__m512i*)(dst + 0x2 * dD + 0), d20), _mm512_storeu_si512((__m512i*)(dst + 0x2 * dD + F), d21);
                if (M > 0x3) _mm512_storeu_si512((__m512i*)(dst + 0x3 * dD + 0), d30), _mm512_storeu_si512((__m512i*)(dst + 0x3 * dD + F), d31);
                if (M > 0x4) _mm512_storeu_si512((__m512i*)(dst + 0x4 * dD + 0), d40), _mm512_storeu_si512((__m512i*)(dst + 0x4 * dD + F), d41);
                if (M > 0x5) _mm512_storeu_si512((__m512i*)(dst + 0x5 * dD + 0), d50), _mm512_storeu_si512((__m512i*)(dst + 0x5 * dD + F), d51);
                if (M > 0x6) _mm512_storeu_si512((__m512i*)(dst + 0x6 * dD + 0), d60), _mm512_storeu_si512((__m512i*)(dst + 0x6 * dD + F), d61);
                if (M > 0x7) _mm512_storeu_si512((__m512i*)(dst + 0x7 * dD + 0), d70), _mm512_storeu_si512((__m512i*)(dst + 0x7 * dD + F), d71);
                if (M > 0x8) _mm512_storeu_si512((__m512i*)(dst + 0x8 * dD + 0), d80), _mm512_storeu_si512((__m512i*)(dst + 0x8 * dD + F), d81);
                if (M > 0x9) _mm512_storeu_si512((__m512i*)(dst + 0x9 * dD + 0), d90), _mm512_storeu_si512((__m512i*)(dst + 0x9 * dD + F), d91);
                if (M > 0xA) _mm512_storeu_si512((__m512i*)(dst + 0xA * dD + 0), dA0), _mm512_storeu_si512((__m512i*)(dst + 0xA * dD + F), dA1);
                if (M > 0xB) _mm512_storeu_si512((__m512i*)(dst + 0xB * dD + 0), dB0), _mm512_storeu_si512((__m512i*)(dst + 0xB * dD + F), dB1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm512_loadu_si512(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = _mm512_loadu_si512(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = _mm512_loadu_si512(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = _mm512_loadu_si512(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = _mm512_loadu_si512(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = _mm512_loadu_si512(dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = _mm512_loadu_si512(dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = _mm512_loadu_si512(dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = _mm512_loadu_si512(dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = _mm512_loadu_si512(dst + 0x9 * dD + 0);
                    if (M > 0xA) dA0 = _mm512_loadu_si512(dst + 0xA * dD + 0);
                    if (M > 0xB) dB0 = _mm512_loadu_si512(dst + 0xB * dD + 0);
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
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs0 = offset[k], end = offs0 + dX, offs6 = offs0 + dX * 6; offs0 < end; offs0 += 4, offs6 += 4)
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
                if (M > 0x0) _mm512_storeu_si512((__m512i*)(dst + 0x0 * dD + 0), d00);
                if (M > 0x1) _mm512_storeu_si512((__m512i*)(dst + 0x1 * dD + 0), d10);
                if (M > 0x2) _mm512_storeu_si512((__m512i*)(dst + 0x2 * dD + 0), d20);
                if (M > 0x3) _mm512_storeu_si512((__m512i*)(dst + 0x3 * dD + 0), d30);
                if (M > 0x4) _mm512_storeu_si512((__m512i*)(dst + 0x4 * dD + 0), d40);
                if (M > 0x5) _mm512_storeu_si512((__m512i*)(dst + 0x5 * dD + 0), d50);
                if (M > 0x6) _mm512_storeu_si512((__m512i*)(dst + 0x6 * dD + 0), d60);
                if (M > 0x7) _mm512_storeu_si512((__m512i*)(dst + 0x7 * dD + 0), d70);
                if (M > 0x8) _mm512_storeu_si512((__m512i*)(dst + 0x8 * dD + 0), d80);
                if (M > 0x9) _mm512_storeu_si512((__m512i*)(dst + 0x9 * dD + 0), d90);
                if (M > 0xA) _mm512_storeu_si512((__m512i*)(dst + 0xA * dD + 0), dA0);
                if (M > 0xB) _mm512_storeu_si512((__m512i*)(dst + 0xB * dD + 0), dB0);

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

        SynetQuantizedConvolutionNhwcSpecV0::SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p)
            : Avx512bw::SynetQuantizedConvolutionNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, 12, F * 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            AlgParam& a = _alg;
            _convolution = QuantizedConvolutionNhwcSpecV0_2;
        }
    }
#endif
}
