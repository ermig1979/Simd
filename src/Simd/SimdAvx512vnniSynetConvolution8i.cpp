/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdAvx512vnni.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;
        using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

        SIMD_INLINE __m512i Set4(const uint8_t* src)
        {
            return _mm512_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m512i& i32, __m512i u8, __m512i i8);

        template<> SIMD_INLINE void Madd4<true>(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_add_epi32(i32, _mm512_madd_epi16(_mm512_maddubs_epi16(u8, i8), Avx512bw::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_dpbusd_epi32(i32, u8, i8);
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2x1(const uint8_t * src0,
            const ConvParam8i& p, const AlgParam & a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t * weight0, 
            const __m512i * bias, const __m512i * params, const __m512 * scale, const __m512* shift, int32_t * buf, uint8_t* dst)
        {
            __m512i d00, d01, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(srcC, 4) * A;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            __m512i norm = _mm512_set1_epi32(a.norm);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                w1 = _mm512_loadu_si512((__m512i*)weight1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm512_set1_epi32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight1);
                                    Madd4<overflow>(d00, s0, w0);
                                    Madd4<overflow>(d01, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                            }
                            else
                                weight0 += dWz, weight1 += dWz;
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                __mmask16 tail = TailMask16(dstC - F);
                Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, tail);
            }
            else
            {
                d00 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight0 += A;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm512_set1_epi32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                                    Madd4<overflow>(d00, s0, w0);
                                    weight0 += A;
                                }
                            }
                            else
                                weight0 += dWz;
                        }
                        weight0 += dW;
                    }
                }
                __mmask16 tail = TailMask16(dstC);
                Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, tail);
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2x12(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t * weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m512i norm = _mm512_set1_epi32(a.norm);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + 12 <= p.srcW);
                            size_t offs0 = (sy + ky) * dY + (sx + kx) * dX, end = offs0 + srcC, offs6 = offs0 + 6 * dS;
                            for (; offs0 < end; offs0 += 4, offs6 += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                w1 = _mm512_loadu_si512((__m512i*)weight1);
                                s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                                s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                                s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                                s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                                s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                                s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                                s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = _mm512_set1_epi32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                w1 = _mm512_loadu_si512((__m512i*)weight1);
                                Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                                Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                                Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                                Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                                Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                                Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                                Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                __mmask16 tail = TailMask16(dstC - F);
                Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d50, d51, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d60, d61, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d70, d71, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d80, d81, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d90, d91, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
            else
            {
                d00 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                d20 = _mm512_setzero_si512();
                d30 = _mm512_setzero_si512();
                d40 = _mm512_setzero_si512();
                d50 = _mm512_setzero_si512();
                d60 = _mm512_setzero_si512();
                d70 = _mm512_setzero_si512();
                d80 = _mm512_setzero_si512();
                d90 = _mm512_setzero_si512();
                dA0 = _mm512_setzero_si512();
                dB0 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + 12 <= p.srcW);
                            size_t offs0 = (sy + ky) * dY + (sx + kx) * dX, end = offs0 + srcC, offs6 = offs0 + 6 * dS;
                            for (; offs0 < end; offs0 += 4, offs6 += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0);
                                s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0);
                                s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0);
                                s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0);
                                s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0);
                                s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0);
                                s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0);
                                s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0);
                                s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0);
                                s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0);
                                s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0);
                                s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = _mm512_set1_epi32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d10, s0, w0);
                                Madd4<overflow>(d20, s0, w0);
                                Madd4<overflow>(d30, s0, w0);
                                Madd4<overflow>(d40, s0, w0);
                                Madd4<overflow>(d50, s0, w0);
                                Madd4<overflow>(d60, s0, w0);
                                Madd4<overflow>(d70, s0, w0);
                                Madd4<overflow>(d80, s0, w0);
                                Madd4<overflow>(d90, s0, w0);
                                Madd4<overflow>(dA0, s0, w0);
                                Madd4<overflow>(dB0, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                __mmask16 tail = TailMask16(dstC);
                Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d50, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d60, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d70, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d80, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d90, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, dA0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, dB0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma, int M> void ConvolutionNhwcDirect_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m512i norm = _mm512_set1_epi32(a.norm);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
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
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs0 = (sy + ky) * dY + (sx + kx) * dX, end = offs0 + srcC, offs6 = offs0 + 6 * dS;
                            for (; offs0 < end; offs0 += 4, offs6 += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                w1 = _mm512_loadu_si512((__m512i*)weight1);
                                if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                                if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                                if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                                if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                                if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                                if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                                if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = _mm512_set1_epi32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                w1 = _mm512_loadu_si512((__m512i*)weight1);
                                if (M > 0x0) Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 0x1) Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 0x2) Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 0x3) Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 0x4) Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                if (M > 0x5) Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                                if (M > 0x6) Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                                if (M > 0x7) Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                                if (M > 0x8) Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                                if (M > 0x9) Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                                if (M > 0xA) Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                                if (M > 0xB) Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                __mmask16 tail = TailMask16(dstC - F);
                if (M > 0x0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x5) Save2<term, type, nofma>(dst, buf, d50, d51, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x6) Save2<term, type, nofma>(dst, buf, d60, d61, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x7) Save2<term, type, nofma>(dst, buf, d70, d71, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x8) Save2<term, type, nofma>(dst, buf, d80, d81, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x9) Save2<term, type, nofma>(dst, buf, d90, d91, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xA) Save2<term, type, nofma>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xB) Save2<term, type, nofma>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
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
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs0 = (sy + ky) * dY + (sx + kx) * dX, end = offs0 + srcC, offs6 = offs0 + 6 * dS;
                            for (; offs0 < end; offs0 += 4, offs6 += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0);
                                if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0);
                                if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0);
                                if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0);
                                if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0);
                                if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0);
                                if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0);
                                if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0);
                                if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0);
                                if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0);
                                if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0);
                                if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = _mm512_set1_epi32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight0);
                                if (M > 0x0) Madd4<overflow>(d00, s0, w0);
                                if (M > 0x1) Madd4<overflow>(d10, s0, w0);
                                if (M > 0x2) Madd4<overflow>(d20, s0, w0);
                                if (M > 0x3) Madd4<overflow>(d30, s0, w0);
                                if (M > 0x4) Madd4<overflow>(d40, s0, w0);
                                if (M > 0x5) Madd4<overflow>(d50, s0, w0);
                                if (M > 0x6) Madd4<overflow>(d60, s0, w0);
                                if (M > 0x7) Madd4<overflow>(d70, s0, w0);
                                if (M > 0x8) Madd4<overflow>(d80, s0, w0);
                                if (M > 0x9) Madd4<overflow>(d90, s0, w0);
                                if (M > 0xA) Madd4<overflow>(dA0, s0, w0);
                                if (M > 0xB) Madd4<overflow>(dB0, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                __mmask16 tail = TailMask16(dstC);
                if (M > 0x0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x5) Save1<term, type, nofma>(dst, buf, d50, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x6) Save1<term, type, nofma>(dst, buf, d60, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x7) Save1<term, type, nofma>(dst, buf, d70, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x8) Save1<term, type, nofma>(dst, buf, d80, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x9) Save1<term, type, nofma>(dst, buf, d90, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xA) Save1<term, type, nofma>(dst, buf, dA0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xB) Save1<term, type, nofma>(dst, buf, dB0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
        }

        typedef void(*ConvolutionNhwcDirect_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, 
            const int8_t* weight0, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: NULL;
            case 0x1: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x1>;
            case 0x2: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x2>;
            case 0x3: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x3>;
            case 0x4: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x4>;
            case 0x5: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x5>;
            case 0x6: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x6>;
            case 0x7: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x7>;
            case 0x8: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x8>;
            case 0x9: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0x9>;
            case 0xA: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0xA>;
            case 0xB: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 0xB>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2(const uint8_t* src,
            const ConvParam8i & p, const AlgParam & a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const int32_t* bias, const int32_t * params, const float * scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 12, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<overflow, term, type, nofma>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            __m512i _params[2], _bias[2];
            _params[0] = _mm512_setzero_si512();
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm512_set1_epi32(a.high);
            __m512 _scale[2], _shift[2];

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc + 0));
                _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc + F));
                _scale[0] = _mm512_loadu_ps(scale + dc + 0);
                _scale[1] = _mm512_loadu_ps(scale + dc + F);
                _shift[0] = _mm512_loadu_ps(shift + dc + 0);
                _shift[1] = _mm512_loadu_ps(shift + dc + F);

                uint8_t * d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t * b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        ConvolutionNhwcDirect_2x12<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        ConvolutionNhwcDirect_2x12<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        ConvolutionNhwcDirect_2x12<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect1x1_2x12(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m512i norm = _mm512_set1_epi32(a.norm);
            if (dstC > F)
            {
                d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    w1 = _mm512_loadu_si512((__m512i*)weight1);
                    s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                    s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                    s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                    s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                    s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                    s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                    s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                    s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                    s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                    s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                    s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                    s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                    weight0 += A, weight1 += A;
                }
                __mmask16 tail = TailMask16(dstC - F);
                Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d50, d51, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d60, d61, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d70, d71, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d80, d81, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, d90, d91, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save2<term, type, nofma>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
            else
            {
                d00 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                d20 = _mm512_setzero_si512();
                d30 = _mm512_setzero_si512();
                d40 = _mm512_setzero_si512();
                d50 = _mm512_setzero_si512();
                d60 = _mm512_setzero_si512();
                d70 = _mm512_setzero_si512();
                d80 = _mm512_setzero_si512();
                d90 = _mm512_setzero_si512();
                dA0 = _mm512_setzero_si512();
                dB0 = _mm512_setzero_si512();
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0);
                    s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0);
                    s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0);
                    s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0);
                    s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0);
                    s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0);
                    s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0);
                    s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0);
                    s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0);
                    s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0);
                    s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0);
                    s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0);
                    weight0 += A;
                }
                __mmask16 tail = TailMask16(dstC);
                Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d50, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d60, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d70, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d80, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, d90, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, dA0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                Save1<term, type, nofma>(dst, buf, dB0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m512i norm = _mm512_set1_epi32(a.norm);
            if (dstC > F)
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
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    w1 = _mm512_loadu_si512((__m512i*)weight1);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0), Madd4<overflow>(d51, s0, w1);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0), Madd4<overflow>(d61, s0, w1);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0), Madd4<overflow>(d71, s0, w1);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0), Madd4<overflow>(d81, s0, w1);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0), Madd4<overflow>(d91, s0, w1);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0), Madd4<overflow>(dA1, s0, w1);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0), Madd4<overflow>(dB1, s0, w1);
                    weight0 += A, weight1 += A;
                }
                __mmask16 tail = TailMask16(dstC - F);
                if (M > 0x0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x5) Save2<term, type, nofma>(dst, buf, d50, d51, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x6) Save2<term, type, nofma>(dst, buf, d60, d61, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x7) Save2<term, type, nofma>(dst, buf, d70, d71, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x8) Save2<term, type, nofma>(dst, buf, d80, d81, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x9) Save2<term, type, nofma>(dst, buf, d90, d91, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xA) Save2<term, type, nofma>(dst, buf, dA0, dA1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xB) Save2<term, type, nofma>(dst, buf, dB0, dB1, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
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
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 4, offs6 += 4)
                {
                    w0 = _mm512_loadu_si512((__m512i*)weight0);
                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<overflow>(d00, s0, w0);
                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<overflow>(d10, s0, w0);
                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<overflow>(d20, s0, w0);
                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<overflow>(d30, s0, w0);
                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<overflow>(d40, s0, w0);
                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<overflow>(d50, s0, w0);
                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<overflow>(d60, s0, w0);
                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<overflow>(d70, s0, w0);
                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<overflow>(d80, s0, w0);
                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<overflow>(d90, s0, w0);
                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<overflow>(dA0, s0, w0);
                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<overflow>(dB0, s0, w0);
                    weight0 += A;
                }
                __mmask16 tail = TailMask16(dstC);
                if (M > 0x0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x5) Save1<term, type, nofma>(dst, buf, d50, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x6) Save1<term, type, nofma>(dst, buf, d60, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x7) Save1<term, type, nofma>(dst, buf, d70, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x8) Save1<term, type, nofma>(dst, buf, d80, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0x9) Save1<term, type, nofma>(dst, buf, d90, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xA) Save1<term, type, nofma>(dst, buf, dA0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
                if (M > 0xB) Save1<term, type, nofma>(dst, buf, dB0, norm, bias, params, scale, shift, tail), dst += dD, buf += dB;
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x1>;
            case 0x2: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x2>;
            case 0x3: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x3>;
            case 0x4: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x4>;
            case 0x5: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x5>;
            case 0x6: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x6>;
            case 0x7: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x7>;
            case 0x8: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x8>;
            case 0x9: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0x9>;
            case 0xA: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0xA>;
            case 0xB: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 0xB>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const int32_t* bias, const int32_t* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = (yEnd - yBeg) * p.dstW, n12 = AlignLoAny(n1, 12), m = n1 - n12;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma>(m);
            __m512i _params[2], _bias[2];
            _params[0] = _mm512_setzero_si512();
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm512_set1_epi32(a.high);
            __m512 _scale[2], _shift[2];

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc + 0));
                _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc + F));
                _scale[0] = _mm512_loadu_ps(scale + dc + 0);
                _scale[1] = _mm512_loadu_ps(scale + dc + F);
                _shift[0] = _mm512_loadu_ps(shift + dc + 0);
                _shift[1] = _mm512_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < n12; i += 12, s += p.srcC * 12, b += p.dstC * 12, d += p.dstC * a.size * 12)
                    ConvolutionNhwcDirect1x1_2x12<overflow, term, type, nofma>(s, p, a, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    convolutionNhwcDirect1x1_2xM(s, p, a, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                weight += DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template <bool overflow, Term8iType term, SimdConvolutionActivationType activation, bool nofma> void Set(const ConvParam8i& p, const AlgParam & a, ConvolutionPtr * d)
        {
            if (p.Is1x1())
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect1x1_2<overflow, term, activation, nofma>; break;
                default:
                    assert(0);
                }
            }
            else
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect_2<overflow, term, activation, nofma>; break;
                default:
                    assert(0);
                }
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (p.compatibility & SimdSynetCompatibilityNoFma)
                Set<overflow, term, activation, true>(p, a, d);
            else
                Set<overflow, term, activation, false>(p, a, d);
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (p.compatibility & SimdSynetCompatibilityOverflow16i)
                Set<true, term, activation>(p, a, d);
            else
                Set<false, term, activation>(p, a, d);
        }        
        
        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iFirst, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iIterim, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iLast8u, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iLast32f, activation>(p, a, d);
        }

        static void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr * d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, a, d); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : Avx512bw::SynetConvolution8iNhwcDirect(p)
        {
            SetAlgParam(F, 2 * F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            Set(p, _alg, _convolutions);
            _convertSrc = Avx512bw::SynetConvert32fTo8u;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
