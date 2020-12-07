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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE)   
    namespace Avx2
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2x1(const uint8_t * src0,
            const ConvParam8i& p, const AlgParam & a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t * weight0, const __m256* norm,
            const __m256 * bias, const __m256* params, const __m256 * scale, const __m256* shift, int32_t * buf, uint8_t* dst)
        {
            __m256i d00, d01, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(srcC, 4) * A;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            __m256i upper = _mm256_set1_epi32(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                w1 = _mm256_loadu_si256((__m256i*)weight1);
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
                                s0 = _mm256_set1_epi32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    w1 = _mm256_loadu_si256((__m256i*)weight1);
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
                if (dstC == DF)
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper);
                else
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F);
            }
            else
            {
                d00 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight0 += A;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm256_set1_epi32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
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
                if (dstC == F)
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, upper);
                else
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i upper = _mm256_set1_epi32(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256(), d11 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256(), d21 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256(), d31 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256(), d41 = _mm256_setzero_si256();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm256_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                                    if (M > 0) Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 1) Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 2) Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 3) Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 4) Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm256_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                                    if (M > 0) Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 1) Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 2) Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 3) Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 4) Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                                weight0 += dW, weight1 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz;
                    }
                }            
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (dstC == DF)
                    {
                        if (M > 0) Save2<term, type, true>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, true>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, true>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, true>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, true>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save2<term, type, true>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, true>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, true>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, true>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, true>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    }
                }
                else
                {
                    if (dstC == DF)
                    {
                        if (M > 0) Save2<term, type, false>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, false>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, false>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, false>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, false>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save2<term, type, false>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, false>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, false>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, false>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, false>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    }
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                                    weight0 += A;
                                }
                                weight0 += dW;
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm256_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    if (M > 0) Madd4<true>(d00, s0, w0);
                                    if (M > 1) Madd4<true>(d10, s0, w0);
                                    if (M > 2) Madd4<true>(d20, s0, w0);
                                    if (M > 3) Madd4<true>(d30, s0, w0);
                                    if (M > 4) Madd4<true>(d40, s0, w0);
                                    weight0 += A;
                                }
                                weight0 += dW;
                            }
                        }
                        else
                            weight0 += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                                    weight0 += A;
                                }
                                weight0 += dW;
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm256_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                                    if (M > 0) Madd4<false>(d00, s0, w0);
                                    if (M > 1) Madd4<false>(d10, s0, w0);
                                    if (M > 2) Madd4<false>(d20, s0, w0);
                                    if (M > 3) Madd4<false>(d30, s0, w0);
                                    if (M > 4) Madd4<false>(d40, s0, w0);
                                    weight0 += A;
                                }
                                weight0 += dW;
                            }
                        }
                        else
                            weight0 += dWz;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (dstC == F)
                    {
                        if (M > 0) Save1<term, type, true>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, true>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, true>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, true>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, true>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save1<term, type, true>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, true>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, true>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, true>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, true>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    }
                }
                else
                {
                    if (dstC == F)
                    {
                        if (M > 0) Save1<term, type, false>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, false>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, false>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, false>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, false>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save1<term, type, false>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, false>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, false>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, false>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, false>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    }
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, 
            const int8_t* weight0, const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2x1(const ConvParam8i& p)
        {
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                return nofma ? ConvolutionNhwcDirect_2x1<true, term, type, true> : ConvolutionNhwcDirect_2x1<true, term, type, false>;
            else
                return nofma ? ConvolutionNhwcDirect_2x1<false, term, type, true> : ConvolutionNhwcDirect_2x1<false, term, type, false>;
        }

        template<Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_2xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect_2xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect_2xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect_2xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const uint8_t* src,
            const ConvParam8i & p, const AlgParam & a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float * params, const float * scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2x1 = GetConvolutionNhwcDirect_2x1<term, type>(p);
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<term, type>(n);
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            __m256 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm256_set1_ps(params[0]);
            _params[1] = _mm256_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = _mm256_loadu_ps(norm + dc + 0);
                _norm[1] = _mm256_loadu_ps(norm + dc + F);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
                }                
                _scale[0] = _mm256_loadu_ps(scale + dc + 0);
                _scale[1] = _mm256_loadu_ps(scale + dc + F);
                _shift[0] = _mm256_loadu_ps(shift + dc + 0);
                _shift[1] = _mm256_loadu_ps(shift + dc + F);

                uint8_t * d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t * b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i upper = _mm256_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256(), d11 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256(), d21 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256(), d31 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256(), d41 = _mm256_setzero_si256();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm256_loadu_si256((__m256i*)weight0);
                        w1 = _mm256_loadu_si256((__m256i*)weight1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight0 += A, weight1 += A;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm256_loadu_si256((__m256i*)weight0);
                        w1 = _mm256_loadu_si256((__m256i*)weight1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        weight0 += A, weight1 += A;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (dstC == DF)
                    {
                        if (M > 0) Save2<term, type, true>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, true>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, true>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, true>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, true>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save2<term, type, true>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, true>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, true>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, true>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, true>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    }
                }
                else
                {
                    if (dstC == DF)
                    {
                        if (M > 0) Save2<term, type, false>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, false>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, false>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, false>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, false>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save2<term, type, false>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 1) Save2<term, type, false>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 2) Save2<term, type, false>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 3) Save2<term, type, false>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                        if (M > 4) Save2<term, type, false>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    }
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm256_loadu_si256((__m256i*)weight0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight0 += A;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm256_loadu_si256((__m256i*)weight0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                        weight0 += A;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (dstC == F)
                    {
                        if (M > 0) Save1<term, type, true>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, true>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, true>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, true>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, true>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save1<term, type, true>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, true>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, true>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, true>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, true>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    }
                }
                else
                {
                    if (dstC == F)
                    {
                        if (M > 0) Save1<term, type, false>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, false>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, false>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, false>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, false>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    }
                    else
                    {
                        if (M > 0) Save1<term, type, false>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 1) Save1<term, type, false>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 2) Save1<term, type, false>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 3) Save1<term, type, false>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                        if (M > 4) Save1<term, type, false>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    }
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_2xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect1x1_2xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect1x1_2xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect1x1_2xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect1x1_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float * params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<term, type>(n);
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);
            __m256 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm256_set1_ps(params[0]);
            _params[1] = _mm256_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = _mm256_loadu_ps(norm + dc + 0);
                _norm[1] = _mm256_loadu_ps(norm + dc + F);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
                }  
                _scale[0] = _mm256_loadu_ps(scale + dc + 0);
                _scale[1] = _mm256_loadu_ps(scale + dc + F);
                _shift[0] = _mm256_loadu_ps(shift + dc + 0);
                _shift[1] = _mm256_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    convolutionNhwcDirect1x1_2xN(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    convolutionNhwcDirect1x1_2xM(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                weight += DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam & a, ConvolutionPtr * d)
        {
            if (p.Is1x1())
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect1x1_2<term, activation>; break;
                default:
                    assert(0);
                }
            }
            else
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect_2<term, activation>; break;
                default:
                    assert(0);
                }
            }
        }
        
        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            Set<Term8iSingle8u, activation>(p, a, d);
            Set<Term8iSingle32f, activation>(p, a, d);
            Set<Term8iFirst, SimdConvolutionActivationIdentity>(p, a, d);
            Set<Term8iIterim, SimdConvolutionActivationIdentity>(p, a, d);
            Set<Term8iLast8u, activation>(p, a, d);
            Set<Term8iLast32f, activation>(p, a, d);
        }

        static void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr * d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : Sse41::SynetConvolution8iNhwcDirect(p)
        {
            SetAlgParam(F, 2 * F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            Set(p, _alg, _convolutions);
            _convertSrc = Avx2::SynetConvert32fTo8u;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution8iNhwcDepthwise::Preferable(param))
                return new SynetConvolution8iNhwcDepthwise(param);
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
