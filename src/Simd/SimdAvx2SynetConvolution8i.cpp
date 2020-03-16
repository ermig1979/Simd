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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;
        using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

        SIMD_INLINE __m256i Set4(const uint8_t* src)
        {
            return _mm256_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m256i& i32, __m256i u8, __m256i i8);

        template<> SIMD_INLINE void Madd4<true>(__m256i& i32, __m256i u8, __m256i i8)
        {
            i32 = _mm256_add_epi32(i32, _mm256_madd_epi16(_mm256_maddubs_epi16(u8, i8), Avx2::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m256i& i32, __m256i u8, __m256i i8)
        {
            __m256i lo = _mm256_madd_epi16(Cvt8uTo16i<0>(u8), Cvt8iTo16i<0>(i8));
            __m256i hi = _mm256_madd_epi16(Cvt8uTo16i<1>(u8), Cvt8iTo16i<1>(i8));
            i32 = _mm256_add_epi32(i32, PermutedHadd32i(lo, hi));
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2x1(const uint8_t * src0,
            const ConvParam8i& p, const AlgParam & a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t * weight0, 
            const __m256i * bias, const __m256i * params, const __m256 * scale, const __m256* shift, int32_t * buf, uint8_t* dst)
        {
            __m256i d00, d01, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(srcC, 4) * A;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            __m256i norm = _mm256_set1_epi32(a.norm);
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
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift);
                else
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, dstC - F);
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
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift);
                else
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, dstC);
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2x5(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t * weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i norm = _mm256_set1_epi32(a.norm);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256(), d11 = _mm256_setzero_si256();
                d20 = _mm256_setzero_si256(), d21 = _mm256_setzero_si256();
                d30 = _mm256_setzero_si256(), d31 = _mm256_setzero_si256();
                d40 = _mm256_setzero_si256(), d41 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + 5 <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                w1 = _mm256_loadu_si256((__m256i*)weight1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                s0 = Set4(src1 + offs);
                                Madd4<overflow>(d10, s0, w0);
                                Madd4<overflow>(d11, s0, w1);
                                s0 = Set4(src2 + offs);
                                Madd4<overflow>(d20, s0, w0);
                                Madd4<overflow>(d21, s0, w1);
                                s0 = Set4(src3 + offs);
                                Madd4<overflow>(d30, s0, w0);
                                Madd4<overflow>(d31, s0, w1);
                                s0 = Set4(src4 + offs);
                                Madd4<overflow>(d40, s0, w0);
                                Madd4<overflow>(d41, s0, w1);
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
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                Madd4<overflow>(d10, s0, w0);
                                Madd4<overflow>(d11, s0, w1);
                                Madd4<overflow>(d20, s0, w0);
                                Madd4<overflow>(d21, s0, w1);
                                Madd4<overflow>(d30, s0, w0);
                                Madd4<overflow>(d31, s0, w1);
                                Madd4<overflow>(d40, s0, w0);
                                Madd4<overflow>(d41, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                }
                else
                {
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, dstC - F);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, dstC - F);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, dstC - F);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, dstC - F);
                    dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, dstC - F);
                    dst += dD, buf += dB;
                }
            }
            else
            {
                d00 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256();
                d20 = _mm256_setzero_si256();
                d30 = _mm256_setzero_si256();
                d40 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + 5 <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                s0 = Set4(src1 + offs);
                                Madd4<overflow>(d10, s0, w0);
                                s0 = Set4(src2 + offs);
                                Madd4<overflow>(d20, s0, w0);
                                s0 = Set4(src3 + offs);
                                Madd4<overflow>(d30, s0, w0);
                                s0 = Set4(src4 + offs);
                                Madd4<overflow>(d40, s0, w0);
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
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d10, s0, w0);
                                Madd4<overflow>(d20, s0, w0);
                                Madd4<overflow>(d30, s0, w0);
                                Madd4<overflow>(d40, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift);
                    dst += dD, buf += dB;
                }
                else
                {
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, dstC);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, dstC);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, dstC);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, dstC);
                    dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, dstC);
                    dst += dD, buf += dB;
                }
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma, int M> void ConvolutionNhwcDirect_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i norm = _mm256_set1_epi32(a.norm);
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
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                w1 = _mm256_loadu_si256((__m256i*)weight1);
                                if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
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
                                if (M > 0) Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 1) Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 2) Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 3) Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 4) Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)weight0);
                                if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0);
                                if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0);
                                if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0);
                                if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0);
                                if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0);
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
                                if (M > 0) Madd4<overflow>(d00, s0, w0);
                                if (M > 1) Madd4<overflow>(d10, s0, w0);
                                if (M > 2) Madd4<overflow>(d20, s0, w0);
                                if (M > 3) Madd4<overflow>(d30, s0, w0);
                                if (M > 4) Madd4<overflow>(d40, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, 
            const int8_t* weight0, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 1>;
            case 2: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 2>;
            case 3: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 3>;
            case 4: return ConvolutionNhwcDirect_2xM<overflow, term, type, nofma, 4>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect_2(const uint8_t* src,
            const ConvParam8i & p, const AlgParam & a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const int32_t* bias, const int32_t * params, const float * scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<overflow, term, type, nofma>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            __m256i _params[2], _bias[2];
            _params[0] = _mm256_setzero_si256();
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm256_set1_epi32(a.high);
            __m256 _scale[2], _shift[2];

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm256_loadu_si256((__m256i*)(bias + dc + 0));
                _bias[1] = _mm256_loadu_si256((__m256i*)(bias + dc + F));
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
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        ConvolutionNhwcDirect_2x5<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
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
                        ConvolutionNhwcDirect_2x5<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
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
                        ConvolutionNhwcDirect_2x5<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type, nofma>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect1x1_2x5(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i norm = _mm256_set1_epi32(a.norm);
            if (dstC > F)
            {
                d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256(), d11 = _mm256_setzero_si256();
                d20 = _mm256_setzero_si256(), d21 = _mm256_setzero_si256();
                d30 = _mm256_setzero_si256(), d31 = _mm256_setzero_si256();
                d40 = _mm256_setzero_si256(), d41 = _mm256_setzero_si256();
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                    s0 = Set4(src0 + offs);
                    Madd4<overflow>(d00, s0, w0);
                    Madd4<overflow>(d01, s0, w1);
                    s0 = Set4(src1 + offs);
                    Madd4<overflow>(d10, s0, w0);
                    Madd4<overflow>(d11, s0, w1);
                    s0 = Set4(src2 + offs);
                    Madd4<overflow>(d20, s0, w0);
                    Madd4<overflow>(d21, s0, w1);
                    s0 = Set4(src3 + offs);
                    Madd4<overflow>(d30, s0, w0);
                    Madd4<overflow>(d31, s0, w1);
                    s0 = Set4(src4 + offs);
                    Madd4<overflow>(d40, s0, w0);
                    Madd4<overflow>(d41, s0, w1);
                    weight0 += A, weight1 += A;
                }
                if (dstC == DF)
                {
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                d00 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256();
                d20 = _mm256_setzero_si256();
                d30 = _mm256_setzero_si256();
                d40 = _mm256_setzero_si256();
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                    s0 = Set4(src0 + offs);
                    Madd4<overflow>(d00, s0, w0);
                    s0 = Set4(src1 + offs);
                    Madd4<overflow>(d10, s0, w0);
                    s0 = Set4(src2 + offs);
                    Madd4<overflow>(d20, s0, w0);
                    s0 = Set4(src3 + offs);
                    Madd4<overflow>(d30, s0, w0);
                    s0 = Set4(src4 + offs);
                    Madd4<overflow>(d40, s0, w0);
                    weight0 += A;
                }
                if (dstC == F)
                {
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                }
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst)
        {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m256i norm = _mm256_set1_epi32(a.norm);
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_si256(), d01 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256(), d11 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256(), d21 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256(), d31 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256(), d41 = _mm256_setzero_si256();
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                    w1 = _mm256_loadu_si256((__m256i*)weight1);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                    weight0 += A, weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type, nofma>(dst, buf, d00, d01, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type, nofma>(dst, buf, d10, d11, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type, nofma>(dst, buf, d20, d21, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type, nofma>(dst, buf, d30, d31, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type, nofma>(dst, buf, d40, d41, norm, bias, params, scale, shift, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_si256();
                if (M > 1) d10 = _mm256_setzero_si256();
                if (M > 2) d20 = _mm256_setzero_si256();
                if (M > 3) d30 = _mm256_setzero_si256();
                if (M > 4) d40 = _mm256_setzero_si256();
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm256_loadu_si256((__m256i*)weight0);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0);
                    weight0 += A;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type, nofma>(dst, buf, d00, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type, nofma>(dst, buf, d10, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type, nofma>(dst, buf, d20, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type, nofma>(dst, buf, d30, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type, nofma>(dst, buf, d40, norm, bias, params, scale, shift, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, int32_t* buf, uint8_t* dst);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 1>;
            case 2: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 2>;
            case 3: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 3>;
            case 4: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma, 4>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, bool nofma> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const int32_t* bias, const int32_t* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = (yEnd - yBeg) * p.dstW, n5 = AlignLoAny(n1, 5), m = n1 - n5;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type, nofma>(m);
            __m256i _params[2], _bias[2];
            _params[0] = _mm256_setzero_si256();
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm256_set1_epi32(a.high);
            __m256 _scale[2], _shift[2];

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm256_loadu_si256((__m256i*)(bias + dc + 0));
                _bias[1] = _mm256_loadu_si256((__m256i*)(bias + dc + F));
                _scale[0] = _mm256_loadu_ps(scale + dc + 0);
                _scale[1] = _mm256_loadu_ps(scale + dc + F);
                _shift[0] = _mm256_loadu_ps(shift + dc + 0);
                _shift[1] = _mm256_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < n5; i += 5, s += p.srcC * 5, b += p.dstC * 5, d += p.dstC * a.size * 5)
                    ConvolutionNhwcDirect1x1_2x5<overflow, term, type, nofma>(s, p, a, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
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
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
