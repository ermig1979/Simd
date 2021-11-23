/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx512f
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first);
        typedef void(*ConvolutionNhwcDirect1x1_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first);

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first)
        {
            __m512 d00, d01, d02, s0, w0, w1, w2;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            if (tails[2])
            {
                if (first)
                    d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps(), d02 = _mm512_setzero_ps();
                else
                    d00 = _mm512_loadu_ps(dst + 0 * F), d01 = _mm512_loadu_ps(dst + 1 * F), d02 = _mm512_maskz_loadu_ps(tails[2], dst + 2 * F);
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    size_t beg = (sy + ky) * dY + sx * dX;
                    for (size_t kx = 0; kx < kX; kx += dilX)
                    {
                        if (sy + ky < srcH && sx + kx < srcW)
                        {
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                w1 = _mm512_loadu_ps(weight1 + offw);
                                w2 = _mm512_loadu_ps(weight2 + offw);
                                s0 = _mm512_set1_ps(src0[offs]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01), d02 = _mm512_fmadd_ps(s0, w2, d02);
                            }
                        }
                        weight0 += dW, weight1 += dW, weight2 += dW;
                    }
                }
                Save3<term, type>(dst, d00, d01, d02, bias, params, tails);
            }
            else if (tails[1])
            {
                if (first)
                    d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                else
                    d00 = _mm512_loadu_ps(dst + 0 * F), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 1 * F);
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    size_t beg = (sy + ky) * dY + sx * dX;
                    for (size_t kx = 0; kx < kX; kx += dilX)
                    {
                        if (sy + ky < srcH && sx + kx < srcW)
                        {
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                w1 = _mm512_loadu_ps(weight1 + offw);
                                s0 = _mm512_set1_ps(src0[offs]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                            }
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                Save2<term, type>(dst, d00, d01, bias, params, tails);
            }
            else
            {
                if (first)
                    d00 = _mm512_setzero_ps();
                else
                    d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0 * F);
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    size_t beg = (sy + ky) * dY + sx * dX;
                    for (size_t kx = 0; kx < kX; kx += dilX)
                    {
                        if (sy + ky < srcH && sx + kx < srcW)
                        {
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                s0 = _mm512_set1_ps(src0[offs]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                            }
                        }
                        weight0 += dW;
                    }
                }
                Save1<term, type>(dst, d00, bias, params, tails);
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first)
        {
            __m512 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, d40, d41, d42, d50, d51, d52, d60, d61, d62, d70, d71, d72, d80, d81, d82, s0, w0, w1, w2;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            if (tails[2])
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps(), d02 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps(), d12 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps(), d22 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps(), d32 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps(), d42 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps(), d52 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps(), d62 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps(), d72 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps(), d82 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm512_loadu_ps(dst + 0 * dD + 1 * F), d02 = _mm512_maskz_loadu_ps(tails[2], dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = _mm512_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm512_loadu_ps(dst + 1 * dD + 1 * F), d12 = _mm512_maskz_loadu_ps(tails[2], dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = _mm512_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm512_loadu_ps(dst + 2 * dD + 1 * F), d22 = _mm512_maskz_loadu_ps(tails[2], dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = _mm512_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm512_loadu_ps(dst + 3 * dD + 1 * F), d32 = _mm512_maskz_loadu_ps(tails[2], dst + 3 * dD + 2 * F);
                    if (M > 4) d40 = _mm512_loadu_ps(dst + 4 * dD + 0 * F), d41 = _mm512_loadu_ps(dst + 4 * dD + 1 * F), d42 = _mm512_maskz_loadu_ps(tails[2], dst + 4 * dD + 2 * F);
                    if (M > 5) d50 = _mm512_loadu_ps(dst + 5 * dD + 0 * F), d51 = _mm512_loadu_ps(dst + 5 * dD + 1 * F), d52 = _mm512_maskz_loadu_ps(tails[2], dst + 5 * dD + 2 * F);
                    if (M > 6) d60 = _mm512_loadu_ps(dst + 6 * dD + 0 * F), d61 = _mm512_loadu_ps(dst + 6 * dD + 1 * F), d62 = _mm512_maskz_loadu_ps(tails[2], dst + 6 * dD + 2 * F);
                    if (M > 7) d70 = _mm512_loadu_ps(dst + 7 * dD + 0 * F), d71 = _mm512_loadu_ps(dst + 7 * dD + 1 * F), d72 = _mm512_maskz_loadu_ps(tails[2], dst + 7 * dD + 2 * F);
                    if (M > 8) d80 = _mm512_loadu_ps(dst + 8 * dD + 0 * F), d81 = _mm512_loadu_ps(dst + 8 * dD + 1 * F), d82 = _mm512_maskz_loadu_ps(tails[2], dst + 8 * dD + 2 * F);
                }
                if (p.kernelY * p.kernelX * srcC * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t ky = 0; ky < kY; ky += dilY)
                    {
                        if (sy + ky < srcH)
                        {
                            size_t beg = (sy + ky) * dY + sx * dX;
                            for (size_t kx = 0; kx < kX; kx += dilX)
                            {
                                assert(sx + kx < srcW&& sx + kx + M <= srcW);
                                size_t off0 = beg + kx * dX, end = off0 + srcC, off5 = off0 + 5 * dS, offw = 0;
                                for (; off0 < end; ++off0, ++off5, offw += F)
                                {
                                    PrefetchL1(weight0 + offw);
                                    PrefetchL1(weight1 + offw);
                                    PrefetchL1(weight2 + offw);
                                    w0 = _mm512_loadu_ps(weight0 + offw);
                                    w1 = _mm512_loadu_ps(weight1 + offw);
                                    w2 = _mm512_loadu_ps(weight2 + offw);
                                    if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01), d02 = _mm512_fmadd_ps(s0, w2, d02);
                                    if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11), d12 = _mm512_fmadd_ps(s0, w2, d12);
                                    if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21), d22 = _mm512_fmadd_ps(s0, w2, d22);
                                    if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31), d32 = _mm512_fmadd_ps(s0, w2, d32);
                                    if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41), d42 = _mm512_fmadd_ps(s0, w2, d42);
                                    if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51), d52 = _mm512_fmadd_ps(s0, w2, d52);
                                    if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61), d62 = _mm512_fmadd_ps(s0, w2, d62);
                                    if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71), d72 = _mm512_fmadd_ps(s0, w2, d72);
                                    if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81), d82 = _mm512_fmadd_ps(s0, w2, d82);
                                }
                                weight0 += dW, weight1 += dW, weight2 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz, weight2 += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += dilY)
                    {
                        if (sy + ky < srcH)
                        {
                            size_t beg = (sy + ky) * dY + sx * dX;
                            for (size_t kx = 0; kx < kX; kx += dilX)
                            {
                                assert(sx + kx < srcW&& sx + kx + M <= srcW);
                                size_t off0 = beg + kx * dX, end = off0 + srcC, off5 = off0 + 5 * dS, offw = 0;
                                for (; off0 < end; ++off0, ++off5, offw += F)
                                {
                                    w0 = _mm512_loadu_ps(weight0 + offw);
                                    w1 = _mm512_loadu_ps(weight1 + offw);
                                    w2 = _mm512_loadu_ps(weight2 + offw);
                                    if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01), d02 = _mm512_fmadd_ps(s0, w2, d02);
                                    if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11), d12 = _mm512_fmadd_ps(s0, w2, d12);
                                    if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21), d22 = _mm512_fmadd_ps(s0, w2, d22);
                                    if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31), d32 = _mm512_fmadd_ps(s0, w2, d32);
                                    if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41), d42 = _mm512_fmadd_ps(s0, w2, d42);
                                    if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51), d52 = _mm512_fmadd_ps(s0, w2, d52);
                                    if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61), d62 = _mm512_fmadd_ps(s0, w2, d62);
                                    if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71), d72 = _mm512_fmadd_ps(s0, w2, d72);
                                    if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81), d82 = _mm512_fmadd_ps(s0, w2, d82);
                                }
                                weight0 += dW, weight1 += dW, weight2 += dW;
                            }
                        }
                        else
                            weight0 += dWz, weight1 += dWz, weight2 += dWz;
                    }
                }
                if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, tails), dst += dD;
                if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, tails), dst += dD;
                if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, tails), dst += dD;
                if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, tails), dst += dD;
                if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params, tails), dst += dD;
                if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params, tails), dst += dD;
                if (M > 6) Save3<term, type>(dst, d60, d61, d62, bias, params, tails), dst += dD;
                if (M > 7) Save3<term, type>(dst, d70, d71, d72, bias, params, tails), dst += dD;
                if (M > 8) Save3<term, type>(dst, d80, d81, d82, bias, params, tails), dst += dD;
            }
            else if (tails[1])
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm512_maskz_loadu_ps(tails[1], dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = _mm512_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm512_maskz_loadu_ps(tails[1], dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = _mm512_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm512_maskz_loadu_ps(tails[1], dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = _mm512_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm512_maskz_loadu_ps(tails[1], dst + 3 * dD + 1 * F);
                    if (M > 4) d40 = _mm512_loadu_ps(dst + 4 * dD + 0 * F), d41 = _mm512_maskz_loadu_ps(tails[1], dst + 4 * dD + 1 * F);
                    if (M > 5) d50 = _mm512_loadu_ps(dst + 5 * dD + 0 * F), d51 = _mm512_maskz_loadu_ps(tails[1], dst + 5 * dD + 1 * F);
                    if (M > 6) d60 = _mm512_loadu_ps(dst + 6 * dD + 0 * F), d61 = _mm512_maskz_loadu_ps(tails[1], dst + 6 * dD + 1 * F);
                    if (M > 7) d70 = _mm512_loadu_ps(dst + 7 * dD + 0 * F), d71 = _mm512_maskz_loadu_ps(tails[1], dst + 7 * dD + 1 * F);
                    if (M > 8) d80 = _mm512_loadu_ps(dst + 8 * dD + 0 * F), d81 = _mm512_maskz_loadu_ps(tails[1], dst + 8 * dD + 1 * F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
                            size_t off0 = beg + kx * dX, end = off0 + srcC, off5 = off0 + 5 * dS, offw = 0;
                            for (; off0 < end; ++off0, ++off5, offw += F)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                w1 = _mm512_loadu_ps(weight1 + offw);
                                if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                                if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                                if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                                if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                                if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                                if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                                if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                                if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                                if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 3 * dD + 0 * F);
                    if (M > 4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 4 * dD + 0 * F);
                    if (M > 5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 5 * dD + 0 * F);
                    if (M > 6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 6 * dD + 0 * F);
                    if (M > 7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 7 * dD + 0 * F);
                    if (M > 8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 8 * dD + 0 * F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
                            size_t off0 = beg + kx * dX, end = off0 + srcC, off5 = off0 + 5 * dS, offw = 0;
                            for (; off0 < end; ++off0, ++off5, offw += F)
                            {
                                w0 = _mm512_loadu_ps(weight0 + offw);
                                if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                                if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10);
                                if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
                                if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30);
                                if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
                                if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50);
                                if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60);
                                if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70);
                                if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (M > 0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_3xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_3xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect_3xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect_3xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect_3xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect_3xM<term, type, 5>;
            case 6: return ConvolutionNhwcDirect_3xM<term, type, 6>;
            case 7: return ConvolutionNhwcDirect_3xM<term, type, 7>;
            case 8: return ConvolutionNhwcDirect_3xM<term, type, 8>;
            case 9: return ConvolutionNhwcDirect_3xM<term, type, 9>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 9, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3x1 = ConvolutionNhwcDirect_3x1<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xN = GetConvolutionNhwcDirect_3xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xM = GetConvolutionNhwcDirect_3xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m512 _params[3], _bias[3];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                __mmask16 tails[3] = { TailMask16(dC - 0 * F), TailMask16(dC - 1 * F), TailMask16(dC - 2 * F) };
                if (dC > 0 * F) _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = _mm512_loadu_ps(bias + dc + 2 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = _mm512_loadu_ps(params + dc + 2 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_3xN(src, p, a, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_3xM(src, p, a, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, weight, _bias, _params, d, tails, first);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, const float* weight0, const __m512* bias, const __m512* params, float* dst, const __mmask16* tails, int first)
        {
            __m512 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, d40, d41, d42, d50, d51, d52, d60, d61, d62, d70, d71, d72, d80, d81, d82, s0, w0, w1, w2;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            if (tails[2])
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps(), d02 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps(), d12 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps(), d22 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps(), d32 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps(), d42 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps(), d52 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps(), d62 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps(), d72 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps(), d82 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm512_loadu_ps(dst + 0 * dD + 1 * F), d02 = _mm512_maskz_loadu_ps(tails[2], dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = _mm512_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm512_loadu_ps(dst + 1 * dD + 1 * F), d12 = _mm512_maskz_loadu_ps(tails[2], dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = _mm512_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm512_loadu_ps(dst + 2 * dD + 1 * F), d22 = _mm512_maskz_loadu_ps(tails[2], dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = _mm512_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm512_loadu_ps(dst + 3 * dD + 1 * F), d32 = _mm512_maskz_loadu_ps(tails[2], dst + 3 * dD + 2 * F);
                    if (M > 4) d40 = _mm512_loadu_ps(dst + 4 * dD + 0 * F), d41 = _mm512_loadu_ps(dst + 4 * dD + 1 * F), d42 = _mm512_maskz_loadu_ps(tails[2], dst + 4 * dD + 2 * F);
                    if (M > 5) d50 = _mm512_loadu_ps(dst + 5 * dD + 0 * F), d51 = _mm512_loadu_ps(dst + 5 * dD + 1 * F), d52 = _mm512_maskz_loadu_ps(tails[2], dst + 5 * dD + 2 * F);
                    if (M > 6) d60 = _mm512_loadu_ps(dst + 6 * dD + 0 * F), d61 = _mm512_loadu_ps(dst + 6 * dD + 1 * F), d62 = _mm512_maskz_loadu_ps(tails[2], dst + 6 * dD + 2 * F);
                    if (M > 7) d70 = _mm512_loadu_ps(dst + 7 * dD + 0 * F), d71 = _mm512_loadu_ps(dst + 7 * dD + 1 * F), d72 = _mm512_maskz_loadu_ps(tails[2], dst + 7 * dD + 2 * F);
                    if (M > 8) d80 = _mm512_loadu_ps(dst + 8 * dD + 0 * F), d81 = _mm512_loadu_ps(dst + 8 * dD + 1 * F), d82 = _mm512_maskz_loadu_ps(tails[2], dst + 8 * dD + 2 * F);
                }
                if (srcC * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t off0 = 0, off5 = 5 * dS, offw = 0; off0 < srcC; ++off0, ++off5, offw += F)
                    {
                        PrefetchL1(weight0 + offw);
                        PrefetchL1(weight1 + offw);
                        PrefetchL1(weight2 + offw);
                        w0 = _mm512_loadu_ps(weight0 + offw);
                        w1 = _mm512_loadu_ps(weight1 + offw);
                        w2 = _mm512_loadu_ps(weight2 + offw);
                        if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01), d02 = _mm512_fmadd_ps(s0, w2, d02);
                        if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11), d12 = _mm512_fmadd_ps(s0, w2, d12);
                        if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21), d22 = _mm512_fmadd_ps(s0, w2, d22);
                        if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31), d32 = _mm512_fmadd_ps(s0, w2, d32);
                        if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41), d42 = _mm512_fmadd_ps(s0, w2, d42);
                        if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51), d52 = _mm512_fmadd_ps(s0, w2, d52);
                        if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61), d62 = _mm512_fmadd_ps(s0, w2, d62);
                        if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71), d72 = _mm512_fmadd_ps(s0, w2, d72);
                        if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81), d82 = _mm512_fmadd_ps(s0, w2, d82);
                    }
                }
                else
                {
                    for (size_t off0 = 0, off5 = 5 * dS, offw = 0; off0 < srcC; ++off0, ++off5, offw += F)
                    {
                        w0 = _mm512_loadu_ps(weight0 + offw);
                        w1 = _mm512_loadu_ps(weight1 + offw);
                        w2 = _mm512_loadu_ps(weight2 + offw);
                        if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01), d02 = _mm512_fmadd_ps(s0, w2, d02);
                        if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11), d12 = _mm512_fmadd_ps(s0, w2, d12);
                        if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21), d22 = _mm512_fmadd_ps(s0, w2, d22);
                        if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31), d32 = _mm512_fmadd_ps(s0, w2, d32);
                        if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41), d42 = _mm512_fmadd_ps(s0, w2, d42);
                        if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51), d52 = _mm512_fmadd_ps(s0, w2, d52);
                        if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61), d62 = _mm512_fmadd_ps(s0, w2, d62);
                        if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71), d72 = _mm512_fmadd_ps(s0, w2, d72);
                        if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81), d82 = _mm512_fmadd_ps(s0, w2, d82);
                    }
                }
                if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, tails), dst += dD;
                if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, tails), dst += dD;
                if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, tails), dst += dD;
                if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, tails), dst += dD;
                if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params, tails), dst += dD;
                if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params, tails), dst += dD;
                if (M > 6) Save3<term, type>(dst, d60, d61, d62, bias, params, tails), dst += dD;
                if (M > 7) Save3<term, type>(dst, d70, d71, d72, bias, params, tails), dst += dD;
                if (M > 8) Save3<term, type>(dst, d80, d81, d82, bias, params, tails), dst += dD;
            }
            else if (tails[1])
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm512_loadu_ps(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = _mm512_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm512_loadu_ps(dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = _mm512_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm512_loadu_ps(dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = _mm512_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm512_loadu_ps(dst + 3 * dD + 1 * F);
                    if (M > 4) d40 = _mm512_loadu_ps(dst + 4 * dD + 0 * F), d41 = _mm512_loadu_ps(dst + 4 * dD + 1 * F);
                    if (M > 5) d50 = _mm512_loadu_ps(dst + 5 * dD + 0 * F), d51 = _mm512_loadu_ps(dst + 5 * dD + 1 * F);
                    if (M > 6) d60 = _mm512_loadu_ps(dst + 6 * dD + 0 * F), d61 = _mm512_loadu_ps(dst + 6 * dD + 1 * F);
                    if (M > 7) d70 = _mm512_loadu_ps(dst + 7 * dD + 0 * F), d71 = _mm512_loadu_ps(dst + 7 * dD + 1 * F);
                    if (M > 8) d80 = _mm512_loadu_ps(dst + 8 * dD + 0 * F), d81 = _mm512_loadu_ps(dst + 8 * dD + 1 * F);
                }
                for (size_t off0 = 0, off5 = 5 * dS, offw = 0; off0 < srcC; ++off0, ++off5, offw += F)
                {
                    w0 = _mm512_loadu_ps(weight0 + offw);
                    w1 = _mm512_loadu_ps(weight1 + offw);
                    if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                    if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                    if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                    if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                    if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                    if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                    if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                    if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                    if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                }
                if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, tails), dst += dD;
                if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, tails), dst += dD;
                if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, tails), dst += dD;
                if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, tails), dst += dD;
                if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, tails), dst += dD;
                if (M > 5) Save2<term, type>(dst, d50, d51, bias, params, tails), dst += dD;
                if (M > 6) Save2<term, type>(dst, d60, d61, bias, params, tails), dst += dD;
                if (M > 7) Save2<term, type>(dst, d70, d71, bias, params, tails), dst += dD;
                if (M > 8) Save2<term, type>(dst, d80, d81, bias, params, tails), dst += dD;
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = _mm512_setzero_ps();
                    if (M > 1) d10 = _mm512_setzero_ps();
                    if (M > 2) d20 = _mm512_setzero_ps();
                    if (M > 3) d30 = _mm512_setzero_ps();
                    if (M > 4) d40 = _mm512_setzero_ps();
                    if (M > 5) d50 = _mm512_setzero_ps();
                    if (M > 6) d60 = _mm512_setzero_ps();
                    if (M > 7) d70 = _mm512_setzero_ps();
                    if (M > 8) d80 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm512_maskz_loadu_ps(tails[0], dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = _mm512_maskz_loadu_ps(tails[0], dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = _mm512_maskz_loadu_ps(tails[0], dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = _mm512_maskz_loadu_ps(tails[0], dst + 3 * dD + 0 * F);
                    if (M > 4) d40 = _mm512_maskz_loadu_ps(tails[0], dst + 4 * dD + 0 * F);
                    if (M > 5) d50 = _mm512_maskz_loadu_ps(tails[0], dst + 5 * dD + 0 * F);
                    if (M > 6) d60 = _mm512_maskz_loadu_ps(tails[0], dst + 6 * dD + 0 * F);
                    if (M > 7) d70 = _mm512_maskz_loadu_ps(tails[0], dst + 7 * dD + 0 * F);
                    if (M > 8) d80 = _mm512_maskz_loadu_ps(tails[0], dst + 8 * dD + 0 * F);
                }
                for (size_t off0 = 0, off5 = 5 * dS, offw = 0; off0 < srcC; ++off0, ++off5, offw += F)
                {
                    w0 = _mm512_loadu_ps(weight0 + offw);
                    if (M > 0) s0 = _mm512_set1_ps(src0[off0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                    if (M > 1) s0 = _mm512_set1_ps(src1[off0]), d10 = _mm512_fmadd_ps(s0, w0, d10);
                    if (M > 2) s0 = _mm512_set1_ps(src2[off0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
                    if (M > 3) s0 = _mm512_set1_ps(src3[off0]), d30 = _mm512_fmadd_ps(s0, w0, d30);
                    if (M > 4) s0 = _mm512_set1_ps(src4[off0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
                    if (M > 5) s0 = _mm512_set1_ps(src0[off5]), d50 = _mm512_fmadd_ps(s0, w0, d50);
                    if (M > 6) s0 = _mm512_set1_ps(src1[off5]), d60 = _mm512_fmadd_ps(s0, w0, d60);
                    if (M > 7) s0 = _mm512_set1_ps(src2[off5]), d70 = _mm512_fmadd_ps(s0, w0, d70);
                    if (M > 8) s0 = _mm512_set1_ps(src3[off5]), d80 = _mm512_fmadd_ps(s0, w0, d80);
                }
                if (M > 0) Save1<term, type>(dst, d00, bias, params, tails), dst += dD;
                if (M > 1) Save1<term, type>(dst, d10, bias, params, tails), dst += dD;
                if (M > 2) Save1<term, type>(dst, d20, bias, params, tails), dst += dD;
                if (M > 3) Save1<term, type>(dst, d30, bias, params, tails), dst += dD;
                if (M > 4) Save1<term, type>(dst, d40, bias, params, tails), dst += dD;
                if (M > 5) Save1<term, type>(dst, d50, bias, params, tails), dst += dD;
                if (M > 6) Save1<term, type>(dst, d60, bias, params, tails), dst += dD;
                if (M > 7) Save1<term, type>(dst, d70, bias, params, tails), dst += dD;
                if (M > 8) Save1<term, type>(dst, d80, bias, params, tails), dst += dD;
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_NxM_Ptr GetConvolutionNhwcDirect1x1_3xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_3xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect1x1_3xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect1x1_3xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect1x1_3xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect1x1_3xM<term, type, 5>;
            case 6: return ConvolutionNhwcDirect1x1_3xM<term, type, 6>;
            case 7: return ConvolutionNhwcDirect1x1_3xM<term, type, 7>;
            case 8: return ConvolutionNhwcDirect1x1_3xM<term, type, 8>;
            case 9: return ConvolutionNhwcDirect1x1_3xM<term, type, 9>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t n = 9, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xN = GetConvolutionNhwcDirect1x1_3xM<term, type>(n);
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xM = GetConvolutionNhwcDirect1x1_3xM<term, type>(m);

            __m512 _params[3], _bias[3];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                __mmask16 tails[3] = { TailMask16(dC - 0 * F), TailMask16(dC - 1 * F), TailMask16(dC - 2 * F) };
                if (dC > 0 * F) _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = _mm512_loadu_ps(bias + dc + 2 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = _mm512_loadu_ps(params + dc + 2 * F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_3xN(ps, p, a, srcC, weight, _bias, _params, pd, tails, first);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_3xM(ps, p, a, srcC, weight, _bias, _params, pd, tails, first);
                weight += p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template <TermType term, SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_3<term, type> : ConvolutionNhwcDirect_3<term, type>;
        }

        template <SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            Set<TermLast, type>(p, a);
            Set<TermInterim, SimdConvolutionActivationIdentity>(p, a);
        }

        bool SynetConvolution32fNhwcDirect::Set3r(const ConvParam32f& p, AlgParam& a)
        {
            assert(a.microD == 3 * F);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, a); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, a); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, a); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, a); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, a); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, a); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, a); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, a); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, a); break;
            default: assert(0);
            }
            return true;
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
