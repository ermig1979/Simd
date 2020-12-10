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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x12(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (tails[1])
            {
                d00 = _mm512_setzero_ps(); d01 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps(); d11 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps(); d21 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps(); d31 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps(); d41 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps(); d51 = _mm512_setzero_ps();
                d60 = _mm512_setzero_ps(); d61 = _mm512_setzero_ps();
                d70 = _mm512_setzero_ps(); d71 = _mm512_setzero_ps();
                d80 = _mm512_setzero_ps(); d81 = _mm512_setzero_ps();
                d90 = _mm512_setzero_ps(); d91 = _mm512_setzero_ps();
                da0 = _mm512_setzero_ps(); da1 = _mm512_setzero_ps();
                db0 = _mm512_setzero_ps(); db1 = _mm512_setzero_ps();
                if (kernelH * kernelW * srcC * 2 * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset0 = ky * dY + kx * dX, offset6 = offset0 + 6 * dS, end0 = offset0 + srcC; offset0 < end0; ++offset0, ++offset6)
                            {
                                PrefetchL1(weight + 0);
                                PrefetchL1(weight + F);
                                w0 = _mm512_loadu_ps(weight + 0);
                                w1 = _mm512_loadu_ps(weight + F);
                                s0 = _mm512_set1_ps(src0[offset0]);
                                d00 = _mm512_fmadd_ps(s0, w0, d00);
                                d01 = _mm512_fmadd_ps(s0, w1, d01);
                                s0 = _mm512_set1_ps(src1[offset0]);
                                d10 = _mm512_fmadd_ps(s0, w0, d10);
                                d11 = _mm512_fmadd_ps(s0, w1, d11);
                                s0 = _mm512_set1_ps(src2[offset0]);
                                d20 = _mm512_fmadd_ps(s0, w0, d20);
                                d21 = _mm512_fmadd_ps(s0, w1, d21);
                                s0 = _mm512_set1_ps(src3[offset0]);
                                d30 = _mm512_fmadd_ps(s0, w0, d30);
                                d31 = _mm512_fmadd_ps(s0, w1, d31);
                                s0 = _mm512_set1_ps(src4[offset0]);
                                d40 = _mm512_fmadd_ps(s0, w0, d40);
                                d41 = _mm512_fmadd_ps(s0, w1, d41);
                                s0 = _mm512_set1_ps(src5[offset0]);
                                d50 = _mm512_fmadd_ps(s0, w0, d50);
                                d51 = _mm512_fmadd_ps(s0, w1, d51);
                                s0 = _mm512_set1_ps(src0[offset6]);
                                d60 = _mm512_fmadd_ps(s0, w0, d60);
                                d61 = _mm512_fmadd_ps(s0, w1, d61);
                                s0 = _mm512_set1_ps(src1[offset6]);
                                d70 = _mm512_fmadd_ps(s0, w0, d70);
                                d71 = _mm512_fmadd_ps(s0, w1, d71);
                                s0 = _mm512_set1_ps(src2[offset6]);
                                d80 = _mm512_fmadd_ps(s0, w0, d80);
                                d81 = _mm512_fmadd_ps(s0, w1, d81);
                                s0 = _mm512_set1_ps(src3[offset6]);
                                d90 = _mm512_fmadd_ps(s0, w0, d90);
                                d91 = _mm512_fmadd_ps(s0, w1, d91);
                                s0 = _mm512_set1_ps(src4[offset6]);
                                da0 = _mm512_fmadd_ps(s0, w0, da0);
                                da1 = _mm512_fmadd_ps(s0, w1, da1);
                                s0 = _mm512_set1_ps(src5[offset6]);
                                db0 = _mm512_fmadd_ps(s0, w0, db0);
                                db1 = _mm512_fmadd_ps(s0, w1, db1);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset0 = ky * dY + kx * dX, offset6 = offset0 + 6 * dS, end0 = offset0 + srcC; offset0 < end0; ++offset0, ++offset6)
                            {
                                w0 = _mm512_loadu_ps(weight + 0);
                                w1 = _mm512_loadu_ps(weight + F);
                                s0 = _mm512_set1_ps(src0[offset0]);
                                d00 = _mm512_fmadd_ps(s0, w0, d00);
                                d01 = _mm512_fmadd_ps(s0, w1, d01);
                                s0 = _mm512_set1_ps(src1[offset0]);
                                d10 = _mm512_fmadd_ps(s0, w0, d10);
                                d11 = _mm512_fmadd_ps(s0, w1, d11);
                                s0 = _mm512_set1_ps(src2[offset0]);
                                d20 = _mm512_fmadd_ps(s0, w0, d20);
                                d21 = _mm512_fmadd_ps(s0, w1, d21);
                                s0 = _mm512_set1_ps(src3[offset0]);
                                d30 = _mm512_fmadd_ps(s0, w0, d30);
                                d31 = _mm512_fmadd_ps(s0, w1, d31);
                                s0 = _mm512_set1_ps(src4[offset0]);
                                d40 = _mm512_fmadd_ps(s0, w0, d40);
                                d41 = _mm512_fmadd_ps(s0, w1, d41);
                                s0 = _mm512_set1_ps(src5[offset0]);
                                d50 = _mm512_fmadd_ps(s0, w0, d50);
                                d51 = _mm512_fmadd_ps(s0, w1, d51);
                                s0 = _mm512_set1_ps(src0[offset6]);
                                d60 = _mm512_fmadd_ps(s0, w0, d60);
                                d61 = _mm512_fmadd_ps(s0, w1, d61);
                                s0 = _mm512_set1_ps(src1[offset6]);
                                d70 = _mm512_fmadd_ps(s0, w0, d70);
                                d71 = _mm512_fmadd_ps(s0, w1, d71);
                                s0 = _mm512_set1_ps(src2[offset6]);
                                d80 = _mm512_fmadd_ps(s0, w0, d80);
                                d81 = _mm512_fmadd_ps(s0, w1, d81);
                                s0 = _mm512_set1_ps(src3[offset6]);
                                d90 = _mm512_fmadd_ps(s0, w0, d90);
                                d91 = _mm512_fmadd_ps(s0, w1, d91);
                                s0 = _mm512_set1_ps(src4[offset6]);
                                da0 = _mm512_fmadd_ps(s0, w0, da0);
                                da1 = _mm512_fmadd_ps(s0, w1, da1);
                                s0 = _mm512_set1_ps(src5[offset6]);
                                db0 = _mm512_fmadd_ps(s0, w0, db0);
                                db1 = _mm512_fmadd_ps(s0, w1, db1);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d60, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d61, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d70, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d71, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d80, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d81, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d90, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d91, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, da0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, da1, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, db0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, db1, bias, params, tails[1]);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps();
                d60 = _mm512_setzero_ps();
                d70 = _mm512_setzero_ps();
                d80 = _mm512_setzero_ps();
                d90 = _mm512_setzero_ps();
                da0 = _mm512_setzero_ps();
                db0 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset0 = ky * dY + kx * dX, offset6 = offset0 + 6 * dS, end0 = offset0 + srcC; offset0 < end0; ++offset0, ++offset6)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            s0 = _mm512_set1_ps(src0[offset0]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            s0 = _mm512_set1_ps(src1[offset0]);
                            d10 = _mm512_fmadd_ps(s0, w0, d10);
                            s0 = _mm512_set1_ps(src2[offset0]);
                            d20 = _mm512_fmadd_ps(s0, w0, d20);
                            s0 = _mm512_set1_ps(src3[offset0]);
                            d30 = _mm512_fmadd_ps(s0, w0, d30);
                            s0 = _mm512_set1_ps(src4[offset0]);
                            d40 = _mm512_fmadd_ps(s0, w0, d40);
                            s0 = _mm512_set1_ps(src5[offset0]);
                            d50 = _mm512_fmadd_ps(s0, w0, d50);
                            s0 = _mm512_set1_ps(src0[offset6]);
                            d60 = _mm512_fmadd_ps(s0, w0, d60);
                            s0 = _mm512_set1_ps(src1[offset6]);
                            d70 = _mm512_fmadd_ps(s0, w0, d70);
                            s0 = _mm512_set1_ps(src2[offset6]);
                            d80 = _mm512_fmadd_ps(s0, w0, d80);
                            s0 = _mm512_set1_ps(src3[offset6]);
                            d90 = _mm512_fmadd_ps(s0, w0, d90);
                            s0 = _mm512_set1_ps(src4[offset6]);
                            da0 = _mm512_fmadd_ps(s0, w0, da0);
                            s0 = _mm512_set1_ps(src5[offset6]);
                            db0 = _mm512_fmadd_ps(s0, w0, db0);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d60, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d70, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d80, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d90, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, da0, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, db0, bias, params, tails[0]);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x6(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (tails[1])
            {
                d00 = _mm512_setzero_ps(); d01 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps(); d11 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps(); d21 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps(); d31 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps(); d41 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps(); d51 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            w1 = _mm512_loadu_ps(weight + F);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            d01 = _mm512_fmadd_ps(s0, w1, d01);
                            s0 = _mm512_set1_ps(src1[offset]);
                            d10 = _mm512_fmadd_ps(s0, w0, d10);
                            d11 = _mm512_fmadd_ps(s0, w1, d11);
                            s0 = _mm512_set1_ps(src2[offset]);
                            d20 = _mm512_fmadd_ps(s0, w0, d20);
                            d21 = _mm512_fmadd_ps(s0, w1, d21);
                            s0 = _mm512_set1_ps(src3[offset]);
                            d30 = _mm512_fmadd_ps(s0, w0, d30);
                            d31 = _mm512_fmadd_ps(s0, w1, d31);
                            s0 = _mm512_set1_ps(src4[offset]);
                            d40 = _mm512_fmadd_ps(s0, w0, d40);
                            d41 = _mm512_fmadd_ps(s0, w1, d41);
                            s0 = _mm512_set1_ps(src5[offset]);
                            d50 = _mm512_fmadd_ps(s0, w0, d50);
                            d51 = _mm512_fmadd_ps(s0, w1, d51);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            s0 = _mm512_set1_ps(src1[offset]);
                            d10 = _mm512_fmadd_ps(s0, w0, d10);
                            s0 = _mm512_set1_ps(src2[offset]);
                            d20 = _mm512_fmadd_ps(s0, w0, d20);
                            s0 = _mm512_set1_ps(src3[offset]);
                            d30 = _mm512_fmadd_ps(s0, w0, d30);
                            s0 = _mm512_set1_ps(src4[offset]);
                            d40 = _mm512_fmadd_ps(s0, w0, d40);
                            s0 = _mm512_set1_ps(src5[offset]);
                            d50 = _mm512_fmadd_ps(s0, w0, d50);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x3(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            if (tails[1])
            {
                d00 = _mm512_setzero_ps(); d01 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps(); d11 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps(); d21 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            w1 = _mm512_loadu_ps(weight + F);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            d01 = _mm512_fmadd_ps(s0, w1, d01);
                            s0 = _mm512_set1_ps(src1[offset]);
                            d10 = _mm512_fmadd_ps(s0, w0, d10);
                            d11 = _mm512_fmadd_ps(s0, w1, d11);
                            s0 = _mm512_set1_ps(src2[offset]);
                            d20 = _mm512_fmadd_ps(s0, w0, d20);
                            d21 = _mm512_fmadd_ps(s0, w1, d21);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            s0 = _mm512_set1_ps(src1[offset]);
                            d10 = _mm512_fmadd_ps(s0, w0, d10);
                            s0 = _mm512_set1_ps(src2[offset]);
                            d20 = _mm512_fmadd_ps(s0, w0, d20);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, s0, w0, w1;
            size_t dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC;
            if (tails[1])
            {
                d00 = _mm512_setzero_ps();
                d01 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            w1 = _mm512_loadu_ps(weight + F);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            d01 = _mm512_fmadd_ps(s0, w1, d01);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = _mm512_loadu_ps(weight + 0);
                            s0 = _mm512_set1_ps(src0[offset]);
                            d00 = _mm512_fmadd_ps(s0, w0, d00);
                            weight += DF;
                        }
                    }
                    weight += dW;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseH = p.padY, noseW = p.padX;
            size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
            size_t bodyW3 = bodyW < noseW ? 0 : AlignLoAny(bodyW - noseW, 3 * p.strideX) + noseW;
            size_t bodyW6 = bodyW < noseW ? 0 : AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
            size_t bodyW12 = 0;// bodyW < noseW ? 0 : AlignLoAny(bodyW - noseW, 12 * p.strideX) + noseW;
            size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t tail = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(tail), TailMask16(tail - F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg, sy = dy * p.strideY;
                for (; sy < noseH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src;
                    const float* w = weight + (noseH - sy) * p.kernelX * srcC * DF;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, kY + sy, kX + sx, srcC, w + (noseW - sx) * srcC * DF, _bias, _params, d, tails);
                    for (; sx < bodyW12; sx += 12 * p.strideX, d += 12 * p.dstC)
                        ConvolutionNhwcDirect_2x12<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, kW - sx, srcC, w, _bias, _params, d, tails);
                }
                for (; sy < bodyH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                    const float* w = weight;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, p.kernelY, kX + sx, srcC, w + (noseW - sx) * srcC * DF, _bias, _params, d, tails);
                    for (; sx < bodyW12; sx += 12 * p.strideX, d += 12 * p.dstC)
                        ConvolutionNhwcDirect_2x12<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, kW - sx, srcC, w, _bias, _params, d, tails);
                }
                for (; sy < tailH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                    const float* w = weight;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, kH - sy, kX + sx, srcC, w + (noseW - sx) * srcC * DF, _bias, _params, d, tails);
                    for (; sx < bodyW12; sx += 12 * p.strideX, d += 12 * p.dstC)
                        ConvolutionNhwcDirect_2x12<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, w, _bias, _params, d, tails);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, kW - sx, srcC, w, _bias, _params, d, tails);
                }
                weight += p.kernelY * p.kernelX * srcC * DF;
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p,
            const SynetConvolution32fNhwcDirect::AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    size_t macroK = p.kernelY * p.kernelX * macroC;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (a.macroC == p.srcC)
                            ConvolutionNhwcDirect_2<TermSingle, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else if (sc == 0)
                            ConvolutionNhwcDirect_2<TermFirst, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else
                            ConvolutionNhwcDirect_2<TermIterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2x14(const float* src0, const ConvParam32f& p,
            size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            const float* src6 = src0 + 6 * dS;
            if (tails[1])
            {
                d00 = _mm512_setzero_ps(); d01 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps(); d11 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps(); d21 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps(); d31 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps(); d41 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps(); d51 = _mm512_setzero_ps();
                d60 = _mm512_setzero_ps(); d61 = _mm512_setzero_ps();
                d70 = _mm512_setzero_ps(); d71 = _mm512_setzero_ps();
                d80 = _mm512_setzero_ps(); d81 = _mm512_setzero_ps();
                d90 = _mm512_setzero_ps(); d91 = _mm512_setzero_ps();
                da0 = _mm512_setzero_ps(); da1 = _mm512_setzero_ps();
                db0 = _mm512_setzero_ps(); db1 = _mm512_setzero_ps();
                dc0 = _mm512_setzero_ps(); dc1 = _mm512_setzero_ps();
                dd0 = _mm512_setzero_ps(); dd1 = _mm512_setzero_ps();
                if (srcC * 2 * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        PrefetchL1(weight + 0);
                        PrefetchL1(weight + F);
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        s0 = _mm512_set1_ps(src0[offset0]);
                        d00 = _mm512_fmadd_ps(s0, w0, d00);
                        d01 = _mm512_fmadd_ps(s0, w1, d01);
                        s0 = _mm512_set1_ps(src1[offset0]);
                        d10 = _mm512_fmadd_ps(s0, w0, d10);
                        d11 = _mm512_fmadd_ps(s0, w1, d11);
                        s0 = _mm512_set1_ps(src2[offset0]);
                        d20 = _mm512_fmadd_ps(s0, w0, d20);
                        d21 = _mm512_fmadd_ps(s0, w1, d21);
                        s0 = _mm512_set1_ps(src3[offset0]);
                        d30 = _mm512_fmadd_ps(s0, w0, d30);
                        d31 = _mm512_fmadd_ps(s0, w1, d31);
                        s0 = _mm512_set1_ps(src4[offset0]);
                        d40 = _mm512_fmadd_ps(s0, w0, d40);
                        d41 = _mm512_fmadd_ps(s0, w1, d41);
                        s0 = _mm512_set1_ps(src5[offset0]);
                        d50 = _mm512_fmadd_ps(s0, w0, d50);
                        d51 = _mm512_fmadd_ps(s0, w1, d51);
                        s0 = _mm512_set1_ps(src6[offset0]);
                        d60 = _mm512_fmadd_ps(s0, w0, d60);
                        d61 = _mm512_fmadd_ps(s0, w1, d61);
                        s0 = _mm512_set1_ps(src0[offset7]);
                        d70 = _mm512_fmadd_ps(s0, w0, d70);
                        d71 = _mm512_fmadd_ps(s0, w1, d71);
                        s0 = _mm512_set1_ps(src1[offset7]);
                        d80 = _mm512_fmadd_ps(s0, w0, d80);
                        d81 = _mm512_fmadd_ps(s0, w1, d81);
                        s0 = _mm512_set1_ps(src2[offset7]);
                        d90 = _mm512_fmadd_ps(s0, w0, d90);
                        d91 = _mm512_fmadd_ps(s0, w1, d91);
                        s0 = _mm512_set1_ps(src3[offset7]);
                        da0 = _mm512_fmadd_ps(s0, w0, da0);
                        da1 = _mm512_fmadd_ps(s0, w1, da1);
                        s0 = _mm512_set1_ps(src4[offset7]);
                        db0 = _mm512_fmadd_ps(s0, w0, db0);
                        db1 = _mm512_fmadd_ps(s0, w1, db1);
                        s0 = _mm512_set1_ps(src5[offset7]);
                        dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                        dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        s0 = _mm512_set1_ps(src6[offset7]);
                        dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                        dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                else
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        s0 = _mm512_set1_ps(src0[offset0]);
                        d00 = _mm512_fmadd_ps(s0, w0, d00);
                        d01 = _mm512_fmadd_ps(s0, w1, d01);
                        s0 = _mm512_set1_ps(src1[offset0]);
                        d10 = _mm512_fmadd_ps(s0, w0, d10);
                        d11 = _mm512_fmadd_ps(s0, w1, d11);
                        s0 = _mm512_set1_ps(src2[offset0]);
                        d20 = _mm512_fmadd_ps(s0, w0, d20);
                        d21 = _mm512_fmadd_ps(s0, w1, d21);
                        s0 = _mm512_set1_ps(src3[offset0]);
                        d30 = _mm512_fmadd_ps(s0, w0, d30);
                        d31 = _mm512_fmadd_ps(s0, w1, d31);
                        s0 = _mm512_set1_ps(src4[offset0]);
                        d40 = _mm512_fmadd_ps(s0, w0, d40);
                        d41 = _mm512_fmadd_ps(s0, w1, d41);
                        s0 = _mm512_set1_ps(src5[offset0]);
                        d50 = _mm512_fmadd_ps(s0, w0, d50);
                        d51 = _mm512_fmadd_ps(s0, w1, d51);
                        s0 = _mm512_set1_ps(src6[offset0]);
                        d60 = _mm512_fmadd_ps(s0, w0, d60);
                        d61 = _mm512_fmadd_ps(s0, w1, d61);
                        s0 = _mm512_set1_ps(src0[offset7]);
                        d70 = _mm512_fmadd_ps(s0, w0, d70);
                        d71 = _mm512_fmadd_ps(s0, w1, d71);
                        s0 = _mm512_set1_ps(src1[offset7]);
                        d80 = _mm512_fmadd_ps(s0, w0, d80);
                        d81 = _mm512_fmadd_ps(s0, w1, d81);
                        s0 = _mm512_set1_ps(src2[offset7]);
                        d90 = _mm512_fmadd_ps(s0, w0, d90);
                        d91 = _mm512_fmadd_ps(s0, w1, d91);
                        s0 = _mm512_set1_ps(src3[offset7]);
                        da0 = _mm512_fmadd_ps(s0, w0, da0);
                        da1 = _mm512_fmadd_ps(s0, w1, da1);
                        s0 = _mm512_set1_ps(src4[offset7]);
                        db0 = _mm512_fmadd_ps(s0, w0, db0);
                        db1 = _mm512_fmadd_ps(s0, w1, db1);
                        s0 = _mm512_set1_ps(src5[offset7]);
                        dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                        dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        s0 = _mm512_set1_ps(src6[offset7]);
                        dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                        dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d60, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d61, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d70, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d71, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d80, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d81, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d90, bias, params);
                Term<term>::template Save<type, 1>(dst + F, d91, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, da0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, da1, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, db0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, db1, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, dc0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, dc1, bias, params, tails[1]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, dd0, bias, params);
                Term<term>::template Save<type, 1>(dst + F, dd1, bias, params, tails[1]);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                d10 = _mm512_setzero_ps();
                d20 = _mm512_setzero_ps();
                d30 = _mm512_setzero_ps();
                d40 = _mm512_setzero_ps();
                d50 = _mm512_setzero_ps();
                d60 = _mm512_setzero_ps();
                d70 = _mm512_setzero_ps();
                d80 = _mm512_setzero_ps();
                d90 = _mm512_setzero_ps();
                da0 = _mm512_setzero_ps();
                db0 = _mm512_setzero_ps();
                dc0 = _mm512_setzero_ps();
                dd0 = _mm512_setzero_ps();
                for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                {
                    w0 = _mm512_loadu_ps(weight + 0);
                    s0 = _mm512_set1_ps(src0[offset0]);
                    d00 = _mm512_fmadd_ps(s0, w0, d00);
                    s0 = _mm512_set1_ps(src1[offset0]);
                    d10 = _mm512_fmadd_ps(s0, w0, d10);
                    s0 = _mm512_set1_ps(src2[offset0]);
                    d20 = _mm512_fmadd_ps(s0, w0, d20);
                    s0 = _mm512_set1_ps(src3[offset0]);
                    d30 = _mm512_fmadd_ps(s0, w0, d30);
                    s0 = _mm512_set1_ps(src4[offset0]);
                    d40 = _mm512_fmadd_ps(s0, w0, d40);
                    s0 = _mm512_set1_ps(src5[offset0]);
                    d50 = _mm512_fmadd_ps(s0, w0, d50);
                    s0 = _mm512_set1_ps(src6[offset0]);
                    d60 = _mm512_fmadd_ps(s0, w0, d60);
                    s0 = _mm512_set1_ps(src0[offset7]);
                    d70 = _mm512_fmadd_ps(s0, w0, d70);
                    s0 = _mm512_set1_ps(src1[offset7]);
                    d80 = _mm512_fmadd_ps(s0, w0, d80);
                    s0 = _mm512_set1_ps(src2[offset7]);
                    d90 = _mm512_fmadd_ps(s0, w0, d90);
                    s0 = _mm512_set1_ps(src3[offset7]);
                    da0 = _mm512_fmadd_ps(s0, w0, da0);
                    s0 = _mm512_set1_ps(src4[offset7]);
                    db0 = _mm512_fmadd_ps(s0, w0, db0);
                    s0 = _mm512_set1_ps(src5[offset7]);
                    dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                    s0 = _mm512_set1_ps(src6[offset7]);
                    dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                    weight += DF;
                }
                Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d60, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d70, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d80, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, d90, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, da0, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, db0, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, dc0, bias, params, tails[0]);
                dst += dD;
                Term<term>::template Save<type, 0>(dst + 0, dd0, bias, params, tails[0]);
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
            size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            const float* src6 = src0 + 6 * dS;
            if (tails[1])
            {
                if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                if (srcC * 2 * F * sizeof(float) > PREFETCH_SIZE)
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        PrefetchL1(weight + 0);
                        PrefetchL1(weight + F);
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                        if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                        if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                        if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                        if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                        if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                        if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                        if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                        if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                        if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                        if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                        if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                        if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                else
                {
                    for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                    {
                        w0 = _mm512_loadu_ps(weight + 0);
                        w1 = _mm512_loadu_ps(weight + F);
                        if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                        if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                        if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                        if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                        if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                        if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                        if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                        if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                        if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                        if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                        if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
                        if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0), db1 = _mm512_fmadd_ps(s0, w1, db1);
                        if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0), dc1 = _mm512_fmadd_ps(s0, w1, dc1);
                        if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0), dd1 = _mm512_fmadd_ps(s0, w1, dd1);
                        weight += DF;
                    }
                }
                if (M > 0x0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]), dst += dD;
                if (M > 0x1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]), dst += dD;
                if (M > 0x2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]), dst += dD;
                if (M > 0x3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]), dst += dD;
                if (M > 0x4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]), dst += dD;
                if (M > 0x5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]), dst += dD;
                if (M > 0x6) Term<term>::template Save<type, 0>(dst + 0, d60, bias, params), Term<term>::template Save<type, 1>(dst + F, d61, bias, params, tails[1]), dst += dD;
                if (M > 0x7) Term<term>::template Save<type, 0>(dst + 0, d70, bias, params), Term<term>::template Save<type, 1>(dst + F, d71, bias, params, tails[1]), dst += dD;
                if (M > 0x8) Term<term>::template Save<type, 0>(dst + 0, d80, bias, params), Term<term>::template Save<type, 1>(dst + F, d81, bias, params, tails[1]), dst += dD;
                if (M > 0x9) Term<term>::template Save<type, 0>(dst + 0, d90, bias, params), Term<term>::template Save<type, 1>(dst + F, d91, bias, params, tails[1]), dst += dD;
                if (M > 0xa) Term<term>::template Save<type, 0>(dst + 0, da0, bias, params), Term<term>::template Save<type, 1>(dst + F, da1, bias, params, tails[1]), dst += dD;
                if (M > 0xb) Term<term>::template Save<type, 0>(dst + 0, db0, bias, params), Term<term>::template Save<type, 1>(dst + F, db1, bias, params, tails[1]), dst += dD;
                if (M > 0xc) Term<term>::template Save<type, 0>(dst + 0, dc0, bias, params), Term<term>::template Save<type, 1>(dst + F, dc1, bias, params, tails[1]), dst += dD;
                if (M > 0xd) Term<term>::template Save<type, 0>(dst + 0, dd0, bias, params), Term<term>::template Save<type, 1>(dst + F, dd1, bias, params, tails[1]), dst += dD;
            }
            else
            {
                if (M > 0x0) d00 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps();
                for (size_t offset0 = 0, offset7 = 7 * dS; offset0 < srcC; ++offset0, ++offset7)
                {
                    w0 = _mm512_loadu_ps(weight + 0);
                    if (M > 0x0) s0 = _mm512_set1_ps(src0[offset0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
                    if (M > 0x1) s0 = _mm512_set1_ps(src1[offset0]), d10 = _mm512_fmadd_ps(s0, w0, d10);
                    if (M > 0x2) s0 = _mm512_set1_ps(src2[offset0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
                    if (M > 0x3) s0 = _mm512_set1_ps(src3[offset0]), d30 = _mm512_fmadd_ps(s0, w0, d30);
                    if (M > 0x4) s0 = _mm512_set1_ps(src4[offset0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
                    if (M > 0x5) s0 = _mm512_set1_ps(src5[offset0]), d50 = _mm512_fmadd_ps(s0, w0, d50);
                    if (M > 0x6) s0 = _mm512_set1_ps(src6[offset0]), d60 = _mm512_fmadd_ps(s0, w0, d60);
                    if (M > 0x7) s0 = _mm512_set1_ps(src0[offset7]), d70 = _mm512_fmadd_ps(s0, w0, d70);
                    if (M > 0x8) s0 = _mm512_set1_ps(src1[offset7]), d80 = _mm512_fmadd_ps(s0, w0, d80);
                    if (M > 0x9) s0 = _mm512_set1_ps(src2[offset7]), d90 = _mm512_fmadd_ps(s0, w0, d90);
                    if (M > 0xa) s0 = _mm512_set1_ps(src3[offset7]), da0 = _mm512_fmadd_ps(s0, w0, da0);
                    if (M > 0xb) s0 = _mm512_set1_ps(src4[offset7]), db0 = _mm512_fmadd_ps(s0, w0, db0);
                    if (M > 0xc) s0 = _mm512_set1_ps(src5[offset7]), dc0 = _mm512_fmadd_ps(s0, w0, dc0);
                    if (M > 0xd) s0 = _mm512_set1_ps(src6[offset7]), dd0 = _mm512_fmadd_ps(s0, w0, dd0);
                    weight += DF;
                }
                if (M > 0x0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]), dst += dD;
                if (M > 0x1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]), dst += dD;
                if (M > 0x2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]), dst += dD;
                if (M > 0x3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]), dst += dD;
                if (M > 0x4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]), dst += dD;
                if (M > 0x5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]), dst += dD;
                if (M > 0x6) Term<term>::template Save<type, 0>(dst + 0, d60, bias, params, tails[0]), dst += dD;
                if (M > 0x7) Term<term>::template Save<type, 0>(dst + 0, d70, bias, params, tails[0]), dst += dD;
                if (M > 0x8) Term<term>::template Save<type, 0>(dst + 0, d80, bias, params, tails[0]), dst += dD;
                if (M > 0x9) Term<term>::template Save<type, 0>(dst + 0, d90, bias, params, tails[0]), dst += dD;
                if (M > 0xa) Term<term>::template Save<type, 0>(dst + 0, da0, bias, params, tails[0]), dst += dD;
                if (M > 0xb) Term<term>::template Save<type, 0>(dst + 0, db0, bias, params, tails[0]), dst += dD;
                if (M > 0xc) Term<term>::template Save<type, 0>(dst + 0, dc0, bias, params, tails[0]), dst += dD;
                if (M > 0xd) Term<term>::template Save<type, 0>(dst + 0, dd0, bias, params, tails[0]), dst += dD;

            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const float* src0, const ConvParam32f& p, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst, const __mmask16 tails[2]);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x0>;
            case 0x1: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect1x1_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xc>;
            case 0xd: return ConvolutionNhwcDirect1x1_2xM<term, type, 0xd>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t n = 14;
            size_t n1 = (yEnd - yBeg) * p.dstW;
            size_t nn = AlignLoAny(n1, n);
            size_t m = n1 - nn;
            ConvolutionNhwcDirect1x1_2xM_Ptr convN = ConvolutionNhwcDirect1x1_2x14<term, type>;
            ConvolutionNhwcDirect1x1_2xM_Ptr convM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t tail = Simd::Min(DF, dstC - dc);
                __mmask16 tails[2] = { TailMask16(tail), TailMask16(tail - F) };
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convN(ps, p, srcC, weight, _bias, _params, pd, tails);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convM(ps, p, srcC, weight, _bias, _params, pd, tails);
                weight += srcC * DF;
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p,
            const SynetConvolution32fNhwcDirect::AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (a.macroC == p.srcC)
                            ConvolutionNhwcDirect1x1_2<TermSingle, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else if (sc == 0)
                            ConvolutionNhwcDirect1x1_2<TermFirst, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect1x1_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        else
                            ConvolutionNhwcDirect1x1_2<TermIterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroC;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, SynetConvolution32fNhwcDirect::OldConvolutionPtr& convolution)
        {
            if (p.Is1x1())
                convolution = ConvolutionNhwcDirect1x1_2<type>;
            else
                convolution = ConvolutionNhwcDirect_2<type>;
        }

        bool SynetConvolution32fNhwcDirect::Set2f(const ConvParam32f& p, OldConvolutionPtr& convolution)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, convolution); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, convolution); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, convolution); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, convolution); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, convolution); break;
            default: assert(0);
            }
            return true;
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
