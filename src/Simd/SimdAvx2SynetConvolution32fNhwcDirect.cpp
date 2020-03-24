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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

#ifdef SIMD_SYNET_CONVOLUTION_NHWC_DIRECT_OLD
        namespace Old
        {
            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect_2x6(const float* src0, const ConvParam32f& p,
                size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst)
            {
                __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
                size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
                const float* src1 = src0 + 1 * dS;
                const float* src2 = src0 + 2 * dS;
                const float* src3 = src0 + 3 * dS;
                const float* src4 = src0 + 4 * dS;
                const float* src5 = src0 + 5 * dS;
                if (dstC > F)
                {
                    d00 = _mm256_setzero_ps(); d01 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps(); d11 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps(); d21 = _mm256_setzero_ps();
                    d30 = _mm256_setzero_ps(); d31 = _mm256_setzero_ps();
                    d40 = _mm256_setzero_ps(); d41 = _mm256_setzero_ps();
                    d50 = _mm256_setzero_ps(); d51 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                w1 = _mm256_loadu_ps(weight + F);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                d01 = _mm256_fmadd_ps(s0, w1, d01);
                                s0 = _mm256_set1_ps(src1[offset]);
                                d10 = _mm256_fmadd_ps(s0, w0, d10);
                                d11 = _mm256_fmadd_ps(s0, w1, d11);
                                s0 = _mm256_set1_ps(src2[offset]);
                                d20 = _mm256_fmadd_ps(s0, w0, d20);
                                d21 = _mm256_fmadd_ps(s0, w1, d21);
                                s0 = _mm256_set1_ps(src3[offset]);
                                d30 = _mm256_fmadd_ps(s0, w0, d30);
                                d31 = _mm256_fmadd_ps(s0, w1, d31);
                                s0 = _mm256_set1_ps(src4[offset]);
                                d40 = _mm256_fmadd_ps(s0, w0, d40);
                                d41 = _mm256_fmadd_ps(s0, w1, d41);
                                s0 = _mm256_set1_ps(src5[offset]);
                                d50 = _mm256_fmadd_ps(s0, w0, d50);
                                d51 = _mm256_fmadd_ps(s0, w1, d51);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == DF)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d31, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d41, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d51, bias, params);
                    }
                    else
                    {
                        dstC -= F;
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d31, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d41, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d51, bias, params, dstC);
                    }
                }
                else
                {
                    d00 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps();
                    d30 = _mm256_setzero_ps();
                    d40 = _mm256_setzero_ps();
                    d50 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                s0 = _mm256_set1_ps(src1[offset]);
                                d10 = _mm256_fmadd_ps(s0, w0, d10);
                                s0 = _mm256_set1_ps(src2[offset]);
                                d20 = _mm256_fmadd_ps(s0, w0, d20);
                                s0 = _mm256_set1_ps(src3[offset]);
                                d30 = _mm256_fmadd_ps(s0, w0, d30);
                                s0 = _mm256_set1_ps(src4[offset]);
                                d40 = _mm256_fmadd_ps(s0, w0, d40);
                                s0 = _mm256_set1_ps(src5[offset]);
                                d50 = _mm256_fmadd_ps(s0, w0, d50);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == F)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                    }
                    else
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, dstC);
                    }
                }
            }

            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect_2x3(const float* src0, const ConvParam32f& p,
                size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst)
            {
                __m256 d00, d01, d10, d11, d20, d21, s0, w0, w1;
                size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
                const float* src1 = src0 + 1 * dS;
                const float* src2 = src0 + 2 * dS;
                if (dstC > F)
                {
                    d00 = _mm256_setzero_ps(); d01 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps(); d11 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps(); d21 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                w1 = _mm256_loadu_ps(weight + F);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                d01 = _mm256_fmadd_ps(s0, w1, d01);
                                s0 = _mm256_set1_ps(src1[offset]);
                                d10 = _mm256_fmadd_ps(s0, w0, d10);
                                d11 = _mm256_fmadd_ps(s0, w1, d11);
                                s0 = _mm256_set1_ps(src2[offset]);
                                d20 = _mm256_fmadd_ps(s0, w0, d20);
                                d21 = _mm256_fmadd_ps(s0, w1, d21);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == DF)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
                    }
                    else
                    {
                        dstC -= F;
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params, dstC);
                    }
                }
                else
                {
                    d00 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                s0 = _mm256_set1_ps(src1[offset]);
                                d10 = _mm256_fmadd_ps(s0, w0, d10);
                                s0 = _mm256_set1_ps(src2[offset]);
                                d20 = _mm256_fmadd_ps(s0, w0, d20);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == F)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                    }
                    else
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, dstC);
                    }
                }
            }

            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam32f& p,
                size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst)
            {
                __m256 d00, d01, s0, w0, w1;
                size_t dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC;
                if (dstC > F)
                {
                    d00 = _mm256_setzero_ps();
                    d01 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                w1 = _mm256_loadu_ps(weight + F);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                d01 = _mm256_fmadd_ps(s0, w1, d01);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == DF)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
                    }
                    else
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params, dstC - F);
                    }
                }
                else
                {
                    d00 = _mm256_setzero_ps();
                    for (size_t ky = 0; ky < kernelH; ++ky)
                    {
                        for (size_t kx = 0; kx < kernelW; ++kx)
                        {
                            for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                            {
                                w0 = _mm256_loadu_ps(weight + 0);
                                s0 = _mm256_set1_ps(src0[offset]);
                                d00 = _mm256_fmadd_ps(s0, w0, d00);
                                weight += DF;
                            }
                        }
                        weight += dW;
                    }
                    if (dstC == F)
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                    else
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, dstC);
                }
            }

            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p,
                size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
            {
                size_t noseH = p.padY, noseW = p.padX;
                size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
                size_t bodyW3 = AlignLoAny(bodyW - noseW, 3 * p.strideX) + noseW;
                size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
                size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
                size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

                __m256 _params[2], _bias[2];
                _params[0] = _mm256_set1_ps(params[0]);
                if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                    _params[1] = _mm256_set1_ps(params[1]);

                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm256_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm256_loadu_ps(params + dc + 0);
                        _params[1] = _mm256_loadu_ps(params + dc + F);
                    }
                    float* d = dst + dc + yBeg * p.dstW * p.dstC;
                    size_t dy = yBeg, sy = dy * p.strideY;
                    for (; sy < noseH && dy < yEnd; sy += p.strideY, dy++)
                    {
                        size_t sx = 0;
                        const float* s = src;
                        const float* w = weight + (noseH - sy) * p.kernelX * srcC * DF;
                        for (; sx < noseW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s, p, kY + sy, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d);
                        for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                            ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                            ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < tailW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, kW - sx, srcC, dC, w, _bias, _params, d);
                    }
                    for (; sy < bodyH && dy < yEnd; sy += p.strideY, dy++)
                    {
                        size_t sx = 0;
                        const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                        const float* w = weight;
                        for (; sx < noseW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s, p, p.kernelY, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d);
                        for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                            ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                            ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < tailW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, kW - sx, srcC, dC, w, _bias, _params, d);
                    }
                    for (; sy < tailH && dy < yEnd; sy += p.strideY, dy++)
                    {
                        size_t sx = 0;
                        const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                        const float* w = weight;
                        for (; sx < noseW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s, p, kH - sy, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d);
                        for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                            ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                            ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d);
                        for (; sx < tailW; sx += p.strideX, d += p.dstC)
                            ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, kW - sx, srcC, dC, w, _bias, _params, d);
                    }
                    weight += p.kernelY * p.kernelX * srcC * DF;
                }
            }

            template<SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p,
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
                                ConvolutionNhwcDirect_2<TermFirst, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            else if (sc + macroC == p.srcC)
                                ConvolutionNhwcDirect_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            else
                                ConvolutionNhwcDirect_2<TermIterim, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            yBeg = yEnd;
                        }
                        weight += AlignHiAny(macroD, a.microD) * macroK;
                    }
                    if (type == ::SimdConvolutionActivationPrelu)
                        params += macroD;
                }
            }

            //---------------------------------------------------------------------

            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect1x1_2x6(const float* src0, const ConvParam32f& p,
                size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst)
            {
                __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
                size_t dS = p.srcC, dD = p.dstC;
                const float* src1 = src0 + 1 * dS;
                const float* src2 = src0 + 2 * dS;
                const float* src3 = src0 + 3 * dS;
                const float* src4 = src0 + 4 * dS;
                const float* src5 = src0 + 5 * dS;
                if (dstC > F)
                {
                    d00 = _mm256_setzero_ps(); d01 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps(); d11 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps(); d21 = _mm256_setzero_ps();
                    d30 = _mm256_setzero_ps(); d31 = _mm256_setzero_ps();
                    d40 = _mm256_setzero_ps(); d41 = _mm256_setzero_ps();
                    d50 = _mm256_setzero_ps(); d51 = _mm256_setzero_ps();
                    for (size_t offset = 0; offset < srcC; ++offset)
                    {
                        w0 = _mm256_loadu_ps(weight + 0);
                        w1 = _mm256_loadu_ps(weight + F);
                        s0 = _mm256_set1_ps(src0[offset]);
                        d00 = _mm256_fmadd_ps(s0, w0, d00);
                        d01 = _mm256_fmadd_ps(s0, w1, d01);
                        s0 = _mm256_set1_ps(src1[offset]);
                        d10 = _mm256_fmadd_ps(s0, w0, d10);
                        d11 = _mm256_fmadd_ps(s0, w1, d11);
                        s0 = _mm256_set1_ps(src2[offset]);
                        d20 = _mm256_fmadd_ps(s0, w0, d20);
                        d21 = _mm256_fmadd_ps(s0, w1, d21);
                        s0 = _mm256_set1_ps(src3[offset]);
                        d30 = _mm256_fmadd_ps(s0, w0, d30);
                        d31 = _mm256_fmadd_ps(s0, w1, d31);
                        s0 = _mm256_set1_ps(src4[offset]);
                        d40 = _mm256_fmadd_ps(s0, w0, d40);
                        d41 = _mm256_fmadd_ps(s0, w1, d41);
                        s0 = _mm256_set1_ps(src5[offset]);
                        d50 = _mm256_fmadd_ps(s0, w0, d50);
                        d51 = _mm256_fmadd_ps(s0, w1, d51);
                        weight += DF;
                    }
                    if (dstC == DF)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d31, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d41, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d51, bias, params);
                    }
                    else
                    {
                        dstC -= F;
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d01, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d11, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d21, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d31, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d41, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                        Term<term>::template Save<type, 1>(dst + F, d51, bias, params, dstC);
                    }
                }
                else
                {
                    d00 = _mm256_setzero_ps();
                    d10 = _mm256_setzero_ps();
                    d20 = _mm256_setzero_ps();
                    d30 = _mm256_setzero_ps();
                    d40 = _mm256_setzero_ps();
                    d50 = _mm256_setzero_ps();
                    for (size_t offset = 0; offset < srcC; ++offset)
                    {
                        w0 = _mm256_loadu_ps(weight + 0);
                        s0 = _mm256_set1_ps(src0[offset]);
                        d00 = _mm256_fmadd_ps(s0, w0, d00);
                        s0 = _mm256_set1_ps(src1[offset]);
                        d10 = _mm256_fmadd_ps(s0, w0, d10);
                        s0 = _mm256_set1_ps(src2[offset]);
                        d20 = _mm256_fmadd_ps(s0, w0, d20);
                        s0 = _mm256_set1_ps(src3[offset]);
                        d30 = _mm256_fmadd_ps(s0, w0, d30);
                        s0 = _mm256_set1_ps(src4[offset]);
                        d40 = _mm256_fmadd_ps(s0, w0, d40);
                        s0 = _mm256_set1_ps(src5[offset]);
                        d50 = _mm256_fmadd_ps(s0, w0, d50);
                        weight += DF;
                    }
                    if (dstC == F)
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
                    }
                    else
                    {
                        Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, dstC);
                        dst += dD;
                        Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, dstC);
                    }
                }
            }

            template<TermType term, SimdConvolutionActivationType type, int M> SIMD_NOINLINE void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
                size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst)
            {
                __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
                size_t dS = p.srcC, dD = p.dstC;
                const float* src1 = src0 + 1 * dS;
                const float* src2 = src0 + 2 * dS;
                const float* src3 = src0 + 3 * dS;
                const float* src4 = src0 + 4 * dS;
                const float* src5 = src0 + 5 * dS;
                if (dstC > F)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                    if (M > 5) d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
                    for (size_t offset = 0; offset < srcC; ++offset)
                    {
                        w0 = _mm256_loadu_ps(weight + 0);
                        w1 = _mm256_loadu_ps(weight + F);
                        if (M > 0) s0 = _mm256_set1_ps(src0[offset]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                        if (M > 1) s0 = _mm256_set1_ps(src1[offset]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                        if (M > 2) s0 = _mm256_set1_ps(src2[offset]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                        if (M > 3) s0 = _mm256_set1_ps(src3[offset]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                        if (M > 4) s0 = _mm256_set1_ps(src4[offset]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                        if (M > 5) s0 = _mm256_set1_ps(src5[offset]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
                        weight += DF;
                    }
                    if (dstC == DF)
                    {
                        if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params), dst += dD;
                        if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params), dst += dD;
                        if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params), dst += dD;
                        if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params), dst += dD;
                        if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params), dst += dD;
                        if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params), dst += dD;
                    }
                    else
                    {
                        dstC -= F;
                        if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params, dstC), dst += dD;
                        if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params, dstC), dst += dD;
                        if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params, dstC), dst += dD;
                        if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params, dstC), dst += dD;
                        if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params, dstC), dst += dD;
                        if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params, dstC), dst += dD;
                    }
                }
                else
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps();
                    if (M > 5) d50 = _mm256_setzero_ps();
                    for (size_t offset = 0; offset < srcC; ++offset)
                    {
                        w0 = _mm256_loadu_ps(weight + 0);
                        if (M > 0) s0 = _mm256_set1_ps(src0[offset]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                        if (M > 1) s0 = _mm256_set1_ps(src1[offset]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                        if (M > 2) s0 = _mm256_set1_ps(src2[offset]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                        if (M > 3) s0 = _mm256_set1_ps(src3[offset]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                        if (M > 4) s0 = _mm256_set1_ps(src4[offset]), d40 = _mm256_fmadd_ps(s0, w0, d40);
                        if (M > 5) s0 = _mm256_set1_ps(src5[offset]), d50 = _mm256_fmadd_ps(s0, w0, d50);
                        weight += DF;
                    }
                    if (dstC == F)
                    {
                        if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), dst += dD;
                        if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), dst += dD;
                        if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), dst += dD;
                        if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), dst += dD;
                        if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), dst += dD;
                        if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), dst += dD;
                    }
                    else
                    {
                        if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, dstC), dst += dD;
                        if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, dstC), dst += dD;
                        if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, dstC), dst += dD;
                        if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, dstC), dst += dD;
                        if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, dstC), dst += dD;
                        if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, dstC), dst += dD;
                    }
                }
            }

            typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const float* src0, const ConvParam32f& p, size_t srcC, size_t dstC, const float* weight, const __m256* bias, const __m256* params, float* dst);

            template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
            {
                switch (M)
                {
                case 0: return ConvolutionNhwcDirect1x1_2xM<term, type, 0>;
                case 1: return ConvolutionNhwcDirect1x1_2xM<term, type, 1>;
                case 2: return ConvolutionNhwcDirect1x1_2xM<term, type, 2>;
                case 3: return ConvolutionNhwcDirect1x1_2xM<term, type, 3>;
                case 4: return ConvolutionNhwcDirect1x1_2xM<term, type, 4>;
                case 5: return ConvolutionNhwcDirect1x1_2xM<term, type, 5>;
                }
                assert(0);
                return NULL;
            }

            template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p,
                size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
            {
                size_t n1 = (yEnd - yBeg) * p.dstW;
                size_t n6 = AlignLoAny(n1, 6);
                size_t nTail = n1 - n6;
                ConvolutionNhwcDirect1x1_2xM_Ptr tailN = GetConvolutionNhwcDirect1x1_2xM<term, type>(nTail);

                __m256 _params[2], _bias[2];
                _params[0] = _mm256_set1_ps(params[0]);
                if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                    _params[1] = _mm256_set1_ps(params[1]);

                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm256_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm256_loadu_ps(params + dc + 0);
                        _params[1] = _mm256_loadu_ps(params + dc + F);
                    }
                    const float* ps = src + yBeg * p.srcW * p.srcC;
                    float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                    size_t i = 0;
                    for (; i < n6; i += 6, ps += 6 * p.srcC, pd += 6 * p.dstC)
                        ConvolutionNhwcDirect1x1_2x6<term, type>(ps, p, srcC, dC, weight, _bias, _params, pd);
                    if (nTail)
                        tailN(ps, p, srcC, dC, weight, _bias, _params, pd), ps += nTail * p.srcC, pd += nTail * p.dstC;
                    weight += srcC * DF;
                }
            }

            template<SimdConvolutionActivationType type> SIMD_NOINLINE void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p,
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
                                ConvolutionNhwcDirect1x1_2<TermFirst, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            else if (sc + macroC == p.srcC)
                                ConvolutionNhwcDirect1x1_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            else
                                ConvolutionNhwcDirect1x1_2<TermIterim, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc);
                            yBeg = yEnd;
                        }
                        weight += AlignHiAny(macroD, a.microD) * macroC;
                    }
                    if (type == ::SimdConvolutionActivationPrelu)
                        params += macroD;
                }
            }

            //---------------------------------------------------------------------

            template <SimdConvolutionActivationType type> void Set(const ConvParam32f& p, SynetConvolution32fNhwcDirect::OldConvolutionPtr& convolution)
            {
                if (p.Is1x1())
                    convolution = ConvolutionNhwcDirect1x1_2<type>;
                else
                    convolution = ConvolutionNhwcDirect_2<type>;
            }

            bool Set(const ConvParam32f& p, SynetConvolution32fNhwcDirect::OldConvolutionPtr& convolution)
            {
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, convolution); break;
                case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, convolution); break;
                case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(p, convolution); break;
                case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, convolution); break;
                case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, convolution); break;
                case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, convolution); break;
                case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, convolution); break;
                default: assert(0);
                }
                return true;
            }
        }
#endif

        //---------------------------------------------------------------------

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst);
        typedef void(*ConvolutionNhwcDirect1x1_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst);

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
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
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                            }
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                if (dstC == DF)
                    Save2<term, type>(dst, d00, d01, bias, params);
                else
                    Save2<term, type>(dst, d00, d01, bias, params, dstC - F);
            }
            else
            {
                d00 = _mm256_setzero_ps();
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
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                            }
                        }
                        weight0 += dW;
                    }
                }
                if (dstC == F)
                    Save1<term, type>(dst, d00, bias, params);
                else
                    Save1<term, type>(dst, d00, bias, params, dstC);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x6(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 6 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                                s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                                s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                                s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                                s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                                s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                d00 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps();
                d40 = _mm256_setzero_ps();
                d50 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 6 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                                s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                                s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                                s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                                s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40);
                                s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    Save1<term, type>(dst, d00, bias, params), dst += dD;
                    Save1<term, type>(dst, d10, bias, params), dst += dD;
                    Save1<term, type>(dst, d20, bias, params), dst += dD;
                    Save1<term, type>(dst, d30, bias, params), dst += dD;
                    Save1<term, type>(dst, d40, bias, params), dst += dD;
                    Save1<term, type>(dst, d50, bias, params), dst += dD;
                }
                else
                {
                    Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                if (M > 5) d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 6 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                                if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                                if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                                if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                                if (M > 4) s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                                if (M > 5) s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps();
                if (M > 5) d50 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 6 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                                if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                                if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                                if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                                if (M > 4) s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40);
                                if (M > 5) s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                    if (M > 5) Save1<term, type>(dst, d50, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    if (M > 5) Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 6, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2x1 = ConvolutionNhwcDirect_2x1<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xN = ConvolutionNhwcDirect_2x6<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m256 _params[2], _bias[2];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = _mm256_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm256_loadu_ps(bias + dc + 1 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm256_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm256_loadu_ps(params + dc + 1 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2x6(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                    s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                    s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                    s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                    s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                    s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
                }
                if (dstC == DF)
                {
                    Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                d00 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps();
                d40 = _mm256_setzero_ps();
                d50 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                    s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                    s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                    s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                    s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40);
                    s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50);
                }
                if (dstC == F)
                {
                    Save1<term, type>(dst, d00, bias, params), dst += dD;
                    Save1<term, type>(dst, d10, bias, params), dst += dD;
                    Save1<term, type>(dst, d20, bias, params), dst += dD;
                    Save1<term, type>(dst, d30, bias, params), dst += dD;
                    Save1<term, type>(dst, d40, bias, params), dst += dD;
                    Save1<term, type>(dst, d50, bias, params), dst += dD;
                }
                else
                {
                    Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                if (M > 5) d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                    if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                    if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                    if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                    if (M > 4) s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                    if (M > 5) s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps();
                if (M > 4) d40 = _mm256_setzero_ps();
                if (M > 5) d50 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                    if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                    if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                    if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                    if (M > 4) s0 = _mm256_set1_ps(src4[offs]), d40 = _mm256_fmadd_ps(s0, w0, d40);
                    if (M > 5) s0 = _mm256_set1_ps(src5[offs]), d50 = _mm256_fmadd_ps(s0, w0, d50);
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                    if (M > 5) Save1<term, type>(dst, d50, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    if (M > 5) Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_NxM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 6, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xN = ConvolutionNhwcDirect1x1_2x6<term, type>;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);

            __m256 _params[2], _bias[2];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = _mm256_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm256_loadu_ps(bias + dc + 1 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm256_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm256_loadu_ps(params + dc + 1 * F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_2xN(ps, p, a, srcC, dC, weight, _bias, _params, pd);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_2xM(ps, p, a, srcC, dC, weight, _bias, _params, pd);
                weight += srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d02, s0, w0, w1, w2;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            if (dstC > 2 * F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
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
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                w2 = _mm256_loadu_ps(weight2 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01), d02 = _mm256_fmadd_ps(s0, w2, d02);
                            }
                        }
                        weight0 += dW, weight1 += dW, weight2 += dW;
                    }
                }
                if (dstC == 3 * F)
                    Save3<term, type>(dst, d00, d01, d02, bias, params);
                else
                    Save3<term, type>(dst, d00, d01, d02, bias, params, dstC - 2 * F);
            }
            else if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
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
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                            }
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                if (dstC == 2 * F)
                    Save2<term, type>(dst, d00, d01, bias, params);
                else
                    Save2<term, type>(dst, d00, d01, bias, params, dstC - F);
            }
            else
            {
                d00 = _mm256_setzero_ps();
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
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                            }
                        }
                        weight0 += dW;
                    }
                }
                if (dstC == F)
                    Save1<term, type>(dst, d00, bias, params);
                else
                    Save1<term, type>(dst, d00, bias, params, dstC);
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3x4(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, s0, w0, w1, w2;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            if (dstC > 2 * F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                w2 = _mm256_loadu_ps(weight2 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01), d02 = _mm256_fmadd_ps(s0, w2, d02);
                                s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11), d12 = _mm256_fmadd_ps(s0, w2, d12);
                                s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21), d22 = _mm256_fmadd_ps(s0, w2, d22);
                                s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31), d32 = _mm256_fmadd_ps(s0, w2, d32);
                            }
                            weight0 += dW, weight1 += dW, weight2 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz, weight2 += dWz;
                }
                if (dstC == 3 * F)
                {
                    Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                    Save3<term, type>(dst, d20, d21, d22, bias, params), dst += dD;
                    Save3<term, type>(dst, d30, d31, d32, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                                s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                                s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                                s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == 2 * F)
                {
                    Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 1 * F;
                    Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                d00 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                                s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                                s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                                s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    Save1<term, type>(dst, d00, bias, params), dst += dD;
                    Save1<term, type>(dst, d10, bias, params), dst += dD;
                    Save1<term, type>(dst, d20, bias, params), dst += dD;
                    Save1<term, type>(dst, d30, bias, params), dst += dD;
                }
                else
                {
                    Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, s0, w0, w1, w2;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            if (dstC > 2 * F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                w2 = _mm256_loadu_ps(weight2 + offw);
                                if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01), d02 = _mm256_fmadd_ps(s0, w2, d02);
                                if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11), d12 = _mm256_fmadd_ps(s0, w2, d12);
                                if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21), d22 = _mm256_fmadd_ps(s0, w2, d22);
                                if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31), d32 = _mm256_fmadd_ps(s0, w2, d32);
                            }
                            weight0 += dW, weight1 += dW, weight2 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz, weight2 += dWz;
                }
                if (dstC == 3 * F)
                {
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                w1 = _mm256_loadu_ps(weight1 + offw);
                                if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                                if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                                if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                                if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == 2 * F)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 1 * F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + 4 <= srcW);
                            size_t offs = beg + kx * dX, end = offs + srcC, offw = 0;
                            for (; offs < end; ++offs, offw += F)
                            {
                                w0 = _mm256_loadu_ps(weight0 + offw);
                                if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                                if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                                if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                                if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                }
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
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 4, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3x1 = ConvolutionNhwcDirect_3x1<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xN = ConvolutionNhwcDirect_3x4<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xM = GetConvolutionNhwcDirect_3xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m256 _params[3], _bias[3];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = _mm256_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm256_loadu_ps(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = _mm256_loadu_ps(bias + dc + 2 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm256_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm256_loadu_ps(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = _mm256_loadu_ps(params + dc + 2 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_3xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_3xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_3xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_3xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_3xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_3xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_3x4(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, s0, w0, w1, w2;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            if (dstC > 2 * F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    w2 = _mm256_loadu_ps(weight2 + offw);
                    s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01), d02 = _mm256_fmadd_ps(s0, w2, d02);
                    s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11), d12 = _mm256_fmadd_ps(s0, w2, d12);
                    s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21), d22 = _mm256_fmadd_ps(s0, w2, d22);
                    s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31), d32 = _mm256_fmadd_ps(s0, w2, d32);
                }
                if (dstC == 3 * F)
                {
                    Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                    Save3<term, type>(dst, d20, d21, d22, bias, params), dst += dD;
                    Save3<term, type>(dst, d30, d31, d32, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                    s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                    s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                    s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                }
                if (dstC == 2 * F)
                {
                    Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 1 * F;
                    Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                d00 = _mm256_setzero_ps();
                d10 = _mm256_setzero_ps();
                d20 = _mm256_setzero_ps();
                d30 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                    s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                    s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                    s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                }
                if (dstC == F)
                {
                    Save1<term, type>(dst, d00, bias, params), dst += dD;
                    Save1<term, type>(dst, d10, bias, params), dst += dD;
                    Save1<term, type>(dst, d20, bias, params), dst += dD;
                    Save1<term, type>(dst, d30, bias, params), dst += dD;
                }
                else
                {
                    Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32, s0, w0, w1, w2;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            if (dstC > 2 * F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    w2 = _mm256_loadu_ps(weight2 + offw);
                    if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01), d02 = _mm256_fmadd_ps(s0, w2, d02);
                    if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11), d12 = _mm256_fmadd_ps(s0, w2, d12);
                    if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21), d22 = _mm256_fmadd_ps(s0, w2, d22);
                    if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31), d32 = _mm256_fmadd_ps(s0, w2, d32);
                }
                if (dstC == 3 * F)
                {
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    w1 = _mm256_loadu_ps(weight1 + offw);
                    if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                    if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                    if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                    if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (M > 0) d00 = _mm256_setzero_ps();
                if (M > 1) d10 = _mm256_setzero_ps();
                if (M > 2) d20 = _mm256_setzero_ps();
                if (M > 3) d30 = _mm256_setzero_ps();
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = _mm256_loadu_ps(weight0 + offw);
                    if (M > 0) s0 = _mm256_set1_ps(src0[offs]), d00 = _mm256_fmadd_ps(s0, w0, d00);
                    if (M > 1) s0 = _mm256_set1_ps(src1[offs]), d10 = _mm256_fmadd_ps(s0, w0, d10);
                    if (M > 2) s0 = _mm256_set1_ps(src2[offs]), d20 = _mm256_fmadd_ps(s0, w0, d20);
                    if (M > 3) s0 = _mm256_set1_ps(src3[offs]), d30 = _mm256_fmadd_ps(s0, w0, d30);
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                }
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
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 4, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xN = ConvolutionNhwcDirect1x1_3x4<term, type>;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xM = GetConvolutionNhwcDirect1x1_3xM<term, type>(m);

            __m256 _params[3], _bias[3];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = _mm256_loadu_ps(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = _mm256_loadu_ps(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = _mm256_loadu_ps(bias + dc + 2 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = _mm256_loadu_ps(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = _mm256_loadu_ps(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = _mm256_loadu_ps(params + dc + 2 * F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_3xN(ps, p, a, srcC, dC, weight, _bias, _params, pd);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_3xM(ps, p, a, srcC, dC, weight, _bias, _params, pd);
                weight += srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template <TermType term, SimdConvolutionActivationType type> void Set(const ConvParam32f& p, AlgParam& a)
        {
            switch (a.microD)
            {
            case 2 * F: a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_2<term, type> : ConvolutionNhwcDirect_2<term, type>; break;
            case 3 * F: a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_3<term, type> : ConvolutionNhwcDirect_3<term, type>; break;
            default: assert(0);
            }
        }

        template <SimdConvolutionActivationType type> void Set(const ConvParam32f& p, AlgParam& a)
        {
            Set<TermSingle, type>(p, a);
            Set<TermFirst, type>(p, a);
            Set<TermIterim, type>(p, a);
            Set<TermLast, type>(p, a);
        }

        bool Set(const ConvParam32f& p, AlgParam& a)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, a); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, a); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(p, a); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, a); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, a); break;
            default: return false;
            }
            return true;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f& p)
            : Avx::SynetConvolution32fNhwcDirect(p)
        {
#ifdef SIMD_SYNET_CONVOLUTION_NHWC_DIRECT_OLD
            //_old.enable = false;
            if (_old.enable)
            {
                if (Avx2::Old::Set(p, _old.convolution))
                    OldSetAlgParam(F);
            }
            else
#endif
            {
                RunFuncs funcs;
                for (size_t n = 2; n <= 3; ++n)
                {
                    funcs.push_back(RunFunc(Ext() + "-" + ToStr(n)));
                    SetAlgParam(F, n, funcs.back().alg);
                    if (!Set(p, funcs.back().alg))
                        return;
                }
                _run.Init(funcs);
            }
        }
    }
#endif//SIMD_AVX2_ENABLE
}
