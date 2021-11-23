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

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst, int first);
        typedef void(*ConvolutionNhwcDirect1x1_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst, int first);

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst, int first)
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
                if (first)
                    d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                else
                    d00 = _mm256_loadu_ps(dst + 0 * F), d01 = _mm256_loadu_ps(dst + 1 * F), d02 = _mm256_loadu_ps(dst + 2 * F);
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
                if (first)
                    d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                else
                    d00 = _mm256_loadu_ps(dst + 0 * F), d01 = _mm256_loadu_ps(dst + 1 * F);
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
                if (first)
                    d00 = _mm256_setzero_ps();
                else
                    d00 = _mm256_loadu_ps(dst + 0 * F);
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

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst, int first)
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm256_loadu_ps(dst + 0 * dD + 1 * F), d02 = _mm256_loadu_ps(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm256_loadu_ps(dst + 1 * dD + 1 * F), d12 = _mm256_loadu_ps(dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm256_loadu_ps(dst + 2 * dD + 1 * F), d22 = _mm256_loadu_ps(dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm256_loadu_ps(dst + 3 * dD + 1 * F), d32 = _mm256_loadu_ps(dst + 3 * dD + 2 * F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm256_loadu_ps(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm256_loadu_ps(dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm256_loadu_ps(dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm256_loadu_ps(dst + 3 * dD + 1 * F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
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
            case 4: return ConvolutionNhwcDirect_3xM<term, type, 4>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 4, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3x1 = ConvolutionNhwcDirect_3x1<term, type>;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xN = GetConvolutionNhwcDirect_3xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_3xM = GetConvolutionNhwcDirect_3xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            __m256 _params[3], _bias[3];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
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
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_3xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_3xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_3x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_3xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const __m256* bias, const __m256* params, float* dst, int first)
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps(), d02 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps(), d12 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps(), d22 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps(), d32 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm256_loadu_ps(dst + 0 * dD + 1 * F), d02 = _mm256_loadu_ps(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm256_loadu_ps(dst + 1 * dD + 1 * F), d12 = _mm256_loadu_ps(dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm256_loadu_ps(dst + 2 * dD + 1 * F), d22 = _mm256_loadu_ps(dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm256_loadu_ps(dst + 3 * dD + 1 * F), d32 = _mm256_loadu_ps(dst + 3 * dD + 2 * F);
                }
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F), d01 = _mm256_loadu_ps(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F), d11 = _mm256_loadu_ps(dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F), d21 = _mm256_loadu_ps(dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F), d31 = _mm256_loadu_ps(dst + 3 * dD + 1 * F);
                }
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
                if (first)
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0 * F);
                }
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
            case 4: return ConvolutionNhwcDirect1x1_3xM<term, type, 4>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_3(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t n = 4, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xN = GetConvolutionNhwcDirect1x1_3xM<term, type>(n);
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_3xM = GetConvolutionNhwcDirect1x1_3xM<term, type>(m);

            __m256 _params[3], _bias[3];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
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
                    convolutionNhwcDirect1x1_3xN(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_3xM(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
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
#endif//SIMD_AVX2_ENABLE
}
