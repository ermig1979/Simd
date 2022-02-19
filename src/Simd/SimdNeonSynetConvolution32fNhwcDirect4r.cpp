/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Neon
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first);
        typedef void(*ConvolutionNhwcDirect1x1_NxM_Ptr)(const float* src0, const ConvParam32f& p, const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first);

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_4x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d02, d03, s0, w0, w1, w2, w3;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* weight3 = weight2 + a.stepW;
            if (dstC > 3 * F)
            {
                if (first)
                    d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f), d03 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0 * F), d01 = Load<false>(dst + 1 * F), d02 = Load<false>(dst + 2 * F), d03 = Load<false>(dst + 3 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                w3 = Load<false>(weight3 + offw);
                                s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2), d03 = vmlaq_f32(d03, s0, w3);
                            }
                        }
                        weight0 += dW, weight1 += dW, weight2 += dW, weight3 += dW;
                    }
                }
                if (dstC == 4 * F)
                    Save4<term, type>(dst, d00, d01, d02, d03, bias, params);
                else
                    Save4<term, type>(dst, d00, d01, d02, d03, bias, params, dstC - 3 * F);
            }
            else if (dstC > 2 * F)
            {
                if (first)
                    d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0 * F), d01 = Load<false>(dst + 1 * F), d02 = Load<false>(dst + 2 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2);
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
                    d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0 * F), d01 = Load<false>(dst + 1 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
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
                    d00 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0);
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

#if defined(SIMD_ARM64_ENABLE)
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_4xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, d40, d41, d42, d43, d50, d51, d52, d53, s0, w0, w1, w2, w3;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* weight3 = weight2 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > 3 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f), d03 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f), d13 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f), d22 = vdupq_n_f32(0.0f), d23 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f), d32 = vdupq_n_f32(0.0f), d33 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f), d42 = vdupq_n_f32(0.0f), d43 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f), d52 = vdupq_n_f32(0.0f), d53 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F), d03 = Load<false>(dst + 0 * dD + 3 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F), d13 = Load<false>(dst + 1 * dD + 3 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F), d22 = Load<false>(dst + 2 * dD + 2 * F), d23 = Load<false>(dst + 2 * dD + 3 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F), d32 = Load<false>(dst + 3 * dD + 2 * F), d33 = Load<false>(dst + 3 * dD + 3 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F), d42 = Load<false>(dst + 4 * dD + 2 * F), d43 = Load<false>(dst + 4 * dD + 3 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F), d52 = Load<false>(dst + 5 * dD + 2 * F), d53 = Load<false>(dst + 5 * dD + 3 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                w3 = Load<false>(weight3 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2), d03 = vmlaq_f32(d03, s0, w3);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2), d13 = vmlaq_f32(d13, s0, w3);
                                if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1), d22 = vmlaq_f32(d22, s0, w2), d23 = vmlaq_f32(d23, s0, w3);
                                if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1), d32 = vmlaq_f32(d32, s0, w2), d33 = vmlaq_f32(d33, s0, w3);
                                if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1), d42 = vmlaq_f32(d42, s0, w2), d43 = vmlaq_f32(d43, s0, w3);
                                if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1), d52 = vmlaq_f32(d52, s0, w2), d53 = vmlaq_f32(d53, s0, w3);
                            }
                            weight0 += dW, weight1 += dW, weight2 += dW, weight3 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz, weight2 += dWz, weight3 += dWz;
                }
                if (dstC == 4 * F)
                {
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params), dst += dD;
                    if (M > 2) Save4<term, type>(dst, d20, d21, d22, d23, bias, params), dst += dD;
                    if (M > 3) Save4<term, type>(dst, d30, d31, d32, d33, bias, params), dst += dD;
                    if (M > 4) Save4<term, type>(dst, d40, d41, d42, d43, bias, params), dst += dD;
                    if (M > 5) Save4<term, type>(dst, d50, d51, d52, d53, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 3 * F;
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params, dstC), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params, dstC), dst += dD;
                    if (M > 2) Save4<term, type>(dst, d20, d21, d22, d23, bias, params, dstC), dst += dD;
                    if (M > 3) Save4<term, type>(dst, d30, d31, d32, d33, bias, params, dstC), dst += dD;
                    if (M > 4) Save4<term, type>(dst, d40, d41, d42, d43, bias, params, dstC), dst += dD;
                    if (M > 5) Save4<term, type>(dst, d50, d51, d52, d53, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > 2 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f), d22 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f), d32 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f), d42 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f), d52 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F), d22 = Load<false>(dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F), d32 = Load<false>(dst + 3 * dD + 2 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F), d42 = Load<false>(dst + 4 * dD + 2 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F), d52 = Load<false>(dst + 5 * dD + 2 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2);
                                if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1), d22 = vmlaq_f32(d22, s0, w2);
                                if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1), d32 = vmlaq_f32(d32, s0, w2);
                                if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1), d42 = vmlaq_f32(d42, s0, w2);
                                if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1), d52 = vmlaq_f32(d52, s0, w2);
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
                    if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params), dst += dD;
                    if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                    if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params, dstC), dst += dD;
                    if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                                if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                                if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                                if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                                if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
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
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0);
                                if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0);
                                if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0);
                                if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0);
                                if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_4xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_4xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect_4xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect_4xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect_4xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect_4xM<term, type, 5>;
            case 6: return ConvolutionNhwcDirect_4xM<term, type, 6>;
            }
            assert(0);
            return NULL;
        }
#else
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_4xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d02, d03, d10, d11, d12, d13, s0, w0, w1, w2, w3;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F, dWz = p.kernelX * p.srcC * F, dD = p.dstC;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* weight3 = weight2 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            if (dstC > 3 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f), d03 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f), d13 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F), d03 = Load<false>(dst + 0 * dD + 3 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F), d13 = Load<false>(dst + 1 * dD + 3 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                w3 = Load<false>(weight3 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2), d03 = vmlaq_f32(d03, s0, w3);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2), d13 = vmlaq_f32(d13, s0, w3);
                            }
                            weight0 += dW, weight1 += dW, weight2 += dW, weight3 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz, weight2 += dWz, weight3 += dWz;
                }
                if (dstC == 4 * F)
                {
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 3 * F;
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params, dstC), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > 2 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                w2 = Load<false>(weight2 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2);
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
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
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
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F);
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
                                w0 = Load<false>(weight0 + offw);
                                if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0);
                                if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0);
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
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_4xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_4xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect_4xM<term, type, 2>;
            }
            assert(0);
            return NULL;
        }
#endif

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_4(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_4x1 = ConvolutionNhwcDirect_4x1<term, type>;
#if defined(SIMD_ARM64_ENABLE)
            size_t n = 6, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
#else
            size_t n = 2, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
#endif
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_4xN = GetConvolutionNhwcDirect_4xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_4xM = GetConvolutionNhwcDirect_4xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            float32x4_t _params[4], _bias[4];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = Load<false>(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = Load<false>(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = Load<false>(bias + dc + 2 * F);
                if (dC > 3 * F) _bias[3] = Load<false>(bias + dc + 3 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = Load<false>(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = Load<false>(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = Load<false>(params + dc + 2 * F);
                    if (dC > 3 * F) _params[3] = Load<false>(params + dc + 3 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_4x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_4xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_4xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_4x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

#if defined(SIMD_ARM64_ENABLE)
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_4xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, d40, d41, d42, d43, d50, d51, d52, d53, s0, w0, w1, w2, w3;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* weight3 = weight2 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > 3 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f), d03 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f), d13 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f), d22 = vdupq_n_f32(0.0f), d23 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f), d32 = vdupq_n_f32(0.0f), d33 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f), d42 = vdupq_n_f32(0.0f), d43 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f), d52 = vdupq_n_f32(0.0f), d53 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F), d03 = Load<false>(dst + 0 * dD + 3 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F), d13 = Load<false>(dst + 1 * dD + 3 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F), d22 = Load<false>(dst + 2 * dD + 2 * F), d23 = Load<false>(dst + 2 * dD + 3 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F), d32 = Load<false>(dst + 3 * dD + 2 * F), d33 = Load<false>(dst + 3 * dD + 3 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F), d42 = Load<false>(dst + 4 * dD + 2 * F), d43 = Load<false>(dst + 4 * dD + 3 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F), d52 = Load<false>(dst + 5 * dD + 2 * F), d53 = Load<false>(dst + 5 * dD + 3 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    w2 = Load<false>(weight2 + offw);
                    w3 = Load<false>(weight3 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2), d03 = vmlaq_f32(d03, s0, w3);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2), d13 = vmlaq_f32(d13, s0, w3);
                    if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1), d22 = vmlaq_f32(d22, s0, w2), d23 = vmlaq_f32(d23, s0, w3);
                    if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1), d32 = vmlaq_f32(d32, s0, w2), d33 = vmlaq_f32(d33, s0, w3);
                    if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1), d42 = vmlaq_f32(d42, s0, w2), d43 = vmlaq_f32(d43, s0, w3);
                    if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1), d52 = vmlaq_f32(d52, s0, w2), d53 = vmlaq_f32(d53, s0, w3);
                }
                if (dstC == 4 * F)
                {
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params), dst += dD;
                    if (M > 2) Save4<term, type>(dst, d20, d21, d22, d23, bias, params), dst += dD;
                    if (M > 3) Save4<term, type>(dst, d30, d31, d32, d33, bias, params), dst += dD;
                    if (M > 4) Save4<term, type>(dst, d40, d41, d42, d43, bias, params), dst += dD;
                    if (M > 5) Save4<term, type>(dst, d50, d51, d52, d53, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 3 * F;
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params, dstC), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params, dstC), dst += dD;
                    if (M > 2) Save4<term, type>(dst, d20, d21, d22, d23, bias, params, dstC), dst += dD;
                    if (M > 3) Save4<term, type>(dst, d30, d31, d32, d33, bias, params, dstC), dst += dD;
                    if (M > 4) Save4<term, type>(dst, d40, d41, d42, d43, bias, params, dstC), dst += dD;
                    if (M > 5) Save4<term, type>(dst, d50, d51, d52, d53, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > 2 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f), d22 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f), d32 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f), d42 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f), d52 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F), d22 = Load<false>(dst + 2 * dD + 2 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F), d32 = Load<false>(dst + 3 * dD + 2 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F), d42 = Load<false>(dst + 4 * dD + 2 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F), d52 = Load<false>(dst + 5 * dD + 2 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    w2 = Load<false>(weight2 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2);
                    if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1), d22 = vmlaq_f32(d22, s0, w2);
                    if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1), d32 = vmlaq_f32(d32, s0, w2);
                    if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1), d42 = vmlaq_f32(d42, s0, w2);
                    if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1), d52 = vmlaq_f32(d52, s0, w2);
                }
                if (dstC == 3 * F)
                {
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params), dst += dD;
                    if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params), dst += dD;
                    if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                    if (M > 2) Save3<term, type>(dst, d20, d21, d22, bias, params, dstC), dst += dD;
                    if (M > 3) Save3<term, type>(dst, d30, d31, d32, bias, params, dstC), dst += dD;
                    if (M > 4) Save3<term, type>(dst, d40, d41, d42, bias, params, dstC), dst += dD;
                    if (M > 5) Save3<term, type>(dst, d50, d51, d52, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F), d21 = Load<false>(dst + 2 * dD + 1 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F), d31 = Load<false>(dst + 3 * dD + 1 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F), d41 = Load<false>(dst + 4 * dD + 1 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F), d51 = Load<false>(dst + 5 * dD + 1 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                    if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                    if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                    if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                    if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
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
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f);
                    if (M > 5) d50 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0 * F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0 * F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0 * F);
                    if (M > 5) d50 = Load<false>(dst + 5 * dD + 0 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0);
                    if (M > 2) s0 = vdupq_n_f32(src2[offs]), d20 = vmlaq_f32(d20, s0, w0);
                    if (M > 3) s0 = vdupq_n_f32(src3[offs]), d30 = vmlaq_f32(d30, s0, w0);
                    if (M > 4) s0 = vdupq_n_f32(src4[offs]), d40 = vmlaq_f32(d40, s0, w0);
                    if (M > 5) s0 = vdupq_n_f32(src5[offs]), d50 = vmlaq_f32(d50, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_NxM_Ptr GetConvolutionNhwcDirect1x1_4xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_4xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect1x1_4xM<term, type, 2>;
            case 3: return ConvolutionNhwcDirect1x1_4xM<term, type, 3>;
            case 4: return ConvolutionNhwcDirect1x1_4xM<term, type, 4>;
            case 5: return ConvolutionNhwcDirect1x1_4xM<term, type, 5>;
            case 6: return ConvolutionNhwcDirect1x1_4xM<term, type, 6>;
            }
            assert(0);
            return NULL;
        }
#else
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_4xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d02, d03, d10, d11, d12, d13, s0, w0, w1, w2, w3;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* weight2 = weight1 + a.stepW;
            const float* weight3 = weight2 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            if (dstC > 3 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f), d03 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f), d13 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F), d03 = Load<false>(dst + 0 * dD + 3 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F), d13 = Load<false>(dst + 1 * dD + 3 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    w2 = Load<false>(weight2 + offw);
                    w3 = Load<false>(weight3 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2), d03 = vmlaq_f32(d03, s0, w3);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2), d13 = vmlaq_f32(d13, s0, w3);
                }
                if (dstC == 4 * F)
                {
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 3 * F;
                    if (M > 0) Save4<term, type>(dst, d00, d01, d02, d03, bias, params, dstC), dst += dD;
                    if (M > 1) Save4<term, type>(dst, d10, d11, d12, d13, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > 2 * F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f), d02 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f), d12 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F), d02 = Load<false>(dst + 0 * dD + 2 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F), d12 = Load<false>(dst + 1 * dD + 2 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    w2 = Load<false>(weight2 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1), d02 = vmlaq_f32(d02, s0, w2);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1), d12 = vmlaq_f32(d12, s0, w2);
                }
                if (dstC == 3 * F)
                {
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params), dst += dD;
                }
                else
                {
                    dstC -= 2 * F;
                    if (M > 0) Save3<term, type>(dst, d00, d01, d02, bias, params, dstC), dst += dD;
                    if (M > 1) Save3<term, type>(dst, d10, d11, d12, bias, params, dstC), dst += dD;
                }
            }
            else if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F), d01 = Load<false>(dst + 0 * dD + 1 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F), d11 = Load<false>(dst + 1 * dD + 1 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0 * F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0 * F);
                }
                for (size_t offs = 0, offw = 0; offs < srcC; ++offs, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    if (M > 0) s0 = vdupq_n_f32(src0[offs]), d00 = vmlaq_f32(d00, s0, w0);
                    if (M > 1) s0 = vdupq_n_f32(src1[offs]), d10 = vmlaq_f32(d10, s0, w0);
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_NxM_Ptr GetConvolutionNhwcDirect1x1_4xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_4xM<term, type, 1>;
            case 2: return ConvolutionNhwcDirect1x1_4xM<term, type, 2>;
            }
            assert(0);
            return NULL;
        }
#endif

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_4(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
#if defined(SIMD_ARM64_ENABLE)
            size_t n = 6, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
#else
            size_t n = 2, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
#endif
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_4xN = GetConvolutionNhwcDirect1x1_4xM<term, type>(n);
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_4xM = GetConvolutionNhwcDirect1x1_4xM<term, type>(m);

            float32x4_t _params[4], _bias[4];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                if (dC > 0 * F) _bias[0] = Load<false>(bias + dc + 0 * F);
                if (dC > 1 * F) _bias[1] = Load<false>(bias + dc + 1 * F);
                if (dC > 2 * F) _bias[2] = Load<false>(bias + dc + 2 * F);
                if (dC > 3 * F) _bias[3] = Load<false>(bias + dc + 3 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = Load<false>(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = Load<false>(params + dc + 1 * F);
                    if (dC > 2 * F) _params[2] = Load<false>(params + dc + 2 * F);
                    if (dC > 3 * F) _params[3] = Load<false>(params + dc + 3 * F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_4xN(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_4xM(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
                weight += p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template <TermType term, SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_4<term, type> : ConvolutionNhwcDirect_4<term, type>;
        }

        template <SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            Set<TermLast, type>(p, a);
            Set<TermInterim, SimdConvolutionActivationIdentity>(p, a);
        }

        bool SynetConvolution32fNhwcDirect::Set4r(const ConvParam32f& p, AlgParam& a)
        {
            assert(a.microD == 4 * F);
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
#endif//SIMD_NEON_ENABLE
}
