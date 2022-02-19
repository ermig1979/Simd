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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            if (dstC > F)
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
                if (dstC == DF)
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
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
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
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                    if (M > 0x6) d60 = vdupq_n_f32(0.0f), d61 = vdupq_n_f32(0.0f);
                    if (M > 0x7) d70 = vdupq_n_f32(0.0f), d71 = vdupq_n_f32(0.0f);
                    if (M > 0x8) d80 = vdupq_n_f32(0.0f), d81 = vdupq_n_f32(0.0f);
                    if (M > 0x9) d90 = vdupq_n_f32(0.0f), d91 = vdupq_n_f32(0.0f);
                    if (M > 0xa) da0 = vdupq_n_f32(0.0f), da1 = vdupq_n_f32(0.0f);
                    if (M > 0xb) db0 = vdupq_n_f32(0.0f), db1 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0), d01 = Load<false>(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0), d11 = Load<false>(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0), d21 = Load<false>(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0), d31 = Load<false>(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0), d41 = Load<false>(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0), d51 = Load<false>(dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = Load<false>(dst + 0x6 * dD + 0), d61 = Load<false>(dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = Load<false>(dst + 0x7 * dD + 0), d71 = Load<false>(dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = Load<false>(dst + 0x8 * dD + 0), d81 = Load<false>(dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = Load<false>(dst + 0x9 * dD + 0), d91 = Load<false>(dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = Load<false>(dst + 0xa * dD + 0), da1 = Load<false>(dst + 0xa * dD + F);
                    if (M > 0xb) db0 = Load<false>(dst + 0xb * dD + 0), db1 = Load<false>(dst + 0xb * dD + F);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
                            size_t off0 = beg + kx * dX, end = off0 + srcC, off6 = off0 + 6 * dS, offw = 0;
                            for (; off0 < end; ++off0, ++off6, offw += F)
                            {
                                w0 = Load<false>(weight0 + offw);
                                w1 = Load<false>(weight1 + offw);
                                if (M > 0x0) s0 = vdupq_n_f32(src0[off0]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                                if (M > 0x1) s0 = vdupq_n_f32(src1[off0]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                                if (M > 0x2) s0 = vdupq_n_f32(src2[off0]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                                if (M > 0x3) s0 = vdupq_n_f32(src3[off0]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                                if (M > 0x4) s0 = vdupq_n_f32(src4[off0]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                                if (M > 0x5) s0 = vdupq_n_f32(src5[off0]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
                                if (M > 0x6) s0 = vdupq_n_f32(src0[off6]), d60 = vmlaq_f32(d60, s0, w0), d61 = vmlaq_f32(d61, s0, w1);
                                if (M > 0x7) s0 = vdupq_n_f32(src1[off6]), d70 = vmlaq_f32(d70, s0, w0), d71 = vmlaq_f32(d71, s0, w1);
                                if (M > 0x8) s0 = vdupq_n_f32(src2[off6]), d80 = vmlaq_f32(d80, s0, w0), d81 = vmlaq_f32(d81, s0, w1);
                                if (M > 0x9) s0 = vdupq_n_f32(src3[off6]), d90 = vmlaq_f32(d90, s0, w0), d91 = vmlaq_f32(d91, s0, w1);
                                if (M > 0xa) s0 = vdupq_n_f32(src4[off6]), da0 = vmlaq_f32(da0, s0, w0), da1 = vmlaq_f32(da1, s0, w1);
                                if (M > 0xb) s0 = vdupq_n_f32(src5[off6]), db0 = vmlaq_f32(db0, s0, w0), db1 = vmlaq_f32(db1, s0, w1);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                    if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params), dst += dD;
                    if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params), dst += dD;
                    if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params), dst += dD;
                    if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params), dst += dD;
                    if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params), dst += dD;
                    if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                    if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, dstC), dst += dD;
                    if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, dstC), dst += dD;
                    if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, dstC), dst += dD;
                    if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, dstC), dst += dD;
                    if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, dstC), dst += dD;
                    if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f);
                    if (M > 0x6) d60 = vdupq_n_f32(0.0f);
                    if (M > 0x7) d70 = vdupq_n_f32(0.0f);
                    if (M > 0x8) d80 = vdupq_n_f32(0.0f);
                    if (M > 0x9) d90 = vdupq_n_f32(0.0f);
                    if (M > 0xa) da0 = vdupq_n_f32(0.0f);
                    if (M > 0xb) db0 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = Load<false>(dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = Load<false>(dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = Load<false>(dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = Load<false>(dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = Load<false>(dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = Load<false>(dst + 0xb * dD + 0);
                }
                for (size_t ky = 0; ky < kY; ky += dilY)
                {
                    if (sy + ky < srcH)
                    {
                        size_t beg = (sy + ky) * dY + sx * dX;
                        for (size_t kx = 0; kx < kX; kx += dilX)
                        {
                            assert(sx + kx < srcW && sx + kx + M <= srcW);
                            size_t off0 = beg + kx * dX, end = off0 + srcC, off6 = off0 + 6 * dS, offw = 0;
                            for (; off0 < end; ++off0, ++off6, offw += F)
                            {
                                w0 = Load<false>(weight0 + offw);
                                if (M > 0x0) s0 = vdupq_n_f32(src0[off0]), d00 = vmlaq_f32(d00, s0, w0);
                                if (M > 0x1) s0 = vdupq_n_f32(src1[off0]), d10 = vmlaq_f32(d10, s0, w0);
                                if (M > 0x2) s0 = vdupq_n_f32(src2[off0]), d20 = vmlaq_f32(d20, s0, w0);
                                if (M > 0x3) s0 = vdupq_n_f32(src3[off0]), d30 = vmlaq_f32(d30, s0, w0);
                                if (M > 0x4) s0 = vdupq_n_f32(src4[off0]), d40 = vmlaq_f32(d40, s0, w0);
                                if (M > 0x5) s0 = vdupq_n_f32(src5[off0]), d50 = vmlaq_f32(d50, s0, w0);
                                if (M > 0x6) s0 = vdupq_n_f32(src0[off6]), d60 = vmlaq_f32(d60, s0, w0);
                                if (M > 0x7) s0 = vdupq_n_f32(src1[off6]), d70 = vmlaq_f32(d70, s0, w0);
                                if (M > 0x8) s0 = vdupq_n_f32(src2[off6]), d80 = vmlaq_f32(d80, s0, w0);
                                if (M > 0x9) s0 = vdupq_n_f32(src3[off6]), d90 = vmlaq_f32(d90, s0, w0);
                                if (M > 0xa) s0 = vdupq_n_f32(src4[off6]), da0 = vmlaq_f32(da0, s0, w0);
                                if (M > 0xb) s0 = vdupq_n_f32(src5[off6]), db0 = vmlaq_f32(db0, s0, w0);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    if (M > 0x0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 0x1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 0x2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 0x3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 0x4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                    if (M > 0x5) Save1<term, type>(dst, d50, bias, params), dst += dD;
                    if (M > 0x6) Save1<term, type>(dst, d60, bias, params), dst += dD;
                    if (M > 0x7) Save1<term, type>(dst, d70, bias, params), dst += dD;
                    if (M > 0x8) Save1<term, type>(dst, d80, bias, params), dst += dD;
                    if (M > 0x9) Save1<term, type>(dst, d90, bias, params), dst += dD;
                    if (M > 0xa) Save1<term, type>(dst, da0, bias, params), dst += dD;
                    if (M > 0xb) Save1<term, type>(dst, db0, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0x0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 0x1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 0x2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 0x3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 0x4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    if (M > 0x5) Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                    if (M > 0x6) Save1<term, type>(dst, d60, bias, params, dstC), dst += dD;
                    if (M > 0x7) Save1<term, type>(dst, d70, bias, params, dstC), dst += dD;
                    if (M > 0x8) Save1<term, type>(dst, d80, bias, params, dstC), dst += dD;
                    if (M > 0x9) Save1<term, type>(dst, d90, bias, params, dstC), dst += dD;
                    if (M > 0xa) Save1<term, type>(dst, da0, bias, params, dstC), dst += dD;
                    if (M > 0xb) Save1<term, type>(dst, db0, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_NxM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect_2xM<term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect_2xM<term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect_2xM<term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect_2xM<term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect_2xM<term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect_2xM<term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect_2xM<term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect_2xM<term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect_2xM<term, type, 0x9>;
            case 0xa: return ConvolutionNhwcDirect_2xM<term, type, 0xa>;
            case 0xb: return ConvolutionNhwcDirect_2xM<term, type, 0xb>;
            case 0xc: return ConvolutionNhwcDirect_2xM<term, type, 0xc>;
            }
            assert(0);
            return NULL;
        }
#else
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
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
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0), d01 = Load<false>(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0), d11 = Load<false>(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0), d21 = Load<false>(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0), d31 = Load<false>(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0), d41 = Load<false>(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0), d51 = Load<false>(dst + 0x5 * dD + F);
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
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0);
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
            case 6: return ConvolutionNhwcDirect_2xM<term, type, 6>;
            }
            assert(0);
            return NULL;
        }
#endif

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2x1 = ConvolutionNhwcDirect_2x1<term, type>;
#if defined(SIMD_ARM64_ENABLE)
            size_t n = 12, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
#else
            size_t n = 6, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
#endif
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            float32x4_t _params[2], _bias[2];
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
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = Load<false>(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = Load<false>(params + dc + 1 * F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyWn; dx += n, d += p.dstC * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < bodyW; dx += m, d += p.dstC * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                    for (; dx < tailW; dx++, d += p.dstC)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, d, first);
                }
                weight += p.kernelY * p.kernelX * p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

#if defined(SIMD_ARM64_ENABLE)
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                    if (M > 0x6) d60 = vdupq_n_f32(0.0f), d61 = vdupq_n_f32(0.0f);
                    if (M > 0x7) d70 = vdupq_n_f32(0.0f), d71 = vdupq_n_f32(0.0f);
                    if (M > 0x8) d80 = vdupq_n_f32(0.0f), d81 = vdupq_n_f32(0.0f);
                    if (M > 0x9) d90 = vdupq_n_f32(0.0f), d91 = vdupq_n_f32(0.0f);
                    if (M > 0xa) da0 = vdupq_n_f32(0.0f), da1 = vdupq_n_f32(0.0f);
                    if (M > 0xb) db0 = vdupq_n_f32(0.0f), db1 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0), d01 = Load<false>(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0), d11 = Load<false>(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0), d21 = Load<false>(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0), d31 = Load<false>(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0), d41 = Load<false>(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0), d51 = Load<false>(dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = Load<false>(dst + 0x6 * dD + 0), d61 = Load<false>(dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = Load<false>(dst + 0x7 * dD + 0), d71 = Load<false>(dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = Load<false>(dst + 0x8 * dD + 0), d81 = Load<false>(dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = Load<false>(dst + 0x9 * dD + 0), d91 = Load<false>(dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = Load<false>(dst + 0xa * dD + 0), da1 = Load<false>(dst + 0xa * dD + F);
                    if (M > 0xb) db0 = Load<false>(dst + 0xb * dD + 0), db1 = Load<false>(dst + 0xb * dD + F);
                }
                for (size_t off0 = 0, off6 = 6 * dS, offw = 0; off0 < srcC; ++off0, ++off6, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    w1 = Load<false>(weight1 + offw);
                    if (M > 0x0) s0 = vdupq_n_f32(src0[off0]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                    if (M > 0x1) s0 = vdupq_n_f32(src1[off0]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                    if (M > 0x2) s0 = vdupq_n_f32(src2[off0]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                    if (M > 0x3) s0 = vdupq_n_f32(src3[off0]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                    if (M > 0x4) s0 = vdupq_n_f32(src4[off0]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                    if (M > 0x5) s0 = vdupq_n_f32(src5[off0]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
                    if (M > 0x6) s0 = vdupq_n_f32(src0[off6]), d60 = vmlaq_f32(d60, s0, w0), d61 = vmlaq_f32(d61, s0, w1);
                    if (M > 0x7) s0 = vdupq_n_f32(src1[off6]), d70 = vmlaq_f32(d70, s0, w0), d71 = vmlaq_f32(d71, s0, w1);
                    if (M > 0x8) s0 = vdupq_n_f32(src2[off6]), d80 = vmlaq_f32(d80, s0, w0), d81 = vmlaq_f32(d81, s0, w1);
                    if (M > 0x9) s0 = vdupq_n_f32(src3[off6]), d90 = vmlaq_f32(d90, s0, w0), d91 = vmlaq_f32(d91, s0, w1);
                    if (M > 0xa) s0 = vdupq_n_f32(src4[off6]), da0 = vmlaq_f32(da0, s0, w0), da1 = vmlaq_f32(da1, s0, w1);
                    if (M > 0xb) s0 = vdupq_n_f32(src5[off6]), db0 = vmlaq_f32(db0, s0, w0), db1 = vmlaq_f32(db1, s0, w1);
                }
                if (dstC == DF)
                {
                    if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                    if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
                    if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params), dst += dD;
                    if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params), dst += dD;
                    if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params), dst += dD;
                    if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params), dst += dD;
                    if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params), dst += dD;
                    if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0x0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 0x1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 0x2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 0x3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 0x4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                    if (M > 0x5) Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
                    if (M > 0x6) Save2<term, type>(dst, d60, d61, bias, params, dstC), dst += dD;
                    if (M > 0x7) Save2<term, type>(dst, d70, d71, bias, params, dstC), dst += dD;
                    if (M > 0x8) Save2<term, type>(dst, d80, d81, bias, params, dstC), dst += dD;
                    if (M > 0x9) Save2<term, type>(dst, d90, d91, bias, params, dstC), dst += dD;
                    if (M > 0xa) Save2<term, type>(dst, da0, da1, bias, params, dstC), dst += dD;
                    if (M > 0xb) Save2<term, type>(dst, db0, db1, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f);
                    if (M > 0x6) d60 = vdupq_n_f32(0.0f);
                    if (M > 0x7) d70 = vdupq_n_f32(0.0f);
                    if (M > 0x8) d80 = vdupq_n_f32(0.0f);
                    if (M > 0x9) d90 = vdupq_n_f32(0.0f);
                    if (M > 0xa) da0 = vdupq_n_f32(0.0f);
                    if (M > 0xb) db0 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = Load<false>(dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = Load<false>(dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = Load<false>(dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = Load<false>(dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = Load<false>(dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = Load<false>(dst + 0xb * dD + 0);
                }
                for (size_t off0 = 0, off6 = 6 * dS, offw = 0; off0 < srcC; ++off0, ++off6, offw += F)
                {
                    w0 = Load<false>(weight0 + offw);
                    if (M > 0x0) s0 = vdupq_n_f32(src0[off0]), d00 = vmlaq_f32(d00, s0, w0);
                    if (M > 0x1) s0 = vdupq_n_f32(src1[off0]), d10 = vmlaq_f32(d10, s0, w0);
                    if (M > 0x2) s0 = vdupq_n_f32(src2[off0]), d20 = vmlaq_f32(d20, s0, w0);
                    if (M > 0x3) s0 = vdupq_n_f32(src3[off0]), d30 = vmlaq_f32(d30, s0, w0);
                    if (M > 0x4) s0 = vdupq_n_f32(src4[off0]), d40 = vmlaq_f32(d40, s0, w0);
                    if (M > 0x5) s0 = vdupq_n_f32(src5[off0]), d50 = vmlaq_f32(d50, s0, w0);
                    if (M > 0x6) s0 = vdupq_n_f32(src0[off6]), d60 = vmlaq_f32(d60, s0, w0);
                    if (M > 0x7) s0 = vdupq_n_f32(src1[off6]), d70 = vmlaq_f32(d70, s0, w0);
                    if (M > 0x8) s0 = vdupq_n_f32(src2[off6]), d80 = vmlaq_f32(d80, s0, w0);
                    if (M > 0x9) s0 = vdupq_n_f32(src3[off6]), d90 = vmlaq_f32(d90, s0, w0);
                    if (M > 0xa) s0 = vdupq_n_f32(src4[off6]), da0 = vmlaq_f32(da0, s0, w0);
                    if (M > 0xb) s0 = vdupq_n_f32(src5[off6]), db0 = vmlaq_f32(db0, s0, w0);
                }
                if (dstC == F)
                {
                    if (M > 0x0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 0x1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 0x2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 0x3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 0x4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                    if (M > 0x5) Save1<term, type>(dst, d50, bias, params), dst += dD;
                    if (M > 0x6) Save1<term, type>(dst, d60, bias, params), dst += dD;
                    if (M > 0x7) Save1<term, type>(dst, d70, bias, params), dst += dD;
                    if (M > 0x8) Save1<term, type>(dst, d80, bias, params), dst += dD;
                    if (M > 0x9) Save1<term, type>(dst, d90, bias, params), dst += dD;
                    if (M > 0xa) Save1<term, type>(dst, da0, bias, params), dst += dD;
                    if (M > 0xb) Save1<term, type>(dst, db0, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0x0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 0x1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 0x2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 0x3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 0x4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                    if (M > 0x5) Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
                    if (M > 0x6) Save1<term, type>(dst, d60, bias, params, dstC), dst += dD;
                    if (M > 0x7) Save1<term, type>(dst, d70, bias, params, dstC), dst += dD;
                    if (M > 0x8) Save1<term, type>(dst, d80, bias, params, dstC), dst += dD;
                    if (M > 0x9) Save1<term, type>(dst, d90, bias, params, dstC), dst += dD;
                    if (M > 0xa) Save1<term, type>(dst, da0, bias, params, dstC), dst += dD;
                    if (M > 0xb) Save1<term, type>(dst, db0, bias, params, dstC), dst += dD;
                }
            }
        }

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_NxM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
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
            }
            assert(0);
            return NULL;
        }
#else
        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
            const float* weight1 = weight0 + a.stepW;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (first)
                {
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0), d01 = Load<false>(dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0), d11 = Load<false>(dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0), d21 = Load<false>(dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0), d31 = Load<false>(dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0), d41 = Load<false>(dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0), d51 = Load<false>(dst + 0x5 * dD + F);
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
                    if (M > 0x0) d00 = vdupq_n_f32(0.0f);
                    if (M > 0x1) d10 = vdupq_n_f32(0.0f);
                    if (M > 0x2) d20 = vdupq_n_f32(0.0f);
                    if (M > 0x3) d30 = vdupq_n_f32(0.0f);
                    if (M > 0x4) d40 = vdupq_n_f32(0.0f);
                    if (M > 0x5) d50 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = Load<false>(dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = Load<false>(dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = Load<false>(dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = Load<false>(dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = Load<false>(dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = Load<false>(dst + 0x5 * dD + 0);
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
            case 6: return ConvolutionNhwcDirect1x1_2xM<term, type, 6>;
            }
            assert(0);
            return NULL;
        }
#endif

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
#if defined(SIMD_ARM64_ENABLE)
            size_t n = 12, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
#else
            size_t n = 6, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
#endif
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<term, type>(n);
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);

            float32x4_t _params[2], _bias[2];
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
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    if (dC > 0 * F) _params[0] = Load<false>(params + dc + 0 * F);
                    if (dC > 1 * F) _params[1] = Load<false>(params + dc + 1 * F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, ps += n * p.srcC, pd += n * p.dstC)
                    convolutionNhwcDirect1x1_2xN(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
                for (; i < n1; i += m, ps += m * p.srcC, pd += m * p.dstC)
                    convolutionNhwcDirect1x1_2xM(ps, p, a, srcC, dC, weight, _bias, _params, pd, first);
                weight += p.srcC * a.microD;
            }
        }

        //---------------------------------------------------------------------

        template <TermType term, SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_2<term, type> : ConvolutionNhwcDirect_2<term, type>;
        }

        template <SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam32f& p, AlgParam& a)
        {
            Set<TermLast, type>(p, a);
            Set<TermInterim, SimdConvolutionActivationIdentity>(p, a);
        }

        bool SynetConvolution32fNhwcDirect::Set2r(const ConvParam32f& p, AlgParam& a)
        {
            assert(a.microD == 2 * F);
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
