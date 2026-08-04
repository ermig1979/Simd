/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        using AlgParam = SynetConvolution32fNhwcDirect::AlgParam;

        typedef void(*ConvolutionNhwcDirect_NxM_Ptr)(const float* src0, const ConvParam& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float* bias, const float* params, float* dst, int first);
        typedef void(*ConvolutionNhwcDirect1x1_NxM_Ptr)(const float* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float* bias, const float* params, float* dst, int first);

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = a.F, DF = 2 * F;
            svfloat32_t d00, d01, s0, w0, w1;
            size_t srcH = p.srcH, srcW = p.srcW, dilY = p.dilationY, dilX = p.dilationX;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dW = p.srcC * F;
            size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const float* weight1 = weight0 + a.stepW;
            if (dstC > F)
            {
                if (first)
                    d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                else
                    d00 = svld1_f32(svptrue_b32(), dst + 0 * F), d01 = svld1_f32(svptrue_b32(), dst + 1 * F);
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
                                w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                                w1 = svld1_f32(svptrue_b32(), weight1 + offw);
                                s0 = svdup_n_f32(src0[offs]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0), d01 = svmla_f32_x(svptrue_b32(), d01, s0, w1);
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
                    d00 = svdup_n_f32(0.0f);
                else
                    d00 = svld1_f32(svptrue_b32(), dst + 0 * F);
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
                                w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                                s0 = svdup_n_f32(src0[offs]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const float* src0, const ConvParam& p,
            const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const float* weight0, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = a.F, DF = 2 * F;
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
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
                    if (M > 0x0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                    if (M > 0x1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                    if (M > 0x2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                    if (M > 0x3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                    if (M > 0x4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                    if (M > 0x5) d50 = svdup_n_f32(0.0f), d51 = svdup_n_f32(0.0f);
                    if (M > 0x6) d60 = svdup_n_f32(0.0f), d61 = svdup_n_f32(0.0f);
                    if (M > 0x7) d70 = svdup_n_f32(0.0f), d71 = svdup_n_f32(0.0f);
                    if (M > 0x8) d80 = svdup_n_f32(0.0f), d81 = svdup_n_f32(0.0f);
                    if (M > 0x9) d90 = svdup_n_f32(0.0f), d91 = svdup_n_f32(0.0f);
                    if (M > 0xa) da0 = svdup_n_f32(0.0f), da1 = svdup_n_f32(0.0f);
                    if (M > 0xb) db0 = svdup_n_f32(0.0f), db1 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + 0), d01 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + 0), d11 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + 0), d21 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + 0), d31 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + 0), d41 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + 0), d51 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + 0), d61 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + 0), d71 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + 0), d81 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + 0), d91 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = svld1_f32(svptrue_b32(), dst + 0xa * dD + 0), da1 = svld1_f32(svptrue_b32(), dst + 0xa * dD + F);
                    if (M > 0xb) db0 = svld1_f32(svptrue_b32(), dst + 0xb * dD + 0), db1 = svld1_f32(svptrue_b32(), dst + 0xb * dD + F);
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
                                w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                                w1 = svld1_f32(svptrue_b32(), weight1 + offw);
                                if (M > 0x0) s0 = svdup_n_f32(src0[off0]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0), d01 = svmla_f32_x(svptrue_b32(), d01, s0, w1);
                                if (M > 0x1) s0 = svdup_n_f32(src1[off0]), d10 = svmla_f32_x(svptrue_b32(), d10, s0, w0), d11 = svmla_f32_x(svptrue_b32(), d11, s0, w1);
                                if (M > 0x2) s0 = svdup_n_f32(src2[off0]), d20 = svmla_f32_x(svptrue_b32(), d20, s0, w0), d21 = svmla_f32_x(svptrue_b32(), d21, s0, w1);
                                if (M > 0x3) s0 = svdup_n_f32(src3[off0]), d30 = svmla_f32_x(svptrue_b32(), d30, s0, w0), d31 = svmla_f32_x(svptrue_b32(), d31, s0, w1);
                                if (M > 0x4) s0 = svdup_n_f32(src4[off0]), d40 = svmla_f32_x(svptrue_b32(), d40, s0, w0), d41 = svmla_f32_x(svptrue_b32(), d41, s0, w1);
                                if (M > 0x5) s0 = svdup_n_f32(src5[off0]), d50 = svmla_f32_x(svptrue_b32(), d50, s0, w0), d51 = svmla_f32_x(svptrue_b32(), d51, s0, w1);
                                if (M > 0x6) s0 = svdup_n_f32(src0[off6]), d60 = svmla_f32_x(svptrue_b32(), d60, s0, w0), d61 = svmla_f32_x(svptrue_b32(), d61, s0, w1);
                                if (M > 0x7) s0 = svdup_n_f32(src1[off6]), d70 = svmla_f32_x(svptrue_b32(), d70, s0, w0), d71 = svmla_f32_x(svptrue_b32(), d71, s0, w1);
                                if (M > 0x8) s0 = svdup_n_f32(src2[off6]), d80 = svmla_f32_x(svptrue_b32(), d80, s0, w0), d81 = svmla_f32_x(svptrue_b32(), d81, s0, w1);
                                if (M > 0x9) s0 = svdup_n_f32(src3[off6]), d90 = svmla_f32_x(svptrue_b32(), d90, s0, w0), d91 = svmla_f32_x(svptrue_b32(), d91, s0, w1);
                                if (M > 0xa) s0 = svdup_n_f32(src4[off6]), da0 = svmla_f32_x(svptrue_b32(), da0, s0, w0), da1 = svmla_f32_x(svptrue_b32(), da1, s0, w1);
                                if (M > 0xb) s0 = svdup_n_f32(src5[off6]), db0 = svmla_f32_x(svptrue_b32(), db0, s0, w0), db1 = svmla_f32_x(svptrue_b32(), db1, s0, w1);
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
                    if (M > 0x0) d00 = svdup_n_f32(0.0f);
                    if (M > 0x1) d10 = svdup_n_f32(0.0f);
                    if (M > 0x2) d20 = svdup_n_f32(0.0f);
                    if (M > 0x3) d30 = svdup_n_f32(0.0f);
                    if (M > 0x4) d40 = svdup_n_f32(0.0f);
                    if (M > 0x5) d50 = svdup_n_f32(0.0f);
                    if (M > 0x6) d60 = svdup_n_f32(0.0f);
                    if (M > 0x7) d70 = svdup_n_f32(0.0f);
                    if (M > 0x8) d80 = svdup_n_f32(0.0f);
                    if (M > 0x9) d90 = svdup_n_f32(0.0f);
                    if (M > 0xa) da0 = svdup_n_f32(0.0f);
                    if (M > 0xb) db0 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = svld1_f32(svptrue_b32(), dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = svld1_f32(svptrue_b32(), dst + 0xb * dD + 0);
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
                                w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                                if (M > 0x0) s0 = svdup_n_f32(src0[off0]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0);
                                if (M > 0x1) s0 = svdup_n_f32(src1[off0]), d10 = svmla_f32_x(svptrue_b32(), d10, s0, w0);
                                if (M > 0x2) s0 = svdup_n_f32(src2[off0]), d20 = svmla_f32_x(svptrue_b32(), d20, s0, w0);
                                if (M > 0x3) s0 = svdup_n_f32(src3[off0]), d30 = svmla_f32_x(svptrue_b32(), d30, s0, w0);
                                if (M > 0x4) s0 = svdup_n_f32(src4[off0]), d40 = svmla_f32_x(svptrue_b32(), d40, s0, w0);
                                if (M > 0x5) s0 = svdup_n_f32(src5[off0]), d50 = svmla_f32_x(svptrue_b32(), d50, s0, w0);
                                if (M > 0x6) s0 = svdup_n_f32(src0[off6]), d60 = svmla_f32_x(svptrue_b32(), d60, s0, w0);
                                if (M > 0x7) s0 = svdup_n_f32(src1[off6]), d70 = svmla_f32_x(svptrue_b32(), d70, s0, w0);
                                if (M > 0x8) s0 = svdup_n_f32(src2[off6]), d80 = svmla_f32_x(svptrue_b32(), d80, s0, w0);
                                if (M > 0x9) s0 = svdup_n_f32(src3[off6]), d90 = svmla_f32_x(svptrue_b32(), d90, s0, w0);
                                if (M > 0xa) s0 = svdup_n_f32(src4[off6]), da0 = svmla_f32_x(svptrue_b32(), da0, s0, w0);
                                if (M > 0xb) s0 = svdup_n_f32(src5[off6]), db0 = svmla_f32_x(svptrue_b32(), db0, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = a.F;
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2x1 = ConvolutionNhwcDirect_2x1<term, type>;
            size_t n = 12, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<term, type>(n);
            ConvolutionNhwcDirect_NxM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                const float* _bias = bias + dc;
                const float* _params = type == ::SimdConvolutionActivationPrelu ? params + dc : params;
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

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam& p,
            const AlgParam& a, size_t srcC, size_t dstC, const float* weight0, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = a.F, DF = 2 * F;
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
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
                    if (M > 0x0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                    if (M > 0x1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                    if (M > 0x2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                    if (M > 0x3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                    if (M > 0x4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                    if (M > 0x5) d50 = svdup_n_f32(0.0f), d51 = svdup_n_f32(0.0f);
                    if (M > 0x6) d60 = svdup_n_f32(0.0f), d61 = svdup_n_f32(0.0f);
                    if (M > 0x7) d70 = svdup_n_f32(0.0f), d71 = svdup_n_f32(0.0f);
                    if (M > 0x8) d80 = svdup_n_f32(0.0f), d81 = svdup_n_f32(0.0f);
                    if (M > 0x9) d90 = svdup_n_f32(0.0f), d91 = svdup_n_f32(0.0f);
                    if (M > 0xa) da0 = svdup_n_f32(0.0f), da1 = svdup_n_f32(0.0f);
                    if (M > 0xb) db0 = svdup_n_f32(0.0f), db1 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + 0), d01 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + F);
                    if (M > 0x1) d10 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + 0), d11 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + F);
                    if (M > 0x2) d20 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + 0), d21 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + F);
                    if (M > 0x3) d30 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + 0), d31 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + F);
                    if (M > 0x4) d40 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + 0), d41 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + F);
                    if (M > 0x5) d50 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + 0), d51 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + F);
                    if (M > 0x6) d60 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + 0), d61 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + F);
                    if (M > 0x7) d70 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + 0), d71 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + F);
                    if (M > 0x8) d80 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + 0), d81 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + F);
                    if (M > 0x9) d90 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + 0), d91 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + F);
                    if (M > 0xa) da0 = svld1_f32(svptrue_b32(), dst + 0xa * dD + 0), da1 = svld1_f32(svptrue_b32(), dst + 0xa * dD + F);
                    if (M > 0xb) db0 = svld1_f32(svptrue_b32(), dst + 0xb * dD + 0), db1 = svld1_f32(svptrue_b32(), dst + 0xb * dD + F);
                }
                for (size_t off0 = 0, off6 = 6 * dS, offw = 0; off0 < srcC; ++off0, ++off6, offw += F)
                {
                    w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                    w1 = svld1_f32(svptrue_b32(), weight1 + offw);
                    if (M > 0x0) s0 = svdup_n_f32(src0[off0]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0), d01 = svmla_f32_x(svptrue_b32(), d01, s0, w1);
                    if (M > 0x1) s0 = svdup_n_f32(src1[off0]), d10 = svmla_f32_x(svptrue_b32(), d10, s0, w0), d11 = svmla_f32_x(svptrue_b32(), d11, s0, w1);
                    if (M > 0x2) s0 = svdup_n_f32(src2[off0]), d20 = svmla_f32_x(svptrue_b32(), d20, s0, w0), d21 = svmla_f32_x(svptrue_b32(), d21, s0, w1);
                    if (M > 0x3) s0 = svdup_n_f32(src3[off0]), d30 = svmla_f32_x(svptrue_b32(), d30, s0, w0), d31 = svmla_f32_x(svptrue_b32(), d31, s0, w1);
                    if (M > 0x4) s0 = svdup_n_f32(src4[off0]), d40 = svmla_f32_x(svptrue_b32(), d40, s0, w0), d41 = svmla_f32_x(svptrue_b32(), d41, s0, w1);
                    if (M > 0x5) s0 = svdup_n_f32(src5[off0]), d50 = svmla_f32_x(svptrue_b32(), d50, s0, w0), d51 = svmla_f32_x(svptrue_b32(), d51, s0, w1);
                    if (M > 0x6) s0 = svdup_n_f32(src0[off6]), d60 = svmla_f32_x(svptrue_b32(), d60, s0, w0), d61 = svmla_f32_x(svptrue_b32(), d61, s0, w1);
                    if (M > 0x7) s0 = svdup_n_f32(src1[off6]), d70 = svmla_f32_x(svptrue_b32(), d70, s0, w0), d71 = svmla_f32_x(svptrue_b32(), d71, s0, w1);
                    if (M > 0x8) s0 = svdup_n_f32(src2[off6]), d80 = svmla_f32_x(svptrue_b32(), d80, s0, w0), d81 = svmla_f32_x(svptrue_b32(), d81, s0, w1);
                    if (M > 0x9) s0 = svdup_n_f32(src3[off6]), d90 = svmla_f32_x(svptrue_b32(), d90, s0, w0), d91 = svmla_f32_x(svptrue_b32(), d91, s0, w1);
                    if (M > 0xa) s0 = svdup_n_f32(src4[off6]), da0 = svmla_f32_x(svptrue_b32(), da0, s0, w0), da1 = svmla_f32_x(svptrue_b32(), da1, s0, w1);
                    if (M > 0xb) s0 = svdup_n_f32(src5[off6]), db0 = svmla_f32_x(svptrue_b32(), db0, s0, w0), db1 = svmla_f32_x(svptrue_b32(), db1, s0, w1);
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
                    if (M > 0x0) d00 = svdup_n_f32(0.0f);
                    if (M > 0x1) d10 = svdup_n_f32(0.0f);
                    if (M > 0x2) d20 = svdup_n_f32(0.0f);
                    if (M > 0x3) d30 = svdup_n_f32(0.0f);
                    if (M > 0x4) d40 = svdup_n_f32(0.0f);
                    if (M > 0x5) d50 = svdup_n_f32(0.0f);
                    if (M > 0x6) d60 = svdup_n_f32(0.0f);
                    if (M > 0x7) d70 = svdup_n_f32(0.0f);
                    if (M > 0x8) d80 = svdup_n_f32(0.0f);
                    if (M > 0x9) d90 = svdup_n_f32(0.0f);
                    if (M > 0xa) da0 = svdup_n_f32(0.0f);
                    if (M > 0xb) db0 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0x0) d00 = svld1_f32(svptrue_b32(), dst + 0x0 * dD + 0);
                    if (M > 0x1) d10 = svld1_f32(svptrue_b32(), dst + 0x1 * dD + 0);
                    if (M > 0x2) d20 = svld1_f32(svptrue_b32(), dst + 0x2 * dD + 0);
                    if (M > 0x3) d30 = svld1_f32(svptrue_b32(), dst + 0x3 * dD + 0);
                    if (M > 0x4) d40 = svld1_f32(svptrue_b32(), dst + 0x4 * dD + 0);
                    if (M > 0x5) d50 = svld1_f32(svptrue_b32(), dst + 0x5 * dD + 0);
                    if (M > 0x6) d60 = svld1_f32(svptrue_b32(), dst + 0x6 * dD + 0);
                    if (M > 0x7) d70 = svld1_f32(svptrue_b32(), dst + 0x7 * dD + 0);
                    if (M > 0x8) d80 = svld1_f32(svptrue_b32(), dst + 0x8 * dD + 0);
                    if (M > 0x9) d90 = svld1_f32(svptrue_b32(), dst + 0x9 * dD + 0);
                    if (M > 0xa) da0 = svld1_f32(svptrue_b32(), dst + 0xa * dD + 0);
                    if (M > 0xb) db0 = svld1_f32(svptrue_b32(), dst + 0xb * dD + 0);
                }
                for (size_t off0 = 0, off6 = 6 * dS, offw = 0; off0 < srcC; ++off0, ++off6, offw += F)
                {
                    w0 = svld1_f32(svptrue_b32(), weight0 + offw);
                    if (M > 0x0) s0 = svdup_n_f32(src0[off0]), d00 = svmla_f32_x(svptrue_b32(), d00, s0, w0);
                    if (M > 0x1) s0 = svdup_n_f32(src1[off0]), d10 = svmla_f32_x(svptrue_b32(), d10, s0, w0);
                    if (M > 0x2) s0 = svdup_n_f32(src2[off0]), d20 = svmla_f32_x(svptrue_b32(), d20, s0, w0);
                    if (M > 0x3) s0 = svdup_n_f32(src3[off0]), d30 = svmla_f32_x(svptrue_b32(), d30, s0, w0);
                    if (M > 0x4) s0 = svdup_n_f32(src4[off0]), d40 = svmla_f32_x(svptrue_b32(), d40, s0, w0);
                    if (M > 0x5) s0 = svdup_n_f32(src5[off0]), d50 = svmla_f32_x(svptrue_b32(), d50, s0, w0);
                    if (M > 0x6) s0 = svdup_n_f32(src0[off6]), d60 = svmla_f32_x(svptrue_b32(), d60, s0, w0);
                    if (M > 0x7) s0 = svdup_n_f32(src1[off6]), d70 = svmla_f32_x(svptrue_b32(), d70, s0, w0);
                    if (M > 0x8) s0 = svdup_n_f32(src2[off6]), d80 = svmla_f32_x(svptrue_b32(), d80, s0, w0);
                    if (M > 0x9) s0 = svdup_n_f32(src3[off6]), d90 = svmla_f32_x(svptrue_b32(), d90, s0, w0);
                    if (M > 0xa) s0 = svdup_n_f32(src4[off6]), da0 = svmla_f32_x(svptrue_b32(), da0, s0, w0);
                    if (M > 0xb) s0 = svdup_n_f32(src5[off6]), db0 = svmla_f32_x(svptrue_b32(), db0, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = a.F;
            size_t n = 12, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<term, type>(n);
            ConvolutionNhwcDirect1x1_NxM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<term, type>(m);
            for (size_t dc = 0; dc < dstC; dc += a.microD)
            {
                size_t dC = Simd::Min(a.microD, dstC - dc);
                const float* _bias = bias + dc;
                const float* _params = type == ::SimdConvolutionActivationPrelu ? params + dc : params;
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

        template <TermType term, SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam& p, AlgParam& a)
        {
            a.convolutions[term] = p.Is1x1() ? ConvolutionNhwcDirect1x1_2<term, type> : ConvolutionNhwcDirect_2<term, type>;
        }

        template <SimdConvolutionActivationType type> static SIMD_INLINE void Set(const ConvParam& p, AlgParam& a)
        {
            Set<TermLast, type>(p, a);
            Set<TermInterim, SimdConvolutionActivationIdentity>(p, a);
        }

        bool SynetConvolution32fNhwcDirect::Set2r(const ConvParam& p, AlgParam& a)
        {
            assert(a.microD == 2 * a.F);
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
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, a); break;
            default: assert(0);
            }
            return true;
        }
    }
#endif//SIMD_SVE2_ENABLE
}
