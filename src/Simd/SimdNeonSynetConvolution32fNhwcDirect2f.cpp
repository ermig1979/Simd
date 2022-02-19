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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x6(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            const float* src3 = src0 + 3 * dS;
            const float* src4 = src0 + 4 * dS;
            const float* src5 = src0 + 5 * dS;
            if (dstC > F)
            {
                if (first)
                {
                    d00 = vdupq_n_f32(0.0f); d01 = vdupq_n_f32(0.0f);
                    d10 = vdupq_n_f32(0.0f); d11 = vdupq_n_f32(0.0f);
                    d20 = vdupq_n_f32(0.0f); d21 = vdupq_n_f32(0.0f);
                    d30 = vdupq_n_f32(0.0f); d31 = vdupq_n_f32(0.0f);
                    d40 = vdupq_n_f32(0.0f); d41 = vdupq_n_f32(0.0f);
                    d50 = vdupq_n_f32(0.0f); d51 = vdupq_n_f32(0.0f);
                }
                else
                {
                    d00 = Load<false>(dst + 0 * dD + 0), d01 = Load<false>(dst + 0 * dD + F);
                    d10 = Load<false>(dst + 1 * dD + 0), d11 = Load<false>(dst + 1 * dD + F);
                    d20 = Load<false>(dst + 2 * dD + 0), d21 = Load<false>(dst + 2 * dD + F);
                    d30 = Load<false>(dst + 3 * dD + 0), d31 = Load<false>(dst + 3 * dD + F);
                    d40 = Load<false>(dst + 4 * dD + 0), d41 = Load<false>(dst + 4 * dD + F);
                    d50 = Load<false>(dst + 5 * dD + 0), d51 = Load<false>(dst + 5 * dD + F);
                }
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            w1 = Load<false>(weight + F);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
                            d01 = vmlaq_f32(d01, s0, w1);
                            s0 = vdupq_n_f32(src1[offset]);
                            d10 = vmlaq_f32(d10, s0, w0);
                            d11 = vmlaq_f32(d11, s0, w1);
                            s0 = vdupq_n_f32(src2[offset]);
                            d20 = vmlaq_f32(d20, s0, w0);
                            d21 = vmlaq_f32(d21, s0, w1);
                            s0 = vdupq_n_f32(src3[offset]);
                            d30 = vmlaq_f32(d30, s0, w0);
                            d31 = vmlaq_f32(d31, s0, w1);
                            s0 = vdupq_n_f32(src4[offset]);
                            d40 = vmlaq_f32(d40, s0, w0);
                            d41 = vmlaq_f32(d41, s0, w1);
                            s0 = vdupq_n_f32(src5[offset]);
                            d50 = vmlaq_f32(d50, s0, w0);
                            d51 = vmlaq_f32(d51, s0, w1);
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
                if (first)
                {
                    d00 = vdupq_n_f32(0.0f);
                    d10 = vdupq_n_f32(0.0f);
                    d20 = vdupq_n_f32(0.0f);
                    d30 = vdupq_n_f32(0.0f);
                    d40 = vdupq_n_f32(0.0f);
                    d50 = vdupq_n_f32(0.0f);
                }
                else
                {
                    d00 = Load<false>(dst + 0 * dD + 0);
                    d10 = Load<false>(dst + 1 * dD + 0);
                    d20 = Load<false>(dst + 2 * dD + 0);
                    d30 = Load<false>(dst + 3 * dD + 0);
                    d40 = Load<false>(dst + 4 * dD + 0);
                    d50 = Load<false>(dst + 5 * dD + 0);
                }
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
                            s0 = vdupq_n_f32(src1[offset]);
                            d10 = vmlaq_f32(d10, s0, w0);
                            s0 = vdupq_n_f32(src2[offset]);
                            d20 = vmlaq_f32(d20, s0, w0);
                            s0 = vdupq_n_f32(src3[offset]);
                            d30 = vmlaq_f32(d30, s0, w0);
                            s0 = vdupq_n_f32(src4[offset]);
                            d40 = vmlaq_f32(d40, s0, w0);
                            s0 = vdupq_n_f32(src5[offset]);
                            d50 = vmlaq_f32(d50, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x3(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, s0, w0, w1;
            size_t dS = p.srcC * p.strideX, dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC;
            const float* src1 = src0 + 1 * dS;
            const float* src2 = src0 + 2 * dS;
            if (dstC > F)
            {
                if (first)
                {
                    d00 = vdupq_n_f32(0.0f); d01 = vdupq_n_f32(0.0f);
                    d10 = vdupq_n_f32(0.0f); d11 = vdupq_n_f32(0.0f);
                    d20 = vdupq_n_f32(0.0f); d21 = vdupq_n_f32(0.0f);
                }
                else
                {
                    d00 = Load<false>(dst + 0 * dD + 0), d01 = Load<false>(dst + 0 * dD + F);
                    d10 = Load<false>(dst + 1 * dD + 0), d11 = Load<false>(dst + 1 * dD + F);
                    d20 = Load<false>(dst + 2 * dD + 0), d21 = Load<false>(dst + 2 * dD + F);
                }
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            w1 = Load<false>(weight + F);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
                            d01 = vmlaq_f32(d01, s0, w1);
                            s0 = vdupq_n_f32(src1[offset]);
                            d10 = vmlaq_f32(d10, s0, w0);
                            d11 = vmlaq_f32(d11, s0, w1);
                            s0 = vdupq_n_f32(src2[offset]);
                            d20 = vmlaq_f32(d20, s0, w0);
                            d21 = vmlaq_f32(d21, s0, w1);
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
                if (first)
                {
                    d00 = vdupq_n_f32(0.0f);
                    d10 = vdupq_n_f32(0.0f);
                    d20 = vdupq_n_f32(0.0f);
                }
                else
                {
                    d00 = Load<false>(dst + 0 * dD + 0);
                    d10 = Load<false>(dst + 1 * dD + 0);
                    d20 = Load<false>(dst + 2 * dD + 0);
                }
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
                            s0 = vdupq_n_f32(src1[offset]);
                            d10 = vmlaq_f32(d10, s0, w0);
                            s0 = vdupq_n_f32(src2[offset]);
                            d20 = vmlaq_f32(d20, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const float* src0, const ConvParam32f& p,
            size_t kernelH, size_t kernelW, size_t srcC, size_t dstC, const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, s0, w0, w1;
            size_t dW = DF * (p.kernelX - kernelW) * srcC, dY = p.srcW * p.srcC, dX = p.srcC;
            if (dstC > F)
            {
                if (first)
                    d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0), d01 = Load<false>(dst + F);
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            w1 = Load<false>(weight + F);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
                            d01 = vmlaq_f32(d01, s0, w1);
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
                if (first)
                    d00 = vdupq_n_f32(0.0f);
                else
                    d00 = Load<false>(dst + 0);
                for (size_t ky = 0; ky < kernelH; ++ky)
                {
                    for (size_t kx = 0; kx < kernelW; ++kx)
                    {
                        for (size_t offset = ky * dY + kx * dX, end = offset + srcC; offset < end; ++offset)
                        {
                            w0 = Load<false>(weight + 0);
                            s0 = vdupq_n_f32(src0[offset]);
                            d00 = vmlaq_f32(d00, s0, w0);
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const float* src, const ConvParam32f& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t noseH = p.padY, noseW = p.padX;
            size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
            size_t bodyW3 = AlignLoAny(bodyW - noseW, 3 * p.strideX) + noseW;
            size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
            size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = Load<false>(bias + dc + 0);
                _bias[1] = Load<false>(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg, sy = dy * p.strideY;
                for (; sy < noseH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src;
                    const float* w = weight + (noseH - sy) * p.kernelX * srcC * DF;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, kY + sy, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d, first);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kY + sy, kW - sx, srcC, dC, w, _bias, _params, d, first);
                }
                for (; sy < bodyH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                    const float* w = weight;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, p.kernelY, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d, first);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, p.kernelY, kW - sx, srcC, dC, w, _bias, _params, d, first);
                }
                for (; sy < tailH && dy < yEnd; sy += p.strideY, dy++)
                {
                    size_t sx = 0;
                    const float* s = src + (sy - noseH) * p.srcW * p.srcC;
                    const float* w = weight;
                    for (; sx < noseW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s, p, kH - sy, kX + sx, srcC, dC, w + (noseW - sx) * srcC * DF, _bias, _params, d, first);
                    for (; sx < bodyW6; sx += 6 * p.strideX, d += 6 * p.dstC)
                        ConvolutionNhwcDirect_2x6<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW3; sx += 3 * p.strideX, d += 3 * p.dstC)
                        ConvolutionNhwcDirect_2x3<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < bodyW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, srcC, dC, w, _bias, _params, d, first);
                    for (; sx < tailW; sx += p.strideX, d += p.dstC)
                        ConvolutionNhwcDirect_2x1<term, type>(s + (sx - noseW) * p.srcC, p, kH - sy, kW - sx, srcC, dC, w, _bias, _params, d, first);
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
                        if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            ConvolutionNhwcDirect_2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(const float* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, int first)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            size_t dS = p.srcC, dD = p.dstC;
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
                for (size_t offset = 0; offset < srcC; ++offset)
                {
                    w0 = Load<false>(weight + 0);
                    w1 = Load<false>(weight + F);
                    if (M > 0) s0 = vdupq_n_f32(src0[offset]), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                    if (M > 1) s0 = vdupq_n_f32(src1[offset]), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                    if (M > 2) s0 = vdupq_n_f32(src2[offset]), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                    if (M > 3) s0 = vdupq_n_f32(src3[offset]), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                    if (M > 4) s0 = vdupq_n_f32(src4[offset]), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                    if (M > 5) s0 = vdupq_n_f32(src5[offset]), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
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
                for (size_t offset = 0; offset < srcC; ++offset)
                {
                    w0 = Load<false>(weight + 0);
                    if (M > 0) s0 = vdupq_n_f32(src0[offset]), d00 = vmlaq_f32(d00, s0, w0);
                    if (M > 1) s0 = vdupq_n_f32(src1[offset]), d10 = vmlaq_f32(d10, s0, w0);
                    if (M > 2) s0 = vdupq_n_f32(src2[offset]), d20 = vmlaq_f32(d20, s0, w0);
                    if (M > 3) s0 = vdupq_n_f32(src3[offset]), d30 = vmlaq_f32(d30, s0, w0);
                    if (M > 4) s0 = vdupq_n_f32(src4[offset]), d40 = vmlaq_f32(d40, s0, w0);
                    if (M > 5) s0 = vdupq_n_f32(src5[offset]), d50 = vmlaq_f32(d50, s0, w0);
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

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const float* src0, const ConvParam32f& p, size_t srcC, size_t dstC, const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, int first);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
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

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const float* src, const ConvParam32f& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t n1 = (yEnd - yBeg) * p.dstW;
            size_t n6 = AlignLoAny(n1, 6);
            size_t nTail = n1 - n6;
            ConvolutionNhwcDirect1x1_2xM_Ptr bodyN = GetConvolutionNhwcDirect1x1_2xM<term, type>(6);
            ConvolutionNhwcDirect1x1_2xM_Ptr tailN = GetConvolutionNhwcDirect1x1_2xM<term, type>(nTail);

            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = Load<false>(bias + dc + 0);
                _bias[1] = Load<false>(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                const float* ps = src + yBeg * p.srcW * p.srcC;
                float* pd = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < n6; i += 6, ps += 6 * p.srcC, pd += 6 * p.dstC)
                    bodyN(ps, p, srcC, dC, weight, _bias, _params, pd, first);
                if (nTail)
                    tailN(ps, p, srcC, dC, weight, _bias, _params, pd, first), ps += nTail * p.srcC, pd += nTail * p.dstC;
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
                        if (sc + macroC == p.srcC)
                            ConvolutionNhwcDirect1x1_2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            ConvolutionNhwcDirect1x1_2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD) * macroC;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        //---------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, SynetConvolution32fNhwcDirect::OldConvolutionPtr & convolution)
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
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, convolution); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, convolution); break;
            default: assert(0);
            }
            return true;
        }
    }
#endif//SIMD_NEON_ENABLE
}
