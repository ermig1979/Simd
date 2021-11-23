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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse2
    {
        SynetConvolution32fDirectNhwc::SynetConvolution32fDirectNhwc(const ConvParam32f & p)
            : Base::SynetConvolution32fDirectNhwc(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        bool SynetConvolution32fDirectNhwc::Preferable(const ConvParam32f & p)
        {
            if (!p.IsDilation(1) || p.trans == 0)
                return false;
            if (p.group == 1)
            {
                if (p.kernelY > p.srcH || p.kernelX > p.srcW)
                    return false;
                double k = double(p.srcC) / p.kernelX / p.kernelY;
                return k < 2.0;
            }
            else if (p.IsDepthwise())
            {
                return true;
            }
            return false;
        }

        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, const ConvParam32f & p, size_t kH, size_t kW, const float * weight, __m128 & sum)
        {
            size_t size = kW * p.srcC, tail = (p.kernelX - kW)*p.srcC*p.dstC, dstC = p.dstC, stride = p.srcW * p.srcC;
            for (size_t ky = 0; ky < kH; ++ky)
            {
                for (size_t i = 0; i < size; ++i, weight += dstC)
                    sum = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(src[i]), _mm_loadu_ps(weight)), sum);
                weight += tail;
                src += stride;
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, const ConvParam32f & p, size_t kH, size_t kW, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF = AlignLo(dstC, F);
            size_t dc = 0;
            for (; dc < dstCF; dc += F)
            {
                __m128 conv = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv);
                _mm_storeu_ps(dst + dc, Activate<type>(conv, params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                __m128 conv = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv);
                _mm_storeu_ps(dst + dc, Activate<type>(conv, params, dc));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x2(const float * src, const ConvParam32f & p, const float * weight, __m128 sums[2][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            __m128 w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm_loadu_ps(weight + 0 * F);
                    w1 = _mm_loadu_ps(weight + 1 * F);
                    s0 = _mm_set1_ps(src0[offset]);
                    sums[0][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[0][0]);
                    sums[0][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[0][1]);
                    s0 = _mm_set1_ps(src1[offset]);
                    sums[1][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[1][0]);
                    sums[1][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[1][1]);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x1(const float * src, const ConvParam32f & p, const float * weight, __m128 sums[2][1])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            __m128 w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm_loadu_ps(weight + 0 * F);
                    s0 = _mm_set1_ps(src0[offset]);
                    sums[0][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[0][0]);
                    s0 = _mm_set1_ps(src1[offset]);
                    sums[1][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[1][0]);
                    weight += dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody2(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m128 sums[2][2];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc + 0 * F) : _mm_setzero_ps();
                __m128 bias1 = bias ? _mm_loadu_ps(bias + dc + 1 * F) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                KernelHwcDefaultBody2x2(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m128 sums[2][1];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                __m128 sums[2][1];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x2(const float * src, const ConvParam32f & p, const float * weight, __m128 sums[6][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            __m128 w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm_loadu_ps(weight + 0 * F);
                    w1 = _mm_loadu_ps(weight + 1 * F);
                    s0 = _mm_set1_ps(src0[offset]);
                    sums[0][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[0][0]);
                    sums[0][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[0][1]);
                    s0 = _mm_set1_ps(src1[offset]);
                    sums[1][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[1][0]);
                    sums[1][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[1][1]);
                    s0 = _mm_set1_ps(src2[offset]);
                    sums[2][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[2][0]);
                    sums[2][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[2][1]);
                    s0 = _mm_set1_ps(src3[offset]);
                    sums[3][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[3][0]);
                    sums[3][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[3][1]);
                    s0 = _mm_set1_ps(src4[offset]);
                    sums[4][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[4][0]);
                    sums[4][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[4][1]);
                    s0 = _mm_set1_ps(src5[offset]);
                    sums[5][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[5][0]);
                    sums[5][1] = _mm_add_ps(_mm_mul_ps(s0, w1), sums[5][1]);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x1(const float * src, const ConvParam32f & p, const float * weight, __m128 sums[6][1])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            __m128 w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm_loadu_ps(weight + 0 * F);
                    s0 = _mm_set1_ps(src0[offset]);
                    sums[0][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[0][0]);
                    s0 = _mm_set1_ps(src1[offset]);
                    sums[1][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[1][0]);
                    s0 = _mm_set1_ps(src2[offset]);
                    sums[2][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[2][0]);
                    s0 = _mm_set1_ps(src3[offset]);
                    sums[3][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[3][0]);
                    s0 = _mm_set1_ps(src4[offset]);
                    sums[4][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[4][0]);
                    s0 = _mm_set1_ps(src5[offset]);
                    sums[5][0] = _mm_add_ps(_mm_mul_ps(s0, w0), sums[5][0]);
                    weight += dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody6(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m128 sums[6][2];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc + 0 * F) : _mm_setzero_ps();
                __m128 bias1 = bias ? _mm_loadu_ps(bias + dc + 1 * F) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                sums[2][0] = bias0;
                sums[2][1] = bias1;
                sums[3][0] = bias0;
                sums[3][1] = bias1;
                sums[4][0] = bias0;
                sums[4][1] = bias1;
                sums[5][0] = bias0;
                sums[5][1] = bias1;
                KernelHwcDefaultBody6x2(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 1 * F));
                _mm_storeu_ps(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 0 * F));
                _mm_storeu_ps(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m128 sums[6][1];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                _mm_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                _mm_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                _mm_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                _mm_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                __m128 sums[6][1];
                __m128 bias0 = bias ? _mm_loadu_ps(bias + dc) : _mm_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                _mm_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                _mm_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                _mm_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                _mm_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                _mm_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
            }
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDefault(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t noseH = p.padY, noseW = p.padX;
            size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
            size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
            size_t bodyW2 = AlignLoAny(bodyW - noseW, 2 * p.strideX) + noseW;
            size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
            size_t wS = p.srcC*p.dstC;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            size_t sy = 0;
            for (; sy < noseH; sy += p.strideY)
            {
                size_t sx = 0;
                const float * w = weight + (noseH - sy) * p.kernelY * wS;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, kY + sy, kX + sx, w + (noseW - sx)*wS, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, w, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kY + sy, kW - sx, w, bias, params, dst);
            }
            src += (sy - noseH)*p.srcW*p.srcC;
            for (; sy < bodyH; sy += p.strideY)
            {
                size_t sx = 0;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, p.kernelY, kX + sx, weight + (noseW - sx)*wS, bias, params, dst);
                for (; sx < bodyW6; sx += 6 * p.strideX, dst += 6 * p.dstC)
                    KernelHwcDefaultBody6<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                for (; sx < bodyW2; sx += 2 * p.strideX, dst += 2 * p.dstC)
                    KernelHwcDefaultBody2<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, weight, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, p.kernelY, kW - sx, weight, bias, params, dst);
                src += p.strideY*p.srcW*p.srcC;
            }
            for (; sy < tailH; sy += p.strideY)
            {
                size_t sx = 0;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, kH - sy, kX + sx, weight + (noseW - sx)*wS, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, weight, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kH - sy, kW - sx, weight, bias, params, dst);
                src += p.strideY*p.srcW*p.srcC;
            }
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t size2F = AlignLo(size, 2 * F);
            size_t size4F = AlignLo(size, 4 * F);
            size_t size8F = AlignLo(size, 8 * F);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < size8F; i += 8 * F)
                    {
                        __m128 sums[8];
                        if (bias)
                        {
                            sums[0] = _mm_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm_loadu_ps(bias + i + 1 * F);
                            sums[2] = _mm_loadu_ps(bias + i + 2 * F);
                            sums[3] = _mm_loadu_ps(bias + i + 3 * F);
                            sums[4] = _mm_loadu_ps(bias + i + 4 * F);
                            sums[5] = _mm_loadu_ps(bias + i + 5 * F);
                            sums[6] = _mm_loadu_ps(bias + i + 6 * F);
                            sums[7] = _mm_loadu_ps(bias + i + 7 * F);
                        }
                        else
                        {
                            sums[0] = _mm_setzero_ps();
                            sums[1] = _mm_setzero_ps();
                            sums[2] = _mm_setzero_ps();
                            sums[3] = _mm_setzero_ps();
                            sums[4] = _mm_setzero_ps();
                            sums[5] = _mm_setzero_ps();
                            sums[6] = _mm_setzero_ps();
                            sums[7] = _mm_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), sums[0]);
                                        sums[1] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), sums[1]);
                                        sums[2] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * F), _mm_loadu_ps(pw + 2 * F)), sums[2]);
                                        sums[3] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * F), _mm_loadu_ps(pw + 3 * F)), sums[3]);
                                        sums[4] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 4 * F), _mm_loadu_ps(pw + 4 * F)), sums[4]);
                                        sums[5] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 5 * F), _mm_loadu_ps(pw + 5 * F)), sums[5]);
                                        sums[6] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 6 * F), _mm_loadu_ps(pw + 6 * F)), sums[6]);
                                        sums[7] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 7 * F), _mm_loadu_ps(pw + 7 * F)), sums[7]);
                                    }
                                }
                            }
                        }
                        _mm_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        _mm_storeu_ps(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        _mm_storeu_ps(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                        _mm_storeu_ps(dst + i + 4 * F, Activate<type>(sums[4], params, i + 4 * F));
                        _mm_storeu_ps(dst + i + 5 * F, Activate<type>(sums[5], params, i + 5 * F));
                        _mm_storeu_ps(dst + i + 6 * F, Activate<type>(sums[6], params, i + 6 * F));
                        _mm_storeu_ps(dst + i + 7 * F, Activate<type>(sums[7], params, i + 7 * F));
                    }
                    for (; i < size4F; i += 4 * F)
                    {
                        __m128 sums[4];
                        if (bias)
                        {
                            sums[0] = _mm_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm_loadu_ps(bias + i + 1 * F);
                            sums[2] = _mm_loadu_ps(bias + i + 2 * F);
                            sums[3] = _mm_loadu_ps(bias + i + 3 * F);
                        }
                        else
                        {
                            sums[0] = _mm_setzero_ps();
                            sums[1] = _mm_setzero_ps();
                            sums[2] = _mm_setzero_ps();
                            sums[3] = _mm_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), sums[0]);
                                        sums[1] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), sums[1]);
                                        sums[2] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * F), _mm_loadu_ps(pw + 2 * F)), sums[2]);
                                        sums[3] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * F), _mm_loadu_ps(pw + 3 * F)), sums[3]);
                                    }
                                }
                            }
                        }
                        _mm_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        _mm_storeu_ps(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        _mm_storeu_ps(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                    }
                    for (; i < size2F; i += 2 * F)
                    {
                        __m128 sums[2];
                        if (bias)
                        {
                            sums[0] = _mm_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm_loadu_ps(bias + i + 1 * F);
                        }
                        else
                        {
                            sums[0] = _mm_setzero_ps();
                            sums[1] = _mm_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), sums[0]);
                                        sums[1] = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), sums[1]);
                                    }
                                }
                            }
                        }
                        _mm_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                    }
                    for (; i < size; i += F)
                    {
                        size_t ci = i >= sizeF ? size - F : i;
                        __m128 sum = bias ? _mm_loadu_ps(bias + ci) : _mm_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + ci;
                                        const float * ps = src + (sy*p.srcW + sx)*size + ci;
                                        sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                    }
                                }
                            }
                        }
                        _mm_storeu_ps(dst + ci, Activate<type>(sum, params, ci));
                    }
                    dst += p.dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge(const float * src, const ConvParam32f & p, size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main1(const float * src, size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * srcC), _mm_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * srcC), _mm_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * srcC), _mm_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * srcC), _mm_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * srcC), _mm_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * srcC), _mm_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main2(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m128 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                dst -= srcCF - c;
                __m128 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4(const float * src, const ConvParam32f & p, size_t dy, size_t dx, const __m128 * weight, __m128 bias, const float * params, float * dst)
        {
            __m128 sum = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                size_t sy = dy * p.strideY + ky - p.padY;
                if (sy < p.srcH)
                {
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        if (sx < p.srcW)
                        {
                            const float * ps = src + (sy*p.srcW + sx) * F;
                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), weight[ky * 3 + kx]), sum);
                        }
                    }
                }
            }
            _mm_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x1(const float * src, size_t srcS, const __m128 * weight, __m128 bias, const float * params, float * dst)
        {
            __m128 sum = bias;
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0 * F), weight[0]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 1 * F), weight[1]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 2 * F), weight[2]), sum);
            src += srcS;
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0 * F), weight[3]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 1 * F), weight[4]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 2 * F), weight[5]), sum);
            src += srcS;
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0 * F), weight[6]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 1 * F), weight[7]), sum);
            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 2 * F), weight[8]), sum);
            _mm_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x2(const float * src, size_t srcS, const __m128 * weight, __m128 bias, const float * params, float * dst)
        {
            __m128 sum0 = bias;
            __m128 sum1 = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                __m128 s0 = _mm_loadu_ps(src + 0 * F);
                __m128 s1 = _mm_loadu_ps(src + 1 * F);
                __m128 s2 = _mm_loadu_ps(src + 2 * F);
                __m128 s3 = _mm_loadu_ps(src + 3 * F);
                sum0 = _mm_add_ps(_mm_mul_ps(s0, weight[0]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(s1, weight[0]), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(s1, weight[1]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(s2, weight[1]), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(s2, weight[2]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(s3, weight[2]), sum1);
                src += srcS;
                weight += 3;
            }
            _mm_storeu_ps(dst + 0, Activate<type>(sum0, params, 0));
            _mm_storeu_ps(dst + F, Activate<type>(sum1, params, 0));
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcS = p.srcC*p.srcW;
            size_t srcX = p.srcC*p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            if (p.dstC == F && p.strideX == 1)
            {
                __m128 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm_loadu_ps(weight + i * F);
                __m128 _bias = bias ? _mm_loadu_ps(bias) : _mm_setzero_ps();
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW2; dx += 2)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x2<type>(src + offset, srcS, _weight, _bias, params, dst), offset += 2 * F, dst += 2 * F;
                    for (; dx < dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x1<type>(src + offset, srcS, _weight, _bias, params, dst), offset += F, dst += F;
                    for (; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
            }
            else
            {
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW4; dx += 4)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * p.dstC, offset += 4 * srcX;
                    for (; dx < dstW2; dx += 2)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * p.dstC, offset += 2 * srcX;
                    for (; dx < dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    for (; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            }
        }

        template <::SimdConvolutionActivationType type> SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr GetConvolutionBiasActivation(const ConvParam32f & p)
        {
            if (p.group == 1)
                return ConvolutionDirectNhwcConvolutionBiasActivationDefault<type>;
            else if (p.IsDepthwise())
            {
                if (p.IsKernel(3) && p.IsDilation(1))
                    return ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3<type>;
                else
                    return ConvolutionDirectNhwcConvolutionBiasActivationDepthwise<type>;
            }
            return NULL;
        }

        SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation()
        {
            const ConvParam32f & p = _param;
            SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr func = NULL;
            if (p.dstC >= F && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: func = GetConvolutionBiasActivation<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: func = GetConvolutionBiasActivation<::SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: func = GetConvolutionBiasActivation<::SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: func = GetConvolutionBiasActivation<::SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: func = GetConvolutionBiasActivation<::SimdConvolutionActivationSwish>(p); break;
                }
            }
            return func ? func : Base::SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation();
        };
    }
#endif//SIMD_SSE2_ENABLE
}
