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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
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

        template<::SimdConvolutionActivationType type> SIMD_INLINE float32x4_t Activate(float32x4_t value, const float * params, size_t offset);

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationIdentity>(float32x4_t value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRelu>(float32x4_t value, const float * params, size_t offset)
        {
            return vmaxq_f32(vdupq_n_f32(0.0f), value);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationLeakyRelu>(float32x4_t value, const float * params, size_t offset)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), vld1q_dup_f32(params + 0), vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRestrictRange>(float32x4_t value, const float * params, size_t offset)
        {
            return vminq_f32(vmaxq_f32(vld1q_dup_f32(params + 0), value), vld1q_dup_f32(params + 1));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationPrelu>(float32x4_t value, const float * params, size_t offset)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), Load<false>(params + offset), vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationElu>(float32x4_t value, const float * params, size_t offset)
        {
            return Neon::Elu(value, vld1q_dup_f32(params + 0));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationHswish>(float32x4_t value, const float * params, size_t offset)
        {
            return Neon::SynetHswish32f(value, vld1q_dup_f32(params + 0), vld1q_dup_f32(params + 1));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationMish>(float32x4_t value, const float* params, size_t offset)
        {
            return Neon::Mish<1>(value, vld1q_dup_f32(params + 0));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationHardSigmoid>(float32x4_t value, const float* params, size_t offset)
        {
            return Neon::SynetHardSigmoid32f(value, vld1q_dup_f32(params + 0), vld1q_dup_f32(params + 1));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationSwish>(float32x4_t value, const float* params, size_t offset)
        {
            return Neon::Swish<1>(value, vld1q_dup_f32(params + 0));
        }

        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, const ConvParam32f & p, size_t kH, size_t kW, const float * weight, float32x4_t & sum)
        {
            size_t size = kW * p.srcC, tail = (p.kernelX - kW)*p.srcC*p.dstC, dstC = p.dstC, stride = p.srcW * p.srcC;
            for (size_t ky = 0; ky < kH; ++ky)
            {
                for (size_t i = 0; i < size; ++i, weight += dstC)
                    sum = vmlaq_f32(sum, vld1q_dup_f32(src + i), Load<false>(weight));
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
                float32x4_t conv = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv);
                Store<false>(dst + dc, Activate<type>(conv, params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                float32x4_t conv = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv);
                Store<false>(dst + dc, Activate<type>(conv, params, dc));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x2(const float * src, const ConvParam32f & p, const float * weight, float32x4_t sums[2][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            float32x4_t w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = Load<false>(weight + 0 * F);
                    w1 = Load<false>(weight + 1 * F);
                    s0 = vld1q_dup_f32(src0 + offset);
                    sums[0][0] = vmlaq_f32(sums[0][0], s0, w0);
                    sums[0][1] = vmlaq_f32(sums[0][1], s0, w1);
                    s0 = vld1q_dup_f32(src1 + offset);
                    sums[1][0] = vmlaq_f32(sums[1][0], s0, w0);
                    sums[1][1] = vmlaq_f32(sums[1][1], s0, w1);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x1(const float * src, const ConvParam32f & p, const float * weight, float32x4_t sums[2][1])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            float32x4_t w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = Load<false>(weight + 0 * F);
                    s0 = vld1q_dup_f32(src0 + offset);
                    sums[0][0] = vmlaq_f32(sums[0][0], s0, w0);
                    s0 = vld1q_dup_f32(src1 + offset);
                    sums[1][0] = vmlaq_f32(sums[1][0], s0, w0);
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
                float32x4_t sums[2][2];
                float32x4_t bias0 = bias ? Load<false>(bias + dc + 0 * F) : vdupq_n_f32(0.0f);
                float32x4_t bias1 = bias ? Load<false>(bias + dc + 1 * F) : vdupq_n_f32(0.0f);
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                KernelHwcDefaultBody2x2(src, p, weight + dc, sums);
                Store<false>(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                Store<false>(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                Store<false>(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                Store<false>(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                float32x4_t sums[2][1];
                float32x4_t bias0 = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                Store<false>(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                Store<false>(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                float32x4_t sums[2][1];
                float32x4_t bias0 = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                Store<false>(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                Store<false>(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x2(const float * src, const ConvParam32f & p, const float * weight, float32x4_t sums[6][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            float32x4_t w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = Load<false>(weight + 0 * F);
                    w1 = Load<false>(weight + 1 * F);
                    s0 = vld1q_dup_f32(src0 + offset);
                    sums[0][0] = vmlaq_f32(sums[0][0], s0, w0);
                    sums[0][1] = vmlaq_f32(sums[0][1], s0, w1);
                    s0 = vld1q_dup_f32(src1 + offset);
                    sums[1][0] = vmlaq_f32(sums[1][0], s0, w0);
                    sums[1][1] = vmlaq_f32(sums[1][1], s0, w1);
                    s0 = vld1q_dup_f32(src2 + offset);
                    sums[2][0] = vmlaq_f32(sums[2][0], s0, w0);
                    sums[2][1] = vmlaq_f32(sums[2][1], s0, w1);
                    s0 = vld1q_dup_f32(src3 + offset);
                    sums[3][0] = vmlaq_f32(sums[3][0], s0, w0);
                    sums[3][1] = vmlaq_f32(sums[3][1], s0, w1);
                    s0 = vld1q_dup_f32(src4 + offset);
                    sums[4][0] = vmlaq_f32(sums[4][0], s0, w0);
                    sums[4][1] = vmlaq_f32(sums[4][1], s0, w1);
                    s0 = vld1q_dup_f32(src5 + offset);
                    sums[5][0] = vmlaq_f32(sums[5][0], s0, w0);
                    sums[5][1] = vmlaq_f32(sums[5][1], s0, w1);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x1(const float * src, const ConvParam32f & p, const float * weight, float32x4_t sums[6][1])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            float32x4_t w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = Load<false>(weight + 0 * F);
                    s0 = vld1q_dup_f32(src0 + offset);
                    sums[0][0] = vmlaq_f32(sums[0][0], s0, w0);
                    s0 = vld1q_dup_f32(src1 + offset);
                    sums[1][0] = vmlaq_f32(sums[1][0], s0, w0);
                    s0 = vld1q_dup_f32(src2 + offset);
                    sums[2][0] = vmlaq_f32(sums[2][0], s0, w0);
                    s0 = vld1q_dup_f32(src3 + offset);
                    sums[3][0] = vmlaq_f32(sums[3][0], s0, w0);
                    s0 = vld1q_dup_f32(src4 + offset);
                    sums[4][0] = vmlaq_f32(sums[4][0], s0, w0);
                    s0 = vld1q_dup_f32(src5 + offset);
                    sums[5][0] = vmlaq_f32(sums[5][0], s0, w0);
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
                float32x4_t sums[6][2];
                float32x4_t bias0 = bias ? Load<false>(bias + dc + 0 * F) : vdupq_n_f32(0.0f);
                float32x4_t bias1 = bias ? Load<false>(bias + dc + 1 * F) : vdupq_n_f32(0.0f);
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
                Store<false>(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                Store<false>(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                Store<false>(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                Store<false>(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
                Store<false>(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 0 * F));
                Store<false>(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 1 * F));
                Store<false>(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 0 * F));
                Store<false>(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 1 * F));
                Store<false>(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 0 * F));
                Store<false>(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 1 * F));
                Store<false>(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 0 * F));
                Store<false>(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                float32x4_t sums[6][1];
                float32x4_t bias0 = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                Store<false>(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                Store<false>(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                Store<false>(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                Store<false>(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                Store<false>(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                Store<false>(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
            }
            if (dc < dstC)
            {
                dc = dstC - F;
                float32x4_t sums[6][1];
                float32x4_t bias0 = bias ? Load<false>(bias + dc) : vdupq_n_f32(0.0f);
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                Store<false>(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                Store<false>(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                Store<false>(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                Store<false>(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                Store<false>(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                Store<false>(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
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
                        float32x4_t sums[8];
                        if (bias)
                        {
                            sums[0] = Load<false>(bias + i + 0 * F);
                            sums[1] = Load<false>(bias + i + 1 * F);
                            sums[2] = Load<false>(bias + i + 2 * F);
                            sums[3] = Load<false>(bias + i + 3 * F);
                            sums[4] = Load<false>(bias + i + 4 * F);
                            sums[5] = Load<false>(bias + i + 5 * F);
                            sums[6] = Load<false>(bias + i + 6 * F);
                            sums[7] = Load<false>(bias + i + 7 * F);
                        }
                        else
                        {
                            sums[0] = vdupq_n_f32(0.0f);
                            sums[1] = vdupq_n_f32(0.0f);
                            sums[2] = vdupq_n_f32(0.0f);
                            sums[3] = vdupq_n_f32(0.0f);
                            sums[4] = vdupq_n_f32(0.0f);
                            sums[5] = vdupq_n_f32(0.0f);
                            sums[6] = vdupq_n_f32(0.0f);
                            sums[7] = vdupq_n_f32(0.0f);
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
                                        sums[0] = vmlaq_f32(sums[0], Load<false>(ps + 0 * F), Load<false>(pw + 0 * F));
                                        sums[1] = vmlaq_f32(sums[1], Load<false>(ps + 1 * F), Load<false>(pw + 1 * F));
                                        sums[2] = vmlaq_f32(sums[2], Load<false>(ps + 2 * F), Load<false>(pw + 2 * F));
                                        sums[3] = vmlaq_f32(sums[3], Load<false>(ps + 3 * F), Load<false>(pw + 3 * F));
                                        sums[4] = vmlaq_f32(sums[4], Load<false>(ps + 4 * F), Load<false>(pw + 4 * F));
                                        sums[5] = vmlaq_f32(sums[5], Load<false>(ps + 5 * F), Load<false>(pw + 5 * F));
                                        sums[6] = vmlaq_f32(sums[6], Load<false>(ps + 6 * F), Load<false>(pw + 6 * F));
                                        sums[7] = vmlaq_f32(sums[7], Load<false>(ps + 7 * F), Load<false>(pw + 7 * F));
                                    }
                                }
                            }
                        }
                        Store<false>(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        Store<false>(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        Store<false>(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        Store<false>(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                        Store<false>(dst + i + 4 * F, Activate<type>(sums[4], params, i + 4 * F));
                        Store<false>(dst + i + 5 * F, Activate<type>(sums[5], params, i + 5 * F));
                        Store<false>(dst + i + 6 * F, Activate<type>(sums[6], params, i + 6 * F));
                        Store<false>(dst + i + 7 * F, Activate<type>(sums[7], params, i + 7 * F));
                    }
                    for (; i < size4F; i += 4 * F)
                    {
                        float32x4_t sums[4];
                        if (bias)
                        {
                            sums[0] = Load<false>(bias + i + 0 * F);
                            sums[1] = Load<false>(bias + i + 1 * F);
                            sums[2] = Load<false>(bias + i + 2 * F);
                            sums[3] = Load<false>(bias + i + 3 * F);
                        }
                        else
                        {
                            sums[0] = vdupq_n_f32(0.0f);
                            sums[1] = vdupq_n_f32(0.0f);
                            sums[2] = vdupq_n_f32(0.0f);
                            sums[3] = vdupq_n_f32(0.0f);
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
                                        sums[0] = vmlaq_f32(sums[0], Load<false>(ps + 0 * F), Load<false>(pw + 0 * F));
                                        sums[1] = vmlaq_f32(sums[1], Load<false>(ps + 1 * F), Load<false>(pw + 1 * F));
                                        sums[2] = vmlaq_f32(sums[2], Load<false>(ps + 2 * F), Load<false>(pw + 2 * F));
                                        sums[3] = vmlaq_f32(sums[3], Load<false>(ps + 3 * F), Load<false>(pw + 3 * F));
                                    }
                                }
                            }
                        }
                        Store<false>(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        Store<false>(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        Store<false>(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        Store<false>(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                    }
                    for (; i < size2F; i += 2 * F)
                    {
                        float32x4_t sums[2];
                        if (bias)
                        {
                            sums[0] = Load<false>(bias + i + 0 * F);
                            sums[1] = Load<false>(bias + i + 1 * F);
                        }
                        else
                        {
                            sums[0] = vdupq_n_f32(0.0f);
                            sums[1] = vdupq_n_f32(0.0f);
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
                                        sums[0] = vmlaq_f32(sums[0], Load<false>(ps + 0 * F), Load<false>(pw + 0 * F));
                                        sums[1] = vmlaq_f32(sums[1], Load<false>(ps + 1 * F), Load<false>(pw + 1 * F));
                                    }
                                }
                            }
                        }
                        Store<false>(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        Store<false>(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                    }
                    for (; i < size; i += F)
                    {
                        size_t ci = i >= sizeF ? size - F : i;
                        float32x4_t sum = bias ? Load<false>(bias + ci) : vdupq_n_f32(0.0f);
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
                                        sum = vmlaq_f32(sum, Load<false>(ps + 0 * F), Load<false>(pw + 0 * F));
                                    }
                                }
                            }
                        }
                        Store<false>(dst + ci, Activate<type>(sum, params, ci));
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
                float32x4_t sum = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
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
                                sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
                            }
                        }
                    }
                }
                Store<false>(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = p.srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                float32x4_t sum = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
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
                                sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
                            }
                        }
                    }
                }
                Store<false>(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main1(const float * src, size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                float32x4_t sum = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = vmlaq_f32(sum, Load<false>(ps + 0 * srcC), Load<false>(pw + 0 * srcC));
                    sum = vmlaq_f32(sum, Load<false>(ps + 1 * srcC), Load<false>(pw + 1 * srcC));
                    sum = vmlaq_f32(sum, Load<false>(ps + 2 * srcC), Load<false>(pw + 2 * srcC));
                }
                Store<false>(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                float32x4_t sum = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = vmlaq_f32(sum, Load<false>(ps + 0 * srcC), Load<false>(pw + 0 * srcC));
                    sum = vmlaq_f32(sum, Load<false>(ps + 1 * srcC), Load<false>(pw + 1 * srcC));
                    sum = vmlaq_f32(sum, Load<false>(ps + 2 * srcC), Load<false>(pw + 2 * srcC));
                }
                Store<false>(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main2(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            float32x4_t sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 0 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 0 * srcC), w0);
                    pw += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 1 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 1 * srcC), w0);
                    pw += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 2 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 2 * srcC), w0);
                    pw += srcC;
                }
                Store<false>(dst + c, Activate<type>(sum0, params, c));
                Store<false>(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                sum0 = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 0 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 0 * srcC), w0);
                    pw += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 1 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 1 * srcC), w0);
                    pw += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + 2 * srcC), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + 2 * srcC), w0);
                    pw += srcC;
                }
                Store<false>(dst + c, Activate<type>(sum0, params, c));
                Store<false>(dst + c + srcC, Activate<type>(sum1, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                float32x4_t sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
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
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                }
                Store<false>(dst + 0 * srcC, Activate<type>(sum0, params, c));
                Store<false>(dst + 1 * srcC, Activate<type>(sum1, params, c));
                Store<false>(dst + 2 * srcC, Activate<type>(sum2, params, c));
                Store<false>(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                dst -= srcCF - c;
                float32x4_t sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
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
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                    w0 = Load<false>(pw);
                    sum0 = vmlaq_f32(sum0, Load<false>(ps0 + offset), w0);
                    sum1 = vmlaq_f32(sum1, Load<false>(ps1 + offset), w0);
                    sum2 = vmlaq_f32(sum2, Load<false>(ps2 + offset), w0);
                    sum3 = vmlaq_f32(sum3, Load<false>(ps3 + offset), w0);
                    pw += srcC, offset += srcC;
                }
                Store<false>(dst + 0 * srcC, Activate<type>(sum0, params, c));
                Store<false>(dst + 1 * srcC, Activate<type>(sum1, params, c));
                Store<false>(dst + 2 * srcC, Activate<type>(sum2, params, c));
                Store<false>(dst + 3 * srcC, Activate<type>(sum3, params, c));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge4(const float * src, const ConvParam32f & p, size_t dy, size_t dx, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
        {
            float32x4_t sum = bias;
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
                            sum = vmlaq_f32(sum, Load<false>(ps), weight[ky * 3 + kx]);
                        }
                    }
                }
            }
            Store<false>(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x1(const float * src, size_t srcS, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
        {
            float32x4_t sum = bias;
            sum = vmlaq_f32(sum, Load<false>(src + 0 * F), weight[0]);
            sum = vmlaq_f32(sum, Load<false>(src + 1 * F), weight[1]);
            sum = vmlaq_f32(sum, Load<false>(src + 2 * F), weight[2]);
            src += srcS;
            sum = vmlaq_f32(sum, Load<false>(src + 0 * F), weight[3]);
            sum = vmlaq_f32(sum, Load<false>(src + 1 * F), weight[4]);
            sum = vmlaq_f32(sum, Load<false>(src + 2 * F), weight[5]);
            src += srcS;
            sum = vmlaq_f32(sum, Load<false>(src + 0 * F), weight[6]);
            sum = vmlaq_f32(sum, Load<false>(src + 1 * F), weight[7]);
            sum = vmlaq_f32(sum, Load<false>(src + 2 * F), weight[8]);
            Store<false>(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4x2(const float * src, size_t srcS, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
        {
            float32x4_t sum0 = bias;
            float32x4_t sum1 = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                float32x4_t s0 = Load<false>(src + 0 * F);
                float32x4_t s1 = Load<false>(src + 1 * F);
                float32x4_t s2 = Load<false>(src + 2 * F);
                float32x4_t s3 = Load<false>(src + 3 * F);
                sum0 = vmlaq_f32(sum0, s0, weight[0]);
                sum1 = vmlaq_f32(sum1, s1, weight[0]);
                sum0 = vmlaq_f32(sum0, s1, weight[1]);
                sum1 = vmlaq_f32(sum1, s2, weight[1]);
                sum0 = vmlaq_f32(sum0, s2, weight[2]);
                sum1 = vmlaq_f32(sum1, s3, weight[2]);
                src += srcS;
                weight += 3;
            }
            Store<false>(dst + 0, Activate<type>(sum0, params, 0));
            Store<false>(dst + F, Activate<type>(sum1, params, 0));
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
                float32x4_t _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = Load<false>(weight + i * F);
                float32x4_t _bias = bias ? Load<false>(bias) : vdupq_n_f32(0.0f);
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
#endif// SIMD_NEON_ENABLE
}
