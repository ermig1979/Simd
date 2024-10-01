/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwiseDefault(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
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

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Edge(const float * src, const ConvParam & p, size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main1(const float * src, size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main2(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main4(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Edge4(const float * src, const ConvParam & p, size_t dy, size_t dx, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main4x1(const float * src, size_t srcS, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
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
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main4x2(const float * src, size_t srcS, const float32x4_t * weight, float32x4_t bias, const float * params, float * dst)
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

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise3x3(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
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
                        Convolution32fNhwcDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW2; dx += 2)
                        Convolution32fNhwcDepthwise3x3Main4x2<type>(src + offset, srcS, _weight, _bias, params, dst), offset += 2 * F, dst += 2 * F;
                    for (; dx < dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Main4x1<type>(src + offset, srcS, _weight, _bias, params, dst), offset += F, dst += F;
                    for (; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge4<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
            }
            else
            {
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW4; dx += 4)
                        Convolution32fNhwcDepthwise3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * p.dstC, offset += 4 * srcX;
                    for (; dx < dstW2; dx += 2)
                        Convolution32fNhwcDepthwise3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * p.dstC, offset += 2 * srcX;
                    for (; dx < dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    for (; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <::SimdConvolutionActivationType type> SynetConvolution32fNhwcDepthwise::ConvolutionPtr Get(const ConvParam & p)
        {
            if (p.IsKernel(3) && p.IsDilation(1))
                return Convolution32fNhwcDepthwise3x3<type>;
            else
                return Convolution32fNhwcDepthwiseDefault<type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fNhwcDepthwise::SynetConvolution32fNhwcDepthwise(const ConvParam& p)
            : Base::SynetConvolution32fNhwcDepthwise(p)
        {
            if (p.dstC >= F && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: _convolution = Get<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: _convolution = Get<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: _convolution = Get<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: _convolution = Get<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: _convolution = Get<::SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: _convolution = Get<::SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: _convolution = Get<::SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: _convolution = Get<::SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: _convolution = Get<::SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: _convolution = Get<::SimdConvolutionActivationSwish>(p); break;
                case ::SimdConvolutionActivationGelu: _convolution = Get<::SimdConvolutionActivationGelu>(p); break;
                }
            }
        }
    }
#endif
}
