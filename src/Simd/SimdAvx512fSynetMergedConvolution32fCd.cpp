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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512f
    {
        namespace Cd
        {
            template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
                size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
            {
                size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
                size_t srcW = p.srcW * F, weightS = p.kernelY * p.kernelX * F, strideXF = strideX * F;
                size_t srcM = (bufH[0] - 1), srcS = bufH[0] * srcW, dstS = p.dstW * p.dstC;
                size_t noseY = (p.padY + p.strideY - 1) / p.strideY;
                size_t bodyY = (p.srcH + p.padY + p.strideY - p.kernelY) / p.strideY;
                size_t noseX = (p.padX + p.strideX - 1) / p.strideX;
                size_t bodyX = (p.srcW + p.padX + p.strideX - p.kernelX) / p.strideX;
                size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
                size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
                size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
                size_t srcCF = AlignLo(srcC, F);

                __m512 _params[2];
                _params[0] = _mm512_set1_ps(params[0]);
                if (type == SimdConvolutionActivationRestrictRange ||
                    type == SimdConvolutionActivationHswish ||
                    type == SimdConvolutionActivationHardSigmoid)
                    _params[1] = _mm512_set1_ps(params[1]);
                for (size_t c = 0; c < srcC; c += F)
                {
                    __m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                    if (type == ::SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + c);
                    __mmask16 tail = TailMask16(srcC - c);
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        float* pd = dst + dy * dstS;
                        if (dy >= noseY && dy < bodyY)
                        {
                            size_t dx = 0;
                            for (; dx < noseX; ++dx, pd += srcC)
                            {
                                __m512 sum = _bias;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * p.strideY + ky - padY;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                                    {
                                        size_t sx = dx * p.strideX + kx - padX;
                                        if (sx < p.srcW)
                                        {
                                            const float* pw = weight + (ky * p.kernelX + kx) * F;
                                            const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
                                            sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                                _mm512_mask_storeu_ps(pd, tail, Activate<type>(sum, _params, 0));
                            }
                            for (; dx < bodyX8; dx += 8, pd += 8 * srcC)
                            {
                                __m512 sum0 = _bias;
                                __m512 sum1 = _bias;
                                __m512 sum2 = _bias;
                                __m512 sum3 = _bias;
                                __m512 sum4 = _bias;
                                __m512 sum5 = _bias;
                                __m512 sum6 = _bias;
                                __m512 sum7 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m512 w0 = _mm512_loadu_ps(pw);
                                        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
                                        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
                                        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * strideXF), w0, sum2);
                                        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * strideXF), w0, sum3);
                                        sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 4 * strideXF), w0, sum4);
                                        sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 5 * strideXF), w0, sum5);
                                        sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 6 * strideXF), w0, sum6);
                                        sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 7 * strideXF), w0, sum7);
                                    }
                                }
                                _mm512_mask_storeu_ps(pd + 0 * srcC, tail, Activate<type>(sum0, _params, 0));
                                _mm512_mask_storeu_ps(pd + 1 * srcC, tail, Activate<type>(sum1, _params, 0));
                                _mm512_mask_storeu_ps(pd + 2 * srcC, tail, Activate<type>(sum2, _params, 0));
                                _mm512_mask_storeu_ps(pd + 3 * srcC, tail, Activate<type>(sum3, _params, 0));
                                _mm512_mask_storeu_ps(pd + 4 * srcC, tail, Activate<type>(sum4, _params, 0));
                                _mm512_mask_storeu_ps(pd + 5 * srcC, tail, Activate<type>(sum5, _params, 0));
                                _mm512_mask_storeu_ps(pd + 6 * srcC, tail, Activate<type>(sum6, _params, 0));
                                _mm512_mask_storeu_ps(pd + 7 * srcC, tail, Activate<type>(sum7, _params, 0));
                            }
                            for (; dx < bodyX4; dx += 4, pd += 4 * srcC)
                            {
                                __m512 sum0 = _bias;
                                __m512 sum1 = _bias;
                                __m512 sum2 = _bias;
                                __m512 sum3 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m512 w0 = _mm512_loadu_ps(pw);
                                        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
                                        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
                                        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * strideXF), w0, sum2);
                                        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * strideXF), w0, sum3);
                                    }
                                }
                                _mm512_mask_storeu_ps(pd + 0 * srcC, tail, Activate<type>(sum0, _params, 0));
                                _mm512_mask_storeu_ps(pd + 1 * srcC, tail, Activate<type>(sum1, _params, 0));
                                _mm512_mask_storeu_ps(pd + 2 * srcC, tail, Activate<type>(sum2, _params, 0));
                                _mm512_mask_storeu_ps(pd + 3 * srcC, tail, Activate<type>(sum3, _params, 0));
                            }
                            for (; dx < bodyX2; dx += 2, pd += 2 * srcC)
                            {
                                __m512 sum0 = _bias;
                                __m512 sum1 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m512 w0 = _mm512_loadu_ps(pw);
                                        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
                                        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
                                    }
                                }
                                _mm512_mask_storeu_ps(pd + 0 * srcC, tail, Activate<type>(sum0, _params, 0));
                                _mm512_mask_storeu_ps(pd + 1 * srcC, tail, Activate<type>(sum1, _params, 0));
                            }
                            for (; dx < bodyX; ++dx, pd += srcC)
                            {
                                __m512 sum = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m512 w0 = _mm512_loadu_ps(pw);
                                        sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), w0, sum);
                                    }
                                }
                                _mm512_mask_storeu_ps(pd, tail, Activate<type>(sum, _params, 0));
                            }
                            for (; dx < p.dstW; ++dx, pd += srcC)
                            {
                                __m512 sum = _bias;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                                    {
                                        size_t sx = dx * strideX + kx - padX;
                                        if (sx < p.srcW)
                                        {
                                            const float* pw = weight + (ky * p.kernelX + kx) * F;
                                            const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
                                            sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                                _mm512_mask_storeu_ps(pd, tail, Activate<type>(sum, _params, 0));
                            }
                        }
                        else
                        {
                            for (size_t dx = 0; dx < p.dstW; ++dx, pd += srcC)
                            {
                                __m512 sum = _bias;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    if (sy < p.srcH)
                                    {
                                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                                        {
                                            size_t sx = dx * strideX + kx - padX;
                                            if (sx < p.srcW)
                                            {
                                                const float* pw = weight + (ky * p.kernelX + kx) * F;
                                                const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
                                                sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                            }
                                        }
                                    }
                                }
                                _mm512_mask_storeu_ps(pd, tail, Activate<type>(sum, _params, 0));
                            }
                        }
                    }

                    src += srcS;
                    dst += F;
                    weight += weightS;
                }
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
                const float* src0, const float* src1, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps();
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
                _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
                const float* src0, const float* src1, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * F), weight[2], sum2);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * F), weight[5], sum2);
                _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
                const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps();
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * F), weight[6], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * F), weight[7], sum1);
                _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
                const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * F), weight[2], sum2);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * F), weight[5], sum2);
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * F), weight[6], sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * F), weight[7], sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 2 * F), weight[8], sum2);
                _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x2(
                const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst, size_t dstC, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = bias, s0;

                s0 = _mm512_loadu_ps(src0 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
                s0 = _mm512_loadu_ps(src0 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[1], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[0], sum1);
                s0 = _mm512_loadu_ps(src0 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[2], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[1], sum1);
                s0 = _mm512_loadu_ps(src0 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[2], sum1);

                s0 = _mm512_loadu_ps(src1 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[3], sum0);
                s0 = _mm512_loadu_ps(src1 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[4], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[3], sum1);
                s0 = _mm512_loadu_ps(src1 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[5], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[4], sum1);
                s0 = _mm512_loadu_ps(src1 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[5], sum1);

                s0 = _mm512_loadu_ps(src2 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[6], sum0);
                s0 = _mm512_loadu_ps(src2 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[7], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[6], sum1);
                s0 = _mm512_loadu_ps(src2 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[8], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[7], sum1);
                s0 = _mm512_loadu_ps(src2 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[8], sum1);

                _mm512_mask_storeu_ps(dst + 0 * dstC, tail, Activate<type>(sum0, params, 0));
                _mm512_mask_storeu_ps(dst + 1 * dstC, tail, Activate<type>(sum1, params, 0));
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x4(
                const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst, size_t dstC, __mmask16 tail)
            {
                __m512 sum0 = bias, sum1 = bias, sum2 = bias, sum3 = bias, s0;

                s0 = _mm512_loadu_ps(src0 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
                s0 = _mm512_loadu_ps(src0 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[1], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[0], sum1);
                s0 = _mm512_loadu_ps(src0 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[2], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[1], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[0], sum2);
                s0 = _mm512_loadu_ps(src0 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[2], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[1], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[0], sum3);
                s0 = _mm512_loadu_ps(src0 + 4 * F);
                sum2 = _mm512_fmadd_ps(s0, weight[2], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[1], sum3);
                s0 = _mm512_loadu_ps(src0 + 5 * F);
                sum3 = _mm512_fmadd_ps(s0, weight[2], sum3);

                s0 = _mm512_loadu_ps(src1 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[3], sum0);
                s0 = _mm512_loadu_ps(src1 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[4], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[3], sum1);
                s0 = _mm512_loadu_ps(src1 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[5], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[4], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[3], sum2);
                s0 = _mm512_loadu_ps(src1 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[5], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[4], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[3], sum3);
                s0 = _mm512_loadu_ps(src1 + 4 * F);
                sum2 = _mm512_fmadd_ps(s0, weight[5], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[4], sum3);
                s0 = _mm512_loadu_ps(src1 + 5 * F);
                sum3 = _mm512_fmadd_ps(s0, weight[5], sum3);

                s0 = _mm512_loadu_ps(src2 + 0 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[6], sum0);
                s0 = _mm512_loadu_ps(src2 + 1 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[7], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[6], sum1);
                s0 = _mm512_loadu_ps(src2 + 2 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[8], sum0);
                sum1 = _mm512_fmadd_ps(s0, weight[7], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[6], sum2);
                s0 = _mm512_loadu_ps(src2 + 3 * F);
                sum1 = _mm512_fmadd_ps(s0, weight[8], sum1);
                sum2 = _mm512_fmadd_ps(s0, weight[7], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[6], sum3);
                s0 = _mm512_loadu_ps(src2 + 4 * F);
                sum2 = _mm512_fmadd_ps(s0, weight[8], sum2);
                sum3 = _mm512_fmadd_ps(s0, weight[7], sum3);
                s0 = _mm512_loadu_ps(src2 + 5 * F);
                sum3 = _mm512_fmadd_ps(s0, weight[8], sum3);

                _mm512_mask_storeu_ps(dst + 0 * dstC, tail, Activate<type>(sum0, params, 0));
                _mm512_mask_storeu_ps(dst + 1 * dstC, tail, Activate<type>(sum1, params, 0));
                _mm512_mask_storeu_ps(dst + 2 * dstC, tail, Activate<type>(sum2, params, 0));
                _mm512_mask_storeu_ps(dst + 3 * dstC, tail, Activate<type>(sum3, params, 0));
            }

            template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
                size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
            {
                size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
                size_t srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
                size_t srcM = (bufH[0] - 1), srcS = bufH[0] * srcW, dstS = p.dstW * p.dstC;
                size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX) * F;
                size_t xMainEnd = p.dstW - p.padW, xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
                size_t yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

                __m512 _params[2];
                _params[0] = _mm512_set1_ps(params[0]);
                if (type == SimdConvolutionActivationRestrictRange ||
                    type == SimdConvolutionActivationHswish ||
                    type == SimdConvolutionActivationHardSigmoid)
                    _params[1] = _mm512_set1_ps(params[1]);
                for (size_t c = 0; c < srcC; c += F)
                {
                    __m512 _weight[9];
                    for (size_t i = 0; i < 9; ++i)
                        _weight[i] = _mm512_loadu_ps(weight + i * F);
                    __m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                    if (type == ::SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + c);
                    __mmask16 tail = TailMask16(srcC - c);

                    size_t dy = yBeg;
                    if (yBeg == 0 && padY)
                    {
                        size_t sy = 0, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        float* pDst = dst + dy * dstS;
                        if (padX)
                            ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst, tail), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0;
                        for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep)
                            ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 3, _bias, _params, pDst, tail);
                        if (padW)
                            ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 3, _bias, _params, pDst, tail);
                        dy++;
                    }
                    for (; dy < yMainEnd; ++dy)
                    {
                        size_t sy = dy * strideY - padY, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        const float* src2 = src + ((sy + 2) & srcM) * srcW;
                        float* pDst = dst + dy * dstS;
                        if (padX)
                            ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst, tail), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
                        for (; dx < xMainEnd2; dx += 2, pDst += 2 * p.dstC, src0 += 2 * xStep, src1 += 2 * xStep, src2 += 2 * xStep)
                            ConvolutionDepthwise3x3Main1x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst, srcC, tail);
                        for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep, src2 += xStep)
                            ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst, tail);
                        if (padW)
                            ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst, tail);
                    }
                    if (dy < yEnd)
                    {
                        size_t sy = dy * strideY - padY, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        float* pDst = dst + dy * dstS;
                        if (padX)
                            ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst, tail), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0;
                        for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep)
                            ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 0, _bias, _params, pDst, tail);
                        if (padW)
                            ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 0, _bias, _params, pDst, tail);
                    }
                    src += srcS;
                    dst += F;
                    weight += weightS;
                }
            }

            //---------------------------------------------------------------------

            template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32fCd::ConvolutionPtr* c)
            {
                switch (t)
                {
                case 1:
                    if (p.conv[i].kernelY == 3)
                        c[i] = DepthwiseConvolution3x3<type>;
                    else
                        c[i] = DepthwiseConvolution<type>;
                    break;
                default:
                    assert(0);
                }
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam32f& p)
            : Avx2::SynetMergedConvolution32fCd(p)
        {
            SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Avx512f::F);
            SynetMergedConvolution32fCdc::Set(_param, 0, 0, _convolution);
            SynetMergedConvolution32fCd::Set(_param, 1, 1, _convolution);
        }

        void SynetMergedConvolution32fCd::Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c)
        {
            switch (p.conv[i].activation)
            {
            case SimdConvolutionActivationIdentity: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
            case SimdConvolutionActivationRelu: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
            case SimdConvolutionActivationLeakyRelu: Cd::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
            case SimdConvolutionActivationRestrictRange: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
            case SimdConvolutionActivationPrelu: Cd::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
            case SimdConvolutionActivationElu: Cd::Set<SimdConvolutionActivationElu>(p, t, i, c); break;
            case SimdConvolutionActivationHswish: Cd::Set<SimdConvolutionActivationHswish>(p, t, i, c); break;
            case SimdConvolutionActivationMish: Cd::Set<SimdConvolutionActivationMish>(p, t, i, c); break;
            case SimdConvolutionActivationHardSigmoid: Cd::Set<SimdConvolutionActivationHardSigmoid>(p, t, i, c); break;
            case SimdConvolutionActivationSwish: Cd::Set<SimdConvolutionActivationSwish>(p, t, i, c); break;
            default: assert(0);
            }
        }
    }
#endif//SIMD_AVX512f_ENABLE
}
