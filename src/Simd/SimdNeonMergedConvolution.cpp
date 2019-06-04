/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdMergedConvolution.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE)
    namespace Neon
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float32x4_t Activate(float32x4_t value, const float32x4_t * params, size_t index);

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationIdentity>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmaxq_f32(vdupq_n_f32(0.0f), value);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationLeakyRelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[0], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRestrictRange>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vminq_f32(vmaxq_f32(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationPrelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[index], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x6(const float * src0, size_t srcC,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst0, float * dst1)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            d00 = bias[0], d01 = bias[1];
            d10 = bias[0], d11 = bias[1];
            d20 = bias[0], d21 = bias[1];
            d30 = bias[0], d31 = bias[1];
            d40 = bias[0], d41 = bias[1];
            d50 = bias[0], d51 = bias[1];
            const float * src1 = src0 + 1 * srcC;
            const float * src2 = src0 + 2 * srcC;
            const float * src3 = src0 + 3 * srcC;
            const float * src4 = src0 + 4 * srcC;
            const float * src5 = src0 + 5 * srcC;
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = Load<false>(weight + 0);
                w1 = Load<false>(weight + F);
                s0 = vdupq_n_f32(src0[sc]);
                d00 = vmlaq_f32(d00, s0, w0);
                d01 = vmlaq_f32(d01, s0, w1);
                s0 = vdupq_n_f32(src1[sc]);
                d10 = vmlaq_f32(d10, s0, w0);
                d11 = vmlaq_f32(d11, s0, w1);
                s0 = vdupq_n_f32(src2[sc]);
                d20 = vmlaq_f32(d20, s0, w0);
                d21 = vmlaq_f32(d21, s0, w1);
                s0 = vdupq_n_f32(src3[sc]);
                d30 = vmlaq_f32(d30, s0, w0);
                d31 = vmlaq_f32(d31, s0, w1);
                s0 = vdupq_n_f32(src4[sc]);
                d40 = vmlaq_f32(d40, s0, w0);
                d41 = vmlaq_f32(d41, s0, w1);
                s0 = vdupq_n_f32(src5[sc]);
                d50 = vmlaq_f32(d50, s0, w0);
                d51 = vmlaq_f32(d51, s0, w1);
                weight += DF;
            }
            Store<false>(dst0 + 0 * F, Activate<type>(d00, params, 0));
            Store<false>(dst1 + 0 * F, Activate<type>(d01, params, 1));
            Store<false>(dst0 + 1 * F, Activate<type>(d10, params, 0));
            Store<false>(dst1 + 1 * F, Activate<type>(d11, params, 1));
            Store<false>(dst0 + 2 * F, Activate<type>(d20, params, 0));
            Store<false>(dst1 + 2 * F, Activate<type>(d21, params, 1));
            Store<false>(dst0 + 3 * F, Activate<type>(d30, params, 0));
            Store<false>(dst1 + 3 * F, Activate<type>(d31, params, 1));
            Store<false>(dst0 + 4 * F, Activate<type>(d40, params, 0));
            Store<false>(dst1 + 4 * F, Activate<type>(d41, params, 1));
            Store<false>(dst0 + 5 * F, Activate<type>(d50, params, 0));
            Store<false>(dst1 + 5 * F, Activate<type>(d51, params, 1));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x1(const float * src0, size_t srcC,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst0, float * dst1)
        {
            float32x4_t d00, d01, s0, w0, w1;
            d00 = bias[0];
            d01 = bias[1];
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = Load<false>(weight + 0);
                w1 = Load<false>(weight + F);
                s0 = vdupq_n_f32(src0[sc]);
                d00 = vmlaq_f32(d00, s0, w0);
                d01 = vmlaq_f32(d01, s0, w1);
                weight += DF;
            }
            Store<false>(dst0, Activate<type>(d00, params, 0));
            Store<false>(dst1, Activate<type>(d01, params, 1));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_1x6(const float * src0, size_t srcC,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst0)
        {
            float32x4_t d00, d10, d20, d30, d40, d50, s0, w0;
            d00 = bias[0];
            d10 = bias[0];
            d20 = bias[0];
            d30 = bias[0];
            d40 = bias[0];
            d50 = bias[0];
            const float * src1 = src0 + 1 * srcC;
            const float * src2 = src0 + 2 * srcC;
            const float * src3 = src0 + 3 * srcC;
            const float * src4 = src0 + 4 * srcC;
            const float * src5 = src0 + 5 * srcC;
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = Load<false>(weight + 0);
                s0 = vdupq_n_f32(src0[sc]);
                d00 = vmlaq_f32(d00, s0, w0);
                s0 = vdupq_n_f32(src1[sc]);
                d10 = vmlaq_f32(d10, s0, w0);
                s0 = vdupq_n_f32(src2[sc]);
                d20 = vmlaq_f32(d20, s0, w0);
                s0 = vdupq_n_f32(src3[sc]);
                d30 = vmlaq_f32(d30, s0, w0);
                s0 = vdupq_n_f32(src4[sc]);
                d40 = vmlaq_f32(d40, s0, w0);
                s0 = vdupq_n_f32(src5[sc]);
                d50 = vmlaq_f32(d50, s0, w0);
                weight += DF;
            }
            Store<false>(dst0 + 0 * F, Activate<type>(d00, params, 0));
            Store<false>(dst0 + 1 * F, Activate<type>(d10, params, 0));
            Store<false>(dst0 + 2 * F, Activate<type>(d20, params, 0));
            Store<false>(dst0 + 3 * F, Activate<type>(d30, params, 0));
            Store<false>(dst0 + 4 * F, Activate<type>(d40, params, 0));
            Store<false>(dst0 + 5 * F, Activate<type>(d50, params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_1x1(const float * src0, size_t srcC,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst0)
        {
            float32x4_t d00, s0, w0;
            d00 = bias[0];
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = Load<false>(weight + 0);
                s0 = vdupq_n_f32(src0[sc]);
                d00 = vmlaq_f32(d00, s0, w0);
                weight += DF;
            }
            Store<false>(dst0, Activate<type>(d00, params, 0));
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW *F;
            size_t dstCDF = AlignLo(dstC, DF), dstW6 = AlignLoAny(dstW, 6);
            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = vdupq_n_f32(params[1]);

            size_t dc = 0;
            for (; dc < dstC; dc += DF)
            {
                _bias[0] = bias ? Load<false>(bias + dc + 0) : vdupq_n_f32(0.0f);
                _bias[1] = bias ? Load<false>(bias + dc + F) : vdupq_n_f32(0.0f);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                const float * pS = src + yBeg * srcW*srcC;
                const float * pW = weight + dc * srcC;
                float * pD = dst + (dc / F)*dstS;
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    float * dst0 = pD + (dy&dstM)*dstW*F;
                    size_t dx = 0;
                    if (dstC - dc > F)
                    {
                        for (; dx < dstW6; dx += 6, pS += 6 * srcC, dst0 += 6 * F)
                            InputConvolution1x1_2x6<type>(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS);
                        for (; dx < dstW; dx += 1, pS += srcC, dst0 += F)
                            InputConvolution1x1_2x1<type>(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS);
                    }
                    else
                    {
                        for (; dx < dstW6; dx += 6, pS += 6 * srcC, dst0 += 6 * F)
                            InputConvolution1x1_1x6<type>(pS, srcC, pW, _bias, _params, dst0);
                        for (; dx < dstW; dx += 1, pS += srcC, dst0 += F)
                            InputConvolution1x1_1x1<type>(pS, srcC, pW, _bias, _params, dst0);
                    }
                }
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float * params, size_t offset);

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<SimdConvolutionActivationType type> void InputConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW *F;
            size_t dstCDF = AlignLo(dstC, DF);
            if (dstC - F > dstCDF)
                dstCDF += DF;

            size_t dy = yBeg;
            if (yBeg == 0 && padY)
            {

            }
            for (; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
#if 1
                    size_t dc = 0;
                    for (; dc < dstCDF; dc += DF)
                    {
                        float buf[DF];
                        if (bias)
                            memcpy(buf, bias + dc, 2 * F * sizeof(float));
                        else
                            memset(buf, 0, 2 * F * sizeof(float));
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*kernelX + kx)*srcC*DF + dc * kernelY*kernelX*srcC;
                                        const float * ps = src + (sy*srcW + sx)*srcC;
                                        for (size_t sc = 0; sc < srcC; ++sc)
                                        {
                                            for (size_t i = 0; i < DF; ++i)
                                                buf[i] += ps[sc] * pw[i];
                                            pw += DF;
                                        }
                                    }
                                }
                            }
                        }
                        float * dst0 = dst + ((dy&dstM)*dstW + dx)*F + (dc / F)*dstS, *dst1 = dst0 + dstS;
                        for (size_t i = 0; i < F; ++i)
                        {
                            dst0[i] = Activate<type>(buf[i + 0], params, dc + i + 0);
                            dst1[i] = Activate<type>(buf[i + F], params, dc + i + F);
                        }
                    }
                    for (; dc < dstC; dc += F)
                    {
                        size_t n = Simd::Min(F, dstC - dc);
                        float buf[F];
                        if (bias)
                            memcpy(buf, bias + dc, n * sizeof(float));
                        else
                            memset(buf, 0, n * sizeof(float));
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*kernelX + kx)*srcC*DF + dc * kernelY*kernelX*srcC;
                                        const float * ps = src + (sy*srcW + sx)*srcC;
                                        for (size_t sc = 0; sc < srcC; ++sc)
                                        {
                                            for (size_t i = 0; i < n; ++i)
                                                buf[i] += ps[sc] * pw[i];
                                            pw += DF;
                                        }
                                    }
                                }
                            }
                        }
                        float * dst0 = dst + ((dy&dstM)*dstW + dx)*F + dc * dstS / F;
                        for (size_t i = 0; i < n; ++i)
                            dst0[i] = Activate<type>(buf[i + 0], params, dc + i + 0);
                    }
#else
                    Array32f buf(dstC);
                    if (bias)
                        memcpy(buf.data, bias, dstC * sizeof(float));
                    else
                        memset(buf.data, 0, dstC * sizeof(float));
                    for (size_t ky = 0; ky < kernelY; ++ky)
                    {
                        size_t sy = dy * strideY + ky - padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; ++kx)
                            {
                                size_t sx = dx * strideX + kx - padX;
                                if (sx < p.srcW)
                                {
                                    const float * pw = weight + (ky*kernelX + kx)*srcC*dstC;
                                    const float * ps = src + (sy*srcW + sx)*srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += dstC;
                                    }
                                }
                            }
                        }
                    }
                    float * pDst = dst + ((dy&dstM)*dstW + dx)*F;
                    for (size_t dc = 0; dc < dstC; dc += F, pDst += dstS)
                        for (size_t i = 0, n = Simd::Min(F, dstC - dc); i < n; ++i)
                            pDst[i] = Activate<type>(buf[dc + i], params, dc + i);
#endif
                }
            }
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
            const float * src0, const float * src1, const float32x4_t * weight, const float32x4_t & bias, const float32x4_t * params, float * dst)
        {
            float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
            sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
            sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
            sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
            sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
            Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
            const float * src0, const float * src1, const float32x4_t * weight, const float32x4_t & bias, const float32x4_t * params, float * dst)
        {
            float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
            sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
            sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
            sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * F), weight[2]);
            sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
            sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
            sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * F), weight[5]);
            Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
            const float * src0, const float * src1, const float * src2, const float32x4_t * weight, const float32x4_t & bias, const float32x4_t * params, float * dst)
        {
            float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
            sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
            sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
            sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
            sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
            sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * F), weight[6]);
            sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * F), weight[7]);
            Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
            const float * src0, const float * src1, const float * src2, const float32x4_t * weight, const float32x4_t & bias, const float32x4_t * params, float * dst)
        {
            float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
            sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
            sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
            sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * F), weight[2]);
            sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
            sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
            sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * F), weight[5]);
            sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * F), weight[6]);
            sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * F), weight[7]);
            sum2 = vmlaq_f32(sum2, Load<false>(src2 + 2 * F), weight[8]);
            Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t srcC = p.srcC, srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
            size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
            size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX)*F;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            float32x4_t _params[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = vdupq_n_f32(params[1]);
            for (size_t c = 0; c < srcC; c += F)
            {
                float32x4_t _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = Load<false>(weight + i * F);
                float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
                        ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 3, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 3, _bias, _params, pDst);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    const float * src2 = src + ((sy + 2)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep, src2 += xStep)
                        ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
                        ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 0, _bias, _params, pDst);
                }
                src += srcS;
                dst += dstS;
                weight += weightS;
            }
        }

        template <UpdateType update> SIMD_INLINE void Update(float  * p, float32x4_t a, size_t tail);

        template <> SIMD_INLINE void Update<UpdateSet>(float  * p, float32x4_t a, size_t tail)
        {
            float t[F];
            Store<false>(t, a);
            for (size_t i = 0; i < tail; ++i)
                p[i] = t[i];
        }

        template <> SIMD_INLINE void Update<UpdateAdd>(float  * p, float32x4_t a, size_t tail)
        {
            float t[F];
            for (size_t i = 0; i < tail; ++i)
                t[i] = p[i];
            Store<false>(t, vaddq_f32(a, Load<false>(t)));
            for (size_t i = 0; i < tail; ++i)
                p[i] = t[i];
        }

        template<SimdConvolutionActivationType type, UpdateType update> SIMD_INLINE void OutputConvolution_2x6(const float * src, size_t srcC, size_t srcS,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst, size_t dstC, size_t tail)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            d00 = bias[0], d01 = bias[1];
            d10 = bias[0], d11 = bias[1];
            d20 = bias[0], d21 = bias[1];
            d30 = bias[0], d31 = bias[1];
            d40 = bias[0], d41 = bias[1];
            d50 = bias[0], d51 = bias[1];
            for (size_t c = 0; c < srcC; c += F)
            {
                size_t n = Simd::Min(F, srcC - c);
                for (size_t i = 0; i < n; ++i, weight += DF)
                {
                    w0 = Load<false>(weight + 0);
                    w1 = Load<false>(weight + F);
                    s0 = vdupq_n_f32(src[i + 0 * F]);
                    d00 = vmlaq_f32(d00, s0, w0);
                    d01 = vmlaq_f32(d01, s0, w1);
                    s0 = vdupq_n_f32(src[i + 1 * F]);
                    d10 = vmlaq_f32(d10, s0, w0);
                    d11 = vmlaq_f32(d11, s0, w1);
                    s0 = vdupq_n_f32(src[i + 2 * F]);
                    d20 = vmlaq_f32(d20, s0, w0);
                    d21 = vmlaq_f32(d21, s0, w1);
                    s0 = vdupq_n_f32(src[i + 3 * F]);
                    d30 = vmlaq_f32(d30, s0, w0);
                    d31 = vmlaq_f32(d31, s0, w1);
                    s0 = vdupq_n_f32(src[i + 4 * F]);
                    d40 = vmlaq_f32(d40, s0, w0);
                    d41 = vmlaq_f32(d41, s0, w1);
                    s0 = vdupq_n_f32(src[i + 5 * F]);
                    d50 = vmlaq_f32(d50, s0, w0);
                    d51 = vmlaq_f32(d51, s0, w1);
                }
                src += srcS;
            }
            if (tail == F)
            {
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
                Update<update, false>(dst + F, Activate<type>(d01, params, 1));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d10, params, 0));
                Update<update, false>(dst + F, Activate<type>(d11, params, 1));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d20, params, 0));
                Update<update, false>(dst + F, Activate<type>(d21, params, 1));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d30, params, 0));
                Update<update, false>(dst + F, Activate<type>(d31, params, 1));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d40, params, 0));
                Update<update, false>(dst + F, Activate<type>(d41, params, 1));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d50, params, 0));
                Update<update, false>(dst + F, Activate<type>(d51, params, 1));
            }
            else
            {
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
                Update<update>(dst + F, Activate<type>(d01, params, 1), tail);
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d10, params, 0));
                Update<update>(dst + F, Activate<type>(d11, params, 1), tail);
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d20, params, 0));
                Update<update>(dst + F, Activate<type>(d21, params, 1), tail);
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d30, params, 0));
                Update<update>(dst + F, Activate<type>(d31, params, 1), tail);
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d40, params, 0));
                Update<update>(dst + F, Activate<type>(d41, params, 1), tail);
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d50, params, 0));
                Update<update>(dst + F, Activate<type>(d51, params, 1), tail);
            }
        }

        template<SimdConvolutionActivationType type, UpdateType update> SIMD_INLINE void OutputConvolution_2x1(const float * src, size_t srcC, size_t srcS,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst, size_t dstC, size_t tail)
        {
            float32x4_t d00, d01, s0, w0, w1;
            d00 = bias[0];
            d01 = bias[1];
            for (size_t c = 0; c < srcC; c += F)
            {
                size_t n = Simd::Min(F, srcC - c);
                for (size_t i = 0; i < n; ++i, weight += DF)
                {
                    w0 = Load<false>(weight + 0);
                    w1 = Load<false>(weight + F);
                    s0 = vdupq_n_f32(src[i]);
                    d00 = vmlaq_f32(d00, s0, w0);
                    d01 = vmlaq_f32(d01, s0, w1);
                }
                src += srcS;
            }
            if (tail == F)
            {
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
                Update<update, false>(dst + F, Activate<type>(d01, params, 1));
            }
            else
            {
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
                Update<update>(dst + F, Activate<type>(d01, params, 1), tail);
            }
        }

        template<SimdConvolutionActivationType type, UpdateType update> SIMD_INLINE void OutputConvolution_1x6(const float * src, size_t srcC, size_t srcS,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst, size_t dstC, size_t tail)
        {
            float32x4_t d00, d10, d20, d30, d40, d50, s0, w0;
            d00 = bias[0];
            d10 = bias[0];
            d20 = bias[0];
            d30 = bias[0];
            d40 = bias[0];
            d50 = bias[0];
            for (size_t c = 0; c < srcC; c += F)
            {
                size_t n = Simd::Min(F, srcC - c);
                for (size_t i = 0; i < n; ++i, weight += DF)
                {
                    w0 = Load<false>(weight + 0);
                    s0 = vdupq_n_f32(src[i + 0 * F]);
                    d00 = vmlaq_f32(d00, s0, w0);
                    s0 = vdupq_n_f32(src[i + 1 * F]);
                    d10 = vmlaq_f32(d10, s0, w0);
                    s0 = vdupq_n_f32(src[i + 2 * F]);
                    d20 = vmlaq_f32(d20, s0, w0);
                    s0 = vdupq_n_f32(src[i + 3 * F]);
                    d30 = vmlaq_f32(d30, s0, w0);
                    s0 = vdupq_n_f32(src[i + 4 * F]);
                    d40 = vmlaq_f32(d40, s0, w0);
                    s0 = vdupq_n_f32(src[i + 5 * F]);
                    d50 = vmlaq_f32(d50, s0, w0);
                }
                src += srcS;
            }
            if (tail == F)
            {
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d10, params, 0));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d20, params, 0));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d30, params, 0));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d40, params, 0));
                dst += dstC;
                Update<update, false>(dst + 0, Activate<type>(d50, params, 0));
            }
            else
            {
                Update<update>(dst + 0, Activate<type>(d00, params, 0), tail);
                dst += dstC;
                Update<update>(dst + 0, Activate<type>(d10, params, 0), tail);
                dst += dstC;
                Update<update>(dst + 0, Activate<type>(d20, params, 0), tail);
                dst += dstC;
                Update<update>(dst + 0, Activate<type>(d30, params, 0), tail);
                dst += dstC;
                Update<update>(dst + 0, Activate<type>(d40, params, 0), tail);
                dst += dstC;
                Update<update>(dst + 0, Activate<type>(d50, params, 0), tail);
            }
        }

        template<SimdConvolutionActivationType type, UpdateType update> SIMD_INLINE void OutputConvolution_1x1(const float * src, size_t srcC, size_t srcS,
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * dst, size_t dstC, size_t tail)
        {
            float32x4_t d00, s0, w0;
            d00 = bias[0];
            for (size_t c = 0; c < srcC; c += F)
            {
                size_t n = Simd::Min(F, srcC - c);
                for (size_t i = 0; i < n; ++i, weight += DF)
                {
                    w0 = Load<false>(weight + 0);
                    s0 = vdupq_n_f32(src[i]);
                    d00 = vmlaq_f32(d00, s0, w0);
                }
                src += srcS;
            }
            if (tail == F)
                Update<update, false>(dst + 0, Activate<type>(d00, params, 0));
            else
                Update<update>(dst + 0, Activate<type>(d00, params, 0), tail);
        }

        template<SimdConvolutionActivationType type, UpdateType update> void OutputConvolution(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t sC = (srcC + F - 1) / F, srcM = (bufH[1] - 1), srcS = bufH[1] * srcW*F;
            size_t dstCDF = AlignLo(dstC, DF), dstW6 = AlignLoAny(dstW, 6);
            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = vdupq_n_f32(params[1]);

            dst += yBeg * p.dstW * p.dstC;
            size_t dc = 0;
            for (; dc < dstC; dc += DF)
            {
                _bias[0] = bias ? Load<false>(bias + dc + 0) : vdupq_n_f32(0.0f);
                _bias[1] = bias ? Load<false>(bias + dc + F) : vdupq_n_f32(0.0f);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                float * pDst = dst + dc;
                for (size_t y = yBeg; y < yEnd; ++y)
                {
                    const float * pSrc = src + (y&srcM)*srcW*F;
                    size_t x = 0;
                    if (dc < dstCDF)
                    {
                        for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
                            OutputConvolution_2x6<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, F);
                        for (; x < dstW; ++x, pDst += dstC, pSrc += F)
                            OutputConvolution_2x1<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, F);
                    }
                    else if (dstC - dstCDF > F)
                    {
                        size_t tail = dstC - dstCDF - F;
                        for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
                            OutputConvolution_2x6<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
                        for (; x < dstW; ++x, pDst += dstC, pSrc += F)
                            OutputConvolution_2x1<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
                    }
                    else
                    {
                        size_t tail = dstC - dstCDF;
                        for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
                            OutputConvolution_1x6<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
                        for (; x < dstW; ++x, pDst += dstC, pSrc += F)
                            OutputConvolution_1x1<type, update>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
                    }
                }
                weight += srcC * DF;
            }
        }

        void InputOutputReorder(const float * src, const SimdConvolutionParameters & p, float * dst)
        {
            size_t size = p.kernelY*p.kernelX*p.srcC, dstC = p.dstC;
            for (size_t c = 0; c < dstC; c += DF)
            {
                size_t n = Simd::Min(DF, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < DF; ++i)
                        dst[i] = 0;
                    dst += DF;
                }
            }
        }

        void DepthwiseReorder(const float * src, const SimdConvolutionParameters & p, float * dst)
        {
            size_t dstC = p.dstC, size = p.kernelY*p.kernelX;
            for (size_t c = 0; c < dstC; c += F)
            {
                size_t n = Simd::Min(F, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < F; ++i)
                        dst[i] = 0;
                    dst += F;
                }
            }
        }

        template <SimdConvolutionActivationType type> void SetConvolutionPtr(const MergConvParam & p, size_t index, MergedConvolution::ConvolutionPtr convolution[3])
        {
            switch (index)
            {
            case 0:
                if (p.conv[0].kernelY == 1 && p.conv[0].strideY == 1)
                    convolution[0] = InputConvolution1x1<type>;
                else
                    convolution[0] = InputConvolution<type>;
                break;
            case 1:
                convolution[1] = DepthwiseConvolution3x3<type>;
                break;
            case 2:
                if (p.add)
                    convolution[2] = OutputConvolution<type, UpdateAdd>;
                else
                    convolution[2] = OutputConvolution<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Base::MergedConvolution(p, false)
        {
            const size_t L1 = 32 * 1024, L2 = 256 * 1024, L3 = 2048 * 1024;
            SetSize(L2, Neon::F);
            for (size_t i = 0; i < _param.count; ++i)
            {
                _reorder[i] = NULL;
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: SetConvolutionPtr<SimdConvolutionActivationIdentity>(_param, i, _convolution); break;
                case SimdConvolutionActivationRelu: SetConvolutionPtr<SimdConvolutionActivationRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: SetConvolutionPtr<SimdConvolutionActivationLeakyRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationRestrictRange: SetConvolutionPtr<SimdConvolutionActivationRestrictRange>(_param, i, _convolution); break;
                case SimdConvolutionActivationPrelu: SetConvolutionPtr<SimdConvolutionActivationPrelu>(_param, i, _convolution); break;
                default: assert(0);
                }
            }
            _rWeight[0].Resize(AlignHi(p.conv[0].dstC, DF)*p.conv[0].kernelY*p.conv[0].kernelX*p.conv[0].srcC);
            _reorder[0] = InputOutputReorder;
            _rWeight[1].Resize(AlignHi(p.conv[1].dstC, F)*p.conv[1].kernelY*p.conv[1].kernelX);
            _reorder[1] = DepthwiseReorder;
            _rWeight[2].Resize(AlignHi(p.conv[2].dstC, DF)*p.conv[2].kernelY*p.conv[2].kernelX*p.conv[2].srcC);
            _reorder[2] = InputOutputReorder;
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new Neon::MergedConvolution(param);
        }
    }
 #endif//SIMD_NEON_ENABLE
}
