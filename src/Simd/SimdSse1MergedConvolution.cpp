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
#if defined(SIMD_SSE_ENABLE)
    namespace Sse
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const __m128 * params, size_t index);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const __m128 * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_min_ps(_mm_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[index], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x6(const float * src0, size_t srcC,
            const float * weight, const __m128 * bias, const __m128 * params, float * dst0, float * dst1)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
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
                w0 = _mm_loadu_ps(weight + 0);
                w1 = _mm_loadu_ps(weight + F);
                s0 = _mm_set1_ps(src0[sc]);
                d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
                s0 = _mm_set1_ps(src1[sc]);
                d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
                d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
                s0 = _mm_set1_ps(src2[sc]);
                d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
                d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
                s0 = _mm_set1_ps(src3[sc]);
                d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
                d31 = _mm_add_ps(_mm_mul_ps(s0, w1), d31);
                s0 = _mm_set1_ps(src4[sc]);
                d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
                d41 = _mm_add_ps(_mm_mul_ps(s0, w1), d41);
                s0 = _mm_set1_ps(src5[sc]);
                d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
                d51 = _mm_add_ps(_mm_mul_ps(s0, w1), d51);
                weight += DF;
            }
            _mm_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
            _mm_storeu_ps(dst1 + 0 * F, Activate<type>(d01, params, 1));
            _mm_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
            _mm_storeu_ps(dst1 + 1 * F, Activate<type>(d11, params, 1));
            _mm_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
            _mm_storeu_ps(dst1 + 2 * F, Activate<type>(d21, params, 1));
            _mm_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
            _mm_storeu_ps(dst1 + 3 * F, Activate<type>(d31, params, 1));
            _mm_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
            _mm_storeu_ps(dst1 + 4 * F, Activate<type>(d41, params, 1));
            _mm_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
            _mm_storeu_ps(dst1 + 5 * F, Activate<type>(d51, params, 1));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x1(const float * src0, size_t srcC, 
            const float * weight, const __m128 * bias, const __m128 * params, float * dst0, float * dst1)
        {
            __m128 d00, d01, s0, w0, w1;
            d00 = bias[0];
            d01 = bias[1];
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = _mm_loadu_ps(weight + 0);
                w1 = _mm_loadu_ps(weight + F);
                s0 = _mm_set1_ps(src0[sc]);
                d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
                weight += DF;
            }
            _mm_storeu_ps(dst0, Activate<type>(d00, params, 0));
            _mm_storeu_ps(dst1, Activate<type>(d01, params, 1));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_1x6(const float * src0, size_t srcC,
            const float * weight, const __m128 * bias, const __m128 * params, float * dst0)
        {
            __m128 d00, d10, d20, d30, d40, d50, s0, w0;
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
                w0 = _mm_loadu_ps(weight + 0);
                s0 = _mm_set1_ps(src0[sc]);
                d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                s0 = _mm_set1_ps(src1[sc]);
                d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
                s0 = _mm_set1_ps(src2[sc]);
                d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
                s0 = _mm_set1_ps(src3[sc]);
                d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
                s0 = _mm_set1_ps(src4[sc]);
                d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
                s0 = _mm_set1_ps(src5[sc]);
                d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
                weight += DF;
            }
            _mm_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
            _mm_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
            _mm_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
            _mm_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
            _mm_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
            _mm_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_1x1(const float * src0, size_t srcC,
            const float * weight, const __m128 * bias, const __m128 * params, float * dst0)
        {
            __m128 d00, s0, w0;
            d00 = bias[0];
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = _mm_loadu_ps(weight + 0);
                s0 = _mm_set1_ps(src0[sc]);
                d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                weight += DF;
            }
            _mm_storeu_ps(dst0, Activate<type>(d00, params, 0));
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW *F;
            size_t dstCDF = AlignLo(dstC, DF), dstW6 = AlignLoAny(dstW, 6);
            __m128 _params[2], _bias[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm_set1_ps(params[1]);

            size_t dc = 0;
            for (; dc < dstC; dc += DF)
            {
                _bias[0] = bias ? _mm_loadu_ps(bias + dc + 0) : _mm_setzero_ps();
                _bias[1] = bias ? _mm_loadu_ps(bias + dc + F) : _mm_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                const float * pS = src + yBeg*srcW*srcC;
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

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const float * params, size_t offset);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_set1_ps(params[0]), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const float * params, size_t offset)
        {
            return _mm_min_ps(_mm_max_ps(_mm_set1_ps(params[0]), value), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_loadu_ps(params + offset), _mm_min_ps(_mm_setzero_ps(), value)));
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
                        float * dst0 = dst + ((dy&dstM)*dstW + dx)*F + (dc/F)*dstS, * dst1 = dst0 + dstS;
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
            const float * src0, const float * src1, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
            const float * src0, const float * src1, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
            const float * src0, const float * src1, const float * src2, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
            const float * src0, const float * src1, const float * src2, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * F), weight[8]), sum2);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, 0));
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t srcC = p.srcC, srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
            size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
            size_t xStep = F * p.strideX, xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm_set1_ps(params[1]);
            for (size_t c = 0; c < srcC; c += F)
            {
                __m128 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm_loadu_ps(weight + i * F);
                __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_loadu_ps(params + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst), pDst += F, dx++;
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
                        ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst), pDst += F, dx++;
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
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst), pDst += F, dx++;
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

        void InputReorder(const float * src, const SimdConvolutionParameters & p, float * dst)
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
                if(p.conv[0].kernelY == 1 && p.conv[0].strideY == 1)
                    convolution[0] = InputConvolution1x1<type>;
                else
                    convolution[0] = InputConvolution<type>;
                break;
            case 1:
                convolution[1] = DepthwiseConvolution3x3<type>;
                break;
            case 2:
                //if (p.add)
                //    convolution[2] = old ? DirectConvolutionOld<type, UpdateAdd> : OutputConvolution<type, UpdateAdd>;
                //else
                //    convolution[2] = old ? DirectConvolutionOld<type, UpdateSet> : OutputConvolution<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Base::MergedConvolution(p, false)
        {
            const size_t L1 = 32 * 1024, L2 = 256 * 1024, L3 = 2048 * 1024;
            SetSize(L2, Sse::F);
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
            _reorder[0] = InputReorder;
            _rWeight[1].Resize(AlignHi(p.conv[1].dstC, F)*p.conv[1].kernelY*p.conv[1].kernelX);
            _reorder[1] = DepthwiseReorder;
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new Sse::MergedConvolution(param);
        }
    }
 #endif//SIMD_SSE_ENABLE
}
