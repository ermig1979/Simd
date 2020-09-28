/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE)
    namespace Sse41
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using Convert8uTo32fPtr = Base::SynetMergedConvolution8i::Convert8uTo32fPtr;
        using Convert32fTo8uPtr = Base::SynetMergedConvolution8i::Convert32fTo8uPtr;
        using InputConvolutionPtr = Base::SynetMergedConvolution8i::InputConvolutionPtr;
        using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;
        using OutputConvolutionPtr = Base::SynetMergedConvolution8i::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput1(float* dst, __m128i sum, const __m128* norm, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps((float*)dst, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), norm[0]), bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps(dst0, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum0), norm[0]), bias[0]), params, 0));
            _mm_storeu_ps(dst1, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum1), norm[1]), bias[1]), params, 1));
        }

        template<bool overflow, SimdConvolutionActivationType type> void InputConvolution_2x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, float * dst0, float* dst1)
        {
            __m128i d00, d01, s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 4) * DA;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    Madd4<overflow>(d00, s0, w0);
                                    Madd4<overflow>(d01, s0, w1);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput2<type>(dst0, dst1, d00, d01, norm, bias, params);
            }
            else
            {
                d00 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    Madd4<overflow>(d00, s0, w0);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput1<type>(dst0, d00, norm, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, 
            size_t dstC, const int8_t* weight, const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2x1(const ConvParam8i& p)
        {
            if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                return InputConvolution_2x1<true, type>;
            else
                return InputConvolution_2x1<false, type>;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a, 
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;

            InputConvolution_2xM_Ptr inputConvolution_2x1 = GetInputConvolution_2x1<type>(p);
            //ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<term, type>(n);
            //ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            //size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            __m128 _bias[2], _norm[2], _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    //for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                    //    convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    //for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                    //    convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    //for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                    //    convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    //for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                    //    convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    //for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                    //    convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    //for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                    //    convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            input = InputConvolution_2<type>;
        }

        SIMD_INLINE void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveDepthwise(uint8_t* dst, __m128 sum, const __m128* params, const __m128 & scale, const __m128& shift, const __m128i& upper)
        {
            __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse2::Activate<type>(sum, params, 0), scale), shift));
            ((int32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam8i& p, const AlgParam& a, size_t maC, 
            size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t srcW = p.srcW * F, dstW = p.dstW * a.maC, weightS = p.kernelY * p.kernelX * F, strideXF = strideX * F;
            size_t srcM = (a.bufH[1] - 1), dstM = (a.bufH[2] - 1), srcS = a.bufH[1] * srcW;// , dstS = bufH[1] * dstW;
            //size_t noseY = p.NoseH(); (p.padY + p.strideY - 1) / p.strideY;
            //size_t bodyY = (p.srcH + p.padY + p.strideY - p.kernelY) / p.strideY;
            //size_t noseX = (p.padX + p.strideX - 1) / p.strideX;
            //size_t bodyX = (p.srcW + p.padX + p.strideX - p.kernelX) / p.strideX;
            //size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            //size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            //size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;

            __m128i _upper = _mm_set1_epi32(a.upper);
            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm_set1_ps(params[1]);
            for (size_t c = 0; c < maC; c += F)
            {
                __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                __m128 _scale = _mm_loadu_ps(scale + c);
                __m128 _shift = _mm_loadu_ps(shift + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_loadu_ps(params + c);

                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t * pd = dst + (dy & dstM) * dstW;
                    //if (dy >= noseY && dy < bodyY)
                    //{
                    //    size_t dx = 0;
                    //    for (; dx < noseX; ++dx, pd += F)
                    //    {
                    //        __m128 sum = _bias;
                    //        for (size_t ky = 0; ky < p.kernelY; ++ky)
                    //        {
                    //            size_t sy = dy * p.strideY + ky - padY;
                    //            for (size_t kx = 0; kx < p.kernelX; ++kx)
                    //            {
                    //                size_t sx = dx * p.strideX + kx - padX;
                    //                if (sx < p.srcW)
                    //                {
                    //                    const float* pw = weight + (ky * p.kernelX + kx) * F;
                    //                    const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
                    //                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                    //                }
                    //            }
                    //        }
                    //        _mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
                    //    }
                        //for (; dx < bodyX8; dx += 8, pd += 8 * F)
                        //{
                        //    __m128 sum0 = _bias;
                        //    __m128 sum1 = _bias;
                        //    __m128 sum2 = _bias;
                        //    __m128 sum3 = _bias;
                        //    __m128 sum4 = _bias;
                        //    __m128 sum5 = _bias;
                        //    __m128 sum6 = _bias;
                        //    __m128 sum7 = _bias;
                        //    const float* pw = weight;
                        //    for (size_t ky = 0; ky < p.kernelY; ++ky)
                        //    {
                        //        size_t sy = dy * strideY + ky - padY;
                        //        const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                        //        for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                        //        {
                        //            __m128 w0 = _mm_loadu_ps(pw);
                        //            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                        //            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                        //            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
                        //            sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
                        //            sum4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 4 * strideXF), w0), sum4);
                        //            sum5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 5 * strideXF), w0), sum5);
                        //            sum6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 6 * strideXF), w0), sum6);
                        //            sum7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 7 * strideXF), w0), sum7);
                        //        }
                        //    }
                        //    _mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
                        //    _mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
                        //    _mm_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
                        //    _mm_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
                        //    _mm_storeu_ps(pd + 4 * F, Activate<type>(sum4, _params, 0));
                        //    _mm_storeu_ps(pd + 5 * F, Activate<type>(sum5, _params, 0));
                        //    _mm_storeu_ps(pd + 6 * F, Activate<type>(sum6, _params, 0));
                        //    _mm_storeu_ps(pd + 7 * F, Activate<type>(sum7, _params, 0));
                        //}
                        //for (; dx < bodyX4; dx += 4, pd += 4 * F)
                        //{
                        //    __m128 sum0 = _bias;
                        //    __m128 sum1 = _bias;
                        //    __m128 sum2 = _bias;
                        //    __m128 sum3 = _bias;
                        //    const float* pw = weight;
                        //    for (size_t ky = 0; ky < p.kernelY; ++ky)
                        //    {
                        //        size_t sy = dy * strideY + ky - padY;
                        //        const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                        //        for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                        //        {
                        //            __m128 w0 = _mm_loadu_ps(pw);
                        //            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                        //            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                        //            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
                        //            sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
                        //        }
                        //    }
                        //    _mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
                        //    _mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
                        //    _mm_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
                        //    _mm_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
                        //}
                        //for (; dx < bodyX2; dx += 2, pd += 2 * F)
                        //{
                        //    __m128 sum0 = _bias;
                        //    __m128 sum1 = _bias;
                        //    const float* pw = weight;
                        //    for (size_t ky = 0; ky < p.kernelY; ++ky)
                        //    {
                        //        size_t sy = dy * strideY + ky - padY;
                        //        const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                        //        for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                        //        {
                        //            __m128 w0 = _mm_loadu_ps(pw);
                        //            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                        //            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                        //        }
                        //    }
                        //    _mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
                        //    _mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
                        //}
                    //    for (; dx < bodyX; ++dx, pd += F)
                    //    {
                    //        __m128 sum = _bias;
                    //        const float* pw = weight;
                    //        for (size_t ky = 0; ky < p.kernelY; ++ky)
                    //        {
                    //            size_t sy = dy * strideY + ky - padY;
                    //            const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                    //            for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                    //            {
                    //                __m128 w0 = _mm_loadu_ps(pw);
                    //                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), w0), sum);
                    //            }
                    //        }
                    //        _mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
                    //    }
                    //    for (; dx < p.dstW; ++dx, pd += F)
                    //    {
                    //        __m128 sum = _bias;
                    //        for (size_t ky = 0; ky < p.kernelY; ++ky)
                    //        {
                    //            size_t sy = dy * strideY + ky - padY;
                    //            for (size_t kx = 0; kx < p.kernelX; ++kx)
                    //            {
                    //                size_t sx = dx * strideX + kx - padX;
                    //                if (sx < p.srcW)
                    //                {
                    //                    const float* pw = weight + (ky * p.kernelX + kx) * F;
                    //                    const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
                    //                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                    //                }
                    //            }
                    //        }
                    //        _mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
                    //    }
                    //}
                    //else
                    {
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += a.maC)
                        {
                            __m128 sum = _bias;
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
                                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                        }
                                    }
                                }
                            }
                            SaveDepthwise<type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                    }
                }
                src += srcS;
                dst += F;
                weight += weightS;
            }
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            depthwise = DepthwiseConvolution<type>;
        }

        SIMD_INLINE void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationRelu: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationLeakyRelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationRestrictRange: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationPrelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationElu: SetDepthwise<SimdConvolutionActivationElu>(p, depthwise); break;
            case SimdConvolutionActivationHswish: SetDepthwise<SimdConvolutionActivationHswish>(p, depthwise); break;
            }
        }

        //---------------------------------------------------------------------

        typedef void(*OutputConvolutionPtr)(const uint8_t* src, const ConvParam8i& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            const int8_t* weight, const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst);


        template<Term8iType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = a.maC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        weight += DA;
                    }
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                        weight += DA;
                    }
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return OutputConvolution1x1_2xM<term, type, 1>;
            case 2: return OutputConvolution1x1_2xM<term, type, 2>;
            case 3: return OutputConvolution1x1_2xM<term, type, 3>;
            case 4: return OutputConvolution1x1_2xM<term, type, 4>;
            case 5: return OutputConvolution1x1_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            //size_t yInt = Simd::Max(yBeg, yEnd & (~dstM)), nBeg = yBeg * dstW, nInt = yInt * dstW, nEnd = yEnd * dstW;
            //size_t nInt6 = AlignLoAny(nInt - nBeg, 6) + nBeg, nEnd6 = AlignLoAny(nEnd - nInt, 6) + nInt, nIntTail = nInt - nInt6, nEndTail = nEnd - nEnd6;
            //InputConvolution1x1_2xM_Ptr tailInt_2 = GetInputConvolution1x1_2xM<type>(nIntTail);
            //InputConvolution1x1_2xM_Ptr tailEnd_2 = GetInputConvolution1x1_2xM<type>(nEndTail);

            size_t dstM = (a.bufH[2] - 1), yInt = Simd::Max(yBeg, yEnd & (~dstM));
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m128 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                _scale[0] = _mm_loadu_ps(scale + dc + 0);
                _scale[1] = _mm_loadu_ps(scale + dc + F);
                _shift[0] = _mm_loadu_ps(shift + dc + 0);
                _shift[1] = _mm_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                weight += DivHi(maC, 4) * DA;
            }
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SetOutput(const ConvParam8i& p, OutputConvolutionPtr * output)
        {
            output[0] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iSingle32f, type> : OutputConvolution1x1_2<Term8iSingle8u, type>;
            output[1] = OutputConvolution1x1_2<Term8iFirst, SimdConvolutionActivationIdentity>;
            output[2] = OutputConvolution1x1_2<Term8iIterim, SimdConvolutionActivationIdentity>;
            output[3] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iLast32f, type> : OutputConvolution1x1_2<Term8iLast8u, type>;
        }

        SIMD_INLINE void SetOutput(const ConvParam8i& p, OutputConvolutionPtr * output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, output); break;
            case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, output); break;
            }
        }

        //---------------------------------------------------------------------


        SynetMergedConvolution8iCdc::SynetMergedConvolution8iCdc(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCdc(p)
        {
            SetSize(Sse::F);
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
            SetOutput(_param.conv[2], _output);
        }

        //---------------------------------------------------------------------

        void* SynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam8i param(batch, convs, count, compatibility);
            if (!param.Valid())
                return NULL;
            if (SynetMergedConvolution8iCdc::Preferable(param))
                return new Sse41::SynetMergedConvolution8iCdc(param);
            else
                return new Base::SynetMergedConvolution8i(param);
        }
    }
#endif
}
