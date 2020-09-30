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
        namespace Cd
        {
            using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
            using Convert8uTo32fPtr = Base::SynetMergedConvolution8i::Convert8uTo32fPtr;
            using Convert32fTo8uPtr = Base::SynetMergedConvolution8i::Convert32fTo8uPtr;
            using InputConvolutionPtr = Base::SynetMergedConvolution8i::InputConvolutionPtr;
            using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;
            using OutputConvolutionPtr = Base::SynetMergedConvolution8i::OutputConvolutionPtr;

            //---------------------------------------------------------------------

            template<Term8iType term, SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam8i& p, const AlgParam& a, size_t dstC,
                size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
            {
                size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
                size_t srcW = p.srcW * F, dX = p.dstC * a.size, dY = p.dstW * dX, weightS = p.kernelY * p.kernelX * F, strideXF = strideX * F;
                size_t srcM = (a.bufH[1] - 1), srcS = a.bufH[1] * srcW;
                size_t noseY = p.NoseH(), bodyY = p.BodyH(), noseX = p.NoseW(), bodyX = p.BodyW();
                size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
                size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
                size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
                size_t dstCF = AlignLo(dstC, F);

                __m128i _upper = _mm_set1_epi32(a.upper);
                __m128 _params[2];
                _params[0] = _mm_set1_ps(params[0]);
                if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                    _params[1] = _mm_set1_ps(params[1]);
                for (size_t c = 0; c < dstC; c += F)
                {
                    __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                    if (type == ::SimdConvolutionActivationPrelu)
                        _params[0] = _mm_loadu_ps(params + c);
                    __m128 _scale = _mm_loadu_ps(scale + c);
                    __m128 _shift = _mm_loadu_ps(shift + c);

                    if (c == dstCF)
                    {
                        size_t tail = dstC - dstCF;
                        for (size_t dy = yBeg; dy < yEnd; ++dy)
                        {
                            uint8_t* pd = dst + dy * dY;
                            for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
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
                                Save1<term, type>(pd, sum, _params, _scale, _shift, _upper, tail);
                            }
                        }
                        return;
                    }
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + dy * dY;
                        if (dy >= noseY && dy < bodyY)
                        {
                            size_t dx = 0;
                            for (; dx < noseX; dx += 1, pd += dX)
                            {
                                __m128 sum = _bias;
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
                                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                        }
                                    }
                                }
                                Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                            }
                            for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                            {
                                __m128 sum0 = _bias;
                                __m128 sum1 = _bias;
                                __m128 sum2 = _bias;
                                __m128 sum3 = _bias;
                                __m128 sum4 = _bias;
                                __m128 sum5 = _bias;
                                __m128 sum6 = _bias;
                                __m128 sum7 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m128 w0 = _mm_loadu_ps(pw);
                                        sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                                        sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                                        sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
                                        sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
                                        sum4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 4 * strideXF), w0), sum4);
                                        sum5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 5 * strideXF), w0), sum5);
                                        sum6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 6 * strideXF), w0), sum6);
                                        sum7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 7 * strideXF), w0), sum7);
                                    }
                                }
                                Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 4 * dX, sum4, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 5 * dX, sum5, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 6 * dX, sum6, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 7 * dX, sum7, _params, _scale, _shift, _upper);
                            }
                            for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                            {
                                __m128 sum0 = _bias;
                                __m128 sum1 = _bias;
                                __m128 sum2 = _bias;
                                __m128 sum3 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m128 w0 = _mm_loadu_ps(pw);
                                        sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                                        sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                                        sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
                                        sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
                                    }
                                }
                                Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper);
                            }
                            for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                            {
                                __m128 sum0 = _bias;
                                __m128 sum1 = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m128 w0 = _mm_loadu_ps(pw);
                                        sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
                                        sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
                                    }
                                }
                                Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                                Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                            }
                            for (; dx < bodyX; dx += 1, pd += dX)
                            {
                                __m128 sum = _bias;
                                const float* pw = weight;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    size_t sy = dy * strideY + ky - padY;
                                    const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
                                    {
                                        __m128 w0 = _mm_loadu_ps(pw);
                                        sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), w0), sum);
                                    }
                                }
                                Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                            }
                            for (; dx < p.dstW; dx += 1, pd += dX)
                            {
                                __m128 sum = _bias;
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
                                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                        }
                                    }
                                }
                                Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                            }
                        }
                        else
                        {
                            for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
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
                                Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                            }
                        }
                    }
                    src += srcS;
                    dst += F * a.size;
                    weight += weightS;
                }
            }

            //---------------------------------------------------------------------

            template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const float* src0, const float* src1,
                const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
            {
                if (nofma)
                {
                    __m128 sum = bias;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum);
                    Save1<term, type>(dst, sum, params, scale, shift, upper);
                }
                else
                {
                    __m128 sum0 = bias, sum1 = _mm_setzero_ps();
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
                    Save1<term, type>(dst, _mm_add_ps(sum0, sum1), params, scale, shift, upper);
                }
            }

            template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const float* src0, const float* src1,
                const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
            {
                if (nofma)
                {
                    __m128 sum = bias;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum);
                    Save1<term, type>(dst, sum, params, scale, shift, upper);
                }
                else
                {
                    __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
                    Save1<term, type>(dst, _mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, scale, shift, upper);
                }
            }

            template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const float* src0, const float* src1, 
                const float* src2, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
            {
                if (nofma)
                {
                    __m128 sum = bias;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum);
                    Save1<term, type>(dst, sum, params, scale, shift, upper);
                }
                else
                {
                    __m128 sum0 = bias, sum1 = _mm_setzero_ps();
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
                    Save1<term, type>(dst, _mm_add_ps(sum0, sum1), params, scale, shift, upper);
                }
            }

            template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const float* src0, const float* src1, 
                const float* src2, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
            {
                if (nofma)
                {
                    __m128 sum = bias;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * F), weight[8]), sum);
                    Save1<term, type>(dst, sum, params, scale, shift, upper);
                }
                else
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
                    Save1<term, type>(dst, _mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, scale, shift, upper);
                }
            }

            template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const float* src, const ConvParam8i& p, const AlgParam& a, 
                size_t dstC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
            {
                size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
                size_t srcW = p.srcW * F, dX = p.dstC * a.size, dY = p.dstW * dX, weightS = p.kernelY * p.kernelX * F;
                size_t srcM = (a.bufH[1] - 1), srcS = a.bufH[1] * srcW;
                size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX) * F;
                size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

                __m128i _upper = _mm_set1_epi32(a.upper);
                __m128 _params[2];
                _params[0] = _mm_set1_ps(params[0]);
                if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                    _params[1] = _mm_set1_ps(params[1]);
                for (size_t c = 0; c < dstC; c += F)
                {
                    __m128 _weight[9];
                    for (size_t i = 0; i < 9; ++i)
                        _weight[i] = _mm_loadu_ps(weight + i * F);
                    __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                    if (type == ::SimdConvolutionActivationPrelu)
                        _params[0] = _mm_loadu_ps(params + c);
                    __m128 _scale = _mm_loadu_ps(scale + c);
                    __m128 _shift = _mm_loadu_ps(shift + c);

                    size_t dy = yBeg;
                    if (yBeg == 0 && padY)
                    {
                        size_t sy = 0, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        uint8_t* pDst = dst + dy * dY;
                        if (padX)
                            DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, _weight + 4, _bias, _params, _scale, _shift, _upper, pDst), 
                            pDst += dX, dx++, src0 += xStep0, src1 += xStep0;
                        for (; dx < xMainEnd; dx++, pDst += dX, src0 += xStep, src1 += xStep)
                            DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
                        if (padW)
                            DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
                        dy++;
                    }
                    for (; dy < yMainEnd; ++dy)
                    {
                        size_t sy = dy * strideY - padY, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        const float* src2 = src + ((sy + 2) & srcM) * srcW;
                        uint8_t* pDst = dst + dy * dY;
                        if (padX)
                            DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst), 
                            pDst += dX, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
                        for (; dx < xMainEnd; dx++, pDst += dX, src0 += xStep, src1 += xStep, src2 += xStep)
                            DepthwiseConvolution3x3Main1x1<term, type, nofma>(src0, src1, src2, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                        if (padW)
                            DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    }
                    if (dy < yEnd)
                    {
                        size_t sy = dy * strideY - padY, dx = 0;
                        const float* src0 = src + ((sy + 0) & srcM) * srcW;
                        const float* src1 = src + ((sy + 1) & srcM) * srcW;
                        uint8_t* pDst = dst + dy * dY;
                        if (padX)
                            DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst), 
                            pDst += dX, dx++, src0 += xStep0, src1 += xStep0;
                        for (; dx < xMainEnd; dx++, pDst += dX, src0 += xStep, src1 += xStep)
                            DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                        if (padW)
                            DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    }
                    src += srcS;
                    dst += F * a.size;
                    weight += weightS;
                }
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCd::SynetMergedConvolution8iCd(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCd(p)
        {
            SetSize(Sse::F);
            SynetMergedConvolution8iCdc::Set(_cvt32fTo8u);
            SynetMergedConvolution8iCdc::Set(_param.conv[0], _input);
            Set(_param.conv[1], _depthwise);
        }

        template<Term8iType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, Cd::DepthwiseConvolutionPtr& depthwise)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && Aligned(p.dstC, F))
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = Cd::DepthwiseConvolution3x3<term, type, true>;
                else
                    depthwise = Cd::DepthwiseConvolution3x3<term, type, false>;
            }
            else
                depthwise = Cd::DepthwiseConvolution<term, type>;
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, Cd::DepthwiseConvolutionPtr& depthwise)
        {
            if(p.dstT == SimdTensorData32f)
                SetDepthwise<Term8iSingle32f, type>(p, depthwise);
            else 
                SetDepthwise<Term8iSingle8u, type>(p, depthwise);
        }

        void SynetMergedConvolution8iCd::Set(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
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
    }
#endif
}
