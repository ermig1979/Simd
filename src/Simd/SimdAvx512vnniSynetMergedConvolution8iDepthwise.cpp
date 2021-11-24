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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx512vnni
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution(const float* src, const ConvParam8i& p, const AlgParam& a, size_t dstC,
            size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX =  strideX * sX;
            size_t noseY = p.NoseH(), bodyY = p.BodyH(), noseX = p.NoseW(), bodyX = p.BodyW();
            size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F);

            __m128i _upper = _mm_set1_epi32(a.upper);
            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                __m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);
                __m512 _scale = _mm512_loadu_ps(scale + c);
                __m512 _shift = _mm512_loadu_ps(shift + c);
                __mmask16 tail = TailMask16(dstC - c);
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    if (dy >= noseY && dy < bodyY)
                    {
                        size_t dx = 0;
                        for (; dx < noseX; dx += 1, pd += dX)
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
                                        const float* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                            Save1<term, type, nofma>(pd, sum, _params, _scale, _shift, _upper, tail);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
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
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(_mm512_loadu_ps(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(_mm512_loadu_ps(ps + 3 * ssX), w0, sum3);
                                    sum4 = Fmadd<nofma>(_mm512_loadu_ps(ps + 4 * ssX), w0, sum4);
                                    sum5 = Fmadd<nofma>(_mm512_loadu_ps(ps + 5 * ssX), w0, sum5);
                                    sum6 = Fmadd<nofma>(_mm512_loadu_ps(ps + 6 * ssX), w0, sum6);
                                    sum7 = Fmadd<nofma>(_mm512_loadu_ps(ps + 7 * ssX), w0, sum7);
                                }
                            }
                            Save1<term, type, nofma>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 4 * dX, sum4, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 5 * dX, sum5, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 6 * dX, sum6, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 7 * dX, sum7, _params, _scale, _shift, _upper, tail);
                        }
                        for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                        {
                            __m512 sum0 = _bias;
                            __m512 sum1 = _bias;
                            __m512 sum2 = _bias;
                            __m512 sum3 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(_mm512_loadu_ps(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(_mm512_loadu_ps(ps + 3 * ssX), w0, sum3);
                                }
                            }
                            Save1<term, type, nofma>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper, tail);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            __m512 sum0 = _bias;
                            __m512 sum1 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
                                }
                            }
                            Save1<term, type, nofma>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper, tail);
                            Save1<term, type, nofma>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper, tail);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            __m512 sum = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum = Fmadd<nofma>(_mm512_loadu_ps(ps), w0, sum);
                                }
                            }
                            Save1<term, type, nofma>(pd, sum, _params, _scale, _shift, _upper, tail);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
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
                                        const float* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                            Save1<term, type, nofma>(pd, sum, _params, _scale, _shift, _upper, tail);
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
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
                                            const float* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = Fmadd<nofma>(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type, nofma>(pd, sum, _params, _scale, _shift, _upper, tail);
                        }
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const float* src0, const float* src1, 
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = bias;
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum);
                Save1<term, type, true>(dst, sum, params, scale, shift, upper, tail);
            }
            else
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
                Save1<term, type, false>(dst, _mm512_add_ps(sum0, sum1), params, scale, shift, upper, tail);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const float* src0, const float* src1,
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = bias;
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum);
                Save1<term, type, true>(dst, sum, params, scale, shift, upper, tail);
            }
            else
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum2);
                Save1<term, type, false>(dst, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, scale, shift, upper, tail);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const float* src0, const float* src1, const float* src2, 
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = bias;
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum);
                Save1<term, type, true>(dst, sum, params, scale, shift, upper, tail);
            }
            else
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum1);
                Save1<term, type, false>(dst, _mm512_add_ps(sum0, sum1), params, scale, shift, upper, tail);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const float* src0, const float* src1, const float* src2,
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = bias;
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum);
                sum = Fmadd<true>(_mm512_loadu_ps(src2 + 2 * sX), weight[8], sum);
                Save1<term, type, true>(dst, sum, params, scale, shift, upper, tail);
            }
            else
            {
                __m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum2);
                sum0 = Fmadd<false>(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum1);
                sum2 = Fmadd<false>(_mm512_loadu_ps(src2 + 2 * sX), weight[8], sum2);
                Save1<term, type, false>(dst, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, scale, shift, upper, tail);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x2(const float* src0, const float* src1, const float* src2,
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = bias, sum1 = bias, s0;

            s0 = _mm512_loadu_ps(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = _mm512_loadu_ps(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = _mm512_loadu_ps(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            s0 = _mm512_loadu_ps(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);

            s0 = _mm512_loadu_ps(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = _mm512_loadu_ps(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = _mm512_loadu_ps(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            s0 = _mm512_loadu_ps(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);

            s0 = _mm512_loadu_ps(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = _mm512_loadu_ps(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = _mm512_loadu_ps(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            s0 = _mm512_loadu_ps(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);

            Save1<term, type, nofma>(dst + 0 * dX, sum0, params, scale, shift, upper, tail);
            Save1<term, type, nofma>(dst + 1 * dX, sum1, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x4(const float* src0, const float* src1, const float* src2,
            size_t sX, const __m512* weight, const __m512& bias, const __m512* params, const __m512& scale, const __m512& shift, const __m128i& upper, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = bias, sum1 = bias, sum2 = bias, sum3 = bias, s0;

            s0 = _mm512_loadu_ps(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = _mm512_loadu_ps(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = _mm512_loadu_ps(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            sum2 = Fmadd<nofma>(s0, weight[0], sum2);
            s0 = _mm512_loadu_ps(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);
            sum2 = Fmadd<nofma>(s0, weight[1], sum2);
            sum3 = Fmadd<nofma>(s0, weight[0], sum3);
            s0 = _mm512_loadu_ps(src0 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[2], sum2);
            sum3 = Fmadd<nofma>(s0, weight[1], sum3);
            s0 = _mm512_loadu_ps(src0 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[2], sum3);

            s0 = _mm512_loadu_ps(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = _mm512_loadu_ps(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = _mm512_loadu_ps(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            sum2 = Fmadd<nofma>(s0, weight[3], sum2);
            s0 = _mm512_loadu_ps(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);
            sum2 = Fmadd<nofma>(s0, weight[4], sum2);
            sum3 = Fmadd<nofma>(s0, weight[3], sum3);
            s0 = _mm512_loadu_ps(src1 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[5], sum2);
            sum3 = Fmadd<nofma>(s0, weight[4], sum3);
            s0 = _mm512_loadu_ps(src1 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[5], sum3);

            s0 = _mm512_loadu_ps(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = _mm512_loadu_ps(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = _mm512_loadu_ps(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            sum2 = Fmadd<nofma>(s0, weight[6], sum2);
            s0 = _mm512_loadu_ps(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);
            sum2 = Fmadd<nofma>(s0, weight[7], sum2);
            sum3 = Fmadd<nofma>(s0, weight[6], sum3);
            s0 = _mm512_loadu_ps(src2 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[8], sum2);
            sum3 = Fmadd<nofma>(s0, weight[7], sum3);
            s0 = _mm512_loadu_ps(src2 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[8], sum3);

            Save1<term, type, nofma>(dst + 0 * dX, sum0, params, scale, shift, upper, tail);
            Save1<term, type, nofma>(dst + 1 * dX, sum1, params, scale, shift, upper, tail);
            Save1<term, type, nofma>(dst + 2 * dX, sum2, params, scale, shift, upper, tail);
            Save1<term, type, nofma>(dst + 3 * dX, sum3, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const float* src, const ConvParam8i& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX)*sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;
            size_t xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t xMainEnd4 = AlignLo(xMainEnd - padX, 4) * (p.strideX == 1 ? 1 : 0) + padX;
            __m128i _upper = _mm_set1_epi32(a.upper);
            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                __mmask16 tail = TailMask16(dstC - c);
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                __m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);
                __m512 _scale = _mm512_loadu_ps(scale + c);
                __m512 _shift = _mm512_loadu_ps(shift + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, _scale, _shift, _upper, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst, tail);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    const float* src2 = src + ((sy + 2) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd4; dx += 4, pDst += dX * 4, src0 += ssX * 4, src1 += ssX * 4, src2 += ssX * 4)
                        DepthwiseConvolution3x3Main1x4<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, dX, tail);
                    for (; dx < xMainEnd2; dx += 2, pDst += dX * 2, src0 += ssX * 2, src1 += ssX * 2, src2 += ssX * 2)
                        DepthwiseConvolution3x3Main1x2<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, dX, tail);
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, tail);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst, tail);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.IsKernel(3) && p.IsDilation(1))
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = DepthwiseConvolution3x3<term, type, true>;
                else
                    depthwise = DepthwiseConvolution3x3<term, type, false>;
            }
            else
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = DepthwiseConvolution<term, type, true>;
                else
                    depthwise = DepthwiseConvolution<term, type, false>;
            }
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.dstT == SimdTensorData32f)
                SetDepthwise<Term8iLast32f, type>(p, depthwise);
            else
                SetDepthwise<Term8iLast8u, type>(p, depthwise);
        }

        void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
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
            case SimdConvolutionActivationMish: SetDepthwise<SimdConvolutionActivationMish>(p, depthwise); break;
            case SimdConvolutionActivationSwish: SetDepthwise<SimdConvolutionActivationSwish>(p, depthwise); break;
            }
        }
    }
#endif
}
