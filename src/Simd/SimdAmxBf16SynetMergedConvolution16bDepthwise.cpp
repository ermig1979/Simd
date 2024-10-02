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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m512 LoadSrc(const T* src, __mmask16 mask = -1);

        template <> SIMD_INLINE __m512 LoadSrc<float>(const float* src, __mmask16 mask)
        {
            return _mm512_maskz_loadu_ps(mask, src);
        }

        template <> SIMD_INLINE __m512 LoadSrc<uint16_t>(const uint16_t* src, __mmask16 mask)
        {
            return BFloat16ToFloat32(_mm256_maskz_loadu_epi16(mask, src));
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolutionDefault(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW, kernelX = p.kernelX, kernelY = p.kernelY;
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
            size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
            size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F), dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstCe; c += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);
                if (c == dstCF)
                {
                    __mmask16 tail = TailMask16(dstC - c);
                    __mmask32 gapMask = a.bufH[2] ? TailMask32(dstCe - dstC) : 0;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m512 sum = _mm512_setzero_ps();
                            for (size_t ky = 0; ky < kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                if (sy < srcH)
                                {
                                    for (size_t kx = 0; kx < kernelX; ++kx)
                                    {
                                        size_t sx = dx * strideX + kx - padX;
                                        if (sx < srcW)
                                        {
                                            const float* pw = weight + (ky * kernelX + kx) * F;
                                            const T* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = Fmadd<nofma>(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, tail);
                            if (gapMask)
                                SetZero((uint16_t*)pd + dstC - dstCF, gapMask);
                        }
                    }
                    return;
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    size_t dx = 0;
                    for (; dx < noseX; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                    for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        __m512 sum2 = _mm512_setzero_ps();
                        __m512 sum3 = _mm512_setzero_ps();
                        __m512 sum4 = _mm512_setzero_ps();
                        __m512 sum5 = _mm512_setzero_ps();
                        __m512 sum6 = _mm512_setzero_ps();
                        __m512 sum7 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(LoadSrc(ps + 3 * ssX), w0, sum3);
                                    sum4 = Fmadd<nofma>(LoadSrc(ps + 4 * ssX), w0, sum4);
                                    sum5 = Fmadd<nofma>(LoadSrc(ps + 5 * ssX), w0, sum5);
                                    sum6 = Fmadd<nofma>(LoadSrc(ps + 6 * ssX), w0, sum6);
                                    sum7 = Fmadd<nofma>(LoadSrc(ps + 7 * ssX), w0, sum7);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                        Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                        Save1<term, type>(pd + 4 * dX, NULL, sum4, _bias, _params);
                        Save1<term, type>(pd + 5 * dX, NULL, sum5, _bias, _params);
                        Save1<term, type>(pd + 6 * dX, NULL, sum6, _bias, _params);
                        Save1<term, type>(pd + 7 * dX, NULL, sum7, _bias, _params);
                    }
                    for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        __m512 sum2 = _mm512_setzero_ps();
                        __m512 sum3 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(LoadSrc(ps + 3 * ssX), w0, sum3);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                        Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                    }
                    for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                    }
                    for (; dx < bodyX; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum = Fmadd<nofma>(LoadSrc(ps), w0, sum);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                    for (; dx < p.dstW; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                Save1<term, type>(dst, NULL, _mm512_add_ps(sum0, sum1), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                Save1<term, type>(dst, NULL, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                Save1<term, type>(dst, NULL, _mm512_add_ps(sum0, sum1), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 2 * sX), weight[8], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                sum2 = Fmadd<false>(LoadSrc(src2 + 2 * sX), weight[8], sum2);
                Save1<term, type>(dst, NULL, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x4(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            sum2 = Fmadd<nofma>(s0, weight[0], sum2);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);
            sum2 = Fmadd<nofma>(s0, weight[1], sum2);
            sum3 = Fmadd<nofma>(s0, weight[0], sum3);
            s0 = LoadSrc(src0 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[2], sum2);
            sum3 = Fmadd<nofma>(s0, weight[1], sum3);
            s0 = LoadSrc(src0 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[2], sum3);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            sum2 = Fmadd<nofma>(s0, weight[3], sum2);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);
            sum2 = Fmadd<nofma>(s0, weight[4], sum2);
            sum3 = Fmadd<nofma>(s0, weight[3], sum3);
            s0 = LoadSrc(src1 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[5], sum2);
            sum3 = Fmadd<nofma>(s0, weight[4], sum3);
            s0 = LoadSrc(src1 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[5], sum3);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            sum2 = Fmadd<nofma>(s0, weight[6], sum2);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);
            sum2 = Fmadd<nofma>(s0, weight[7], sum2);
            sum3 = Fmadd<nofma>(s0, weight[6], sum3);
            s0 = LoadSrc(src2 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[8], sum2);
            sum3 = Fmadd<nofma>(s0, weight[7], sum3);
            s0 = LoadSrc(src2 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[8], sum3);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
            Save1<term, type>(dst + 2 * dX, NULL, sum2, bias, params, tail);
            Save1<term, type>(dst + 3 * dX, NULL, sum3, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = maC;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;
            size_t xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t xMainEnd4 = AlignLo(xMainEnd - padX, 4) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstCe; c += F)
            {
                __mmask16 tail = TailMask16(dstC - c);
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);

                size_t dy = yBeg;
                if (c == dstC)
                {
                    __mmask32 gapMask = a.bufH[2] ? TailMask32(dstCe - dstC) : 0;
                    for (; dy < yEnd; ++dy)
                    {
                        uint8_t* pDst = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; dx++, pDst += dX)
                            SetZero((uint16_t*)pDst, gapMask);
                    }
                    return;
                }
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    const T* src2 = src + ((sy + 2) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd4; dx += 4, pDst += dX * 4, src0 += ssX * 4, src1 += ssX * 4, src2 += ssX * 4)
                        DepthwiseConvolution3x3Main1x4<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd2; dx += 2, pDst += dX * 2, src0 += ssX * 2, src1 += ssX * 2, src2 += ssX * 2)
                        DepthwiseConvolution3x3Main1x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (IsKernel(p, 3) && IsDilation(p, 1) && Aligned(p.dstC, F))
                depthwise = DepthwiseConvolution3x3<T, term, type, nofma>;
            else
                depthwise = DepthwiseConvolutionDefault<T, term, type, nofma>;
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return Base::FmaAvoid(p.compatibility) ? SetDepthwise<T, term, type, true>(p, depthwise) : SetDepthwise<T, term, type, false>(p, depthwise);
        }

        template<typename T, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return p.dstT == SimdTensorData32f ? SetDepthwise<T, Term16bLast32f, type>(p, depthwise) : SetDepthwise<T, Term16bLast16b, type>(p, depthwise);
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return p.srcT == SimdTensorData16b ? SetDepthwise<uint16_t, type>(p, depthwise) : SetDepthwise<float, type>(p, depthwise);
        }

        void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
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
            case SimdConvolutionActivationHardSigmoid: SetDepthwise<SimdConvolutionActivationHardSigmoid>(p, depthwise); break;
            case SimdConvolutionActivationSwish: SetDepthwise<SimdConvolutionActivationSwish>(p, depthwise); break;
            case SimdConvolutionActivationGelu: SetDepthwise<SimdConvolutionActivationGelu>(p, depthwise); break;
            }
        }
    }
#endif
}
