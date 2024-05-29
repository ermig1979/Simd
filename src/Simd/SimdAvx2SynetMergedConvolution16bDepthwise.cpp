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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdFmadd.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        template <class T> SIMD_INLINE __m256 LoadSrc(const T* src);

        template <> SIMD_INLINE __m256 LoadSrc<float>(const float* src)
        {
            return _mm256_loadu_ps(src);
        }

        template <> SIMD_INLINE __m256 LoadSrc<uint16_t>(const uint16_t* src)
        {
            return Avx2::BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)src)));
        }

        //---------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX =  strideX * sX;
            size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
            size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F);

            __m256 _params[2], _bias[1];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm256_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                _bias[0] = _mm256_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm256_loadu_ps(params + c);
                if (c == dstCF)
                {
                    size_t tail = dstC - dstCF;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m256 sum = _mm256_setzero_ps();
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
                                            const T* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = Fmadd<nofma>(LoadSrc(ps), _mm256_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, tail);
                        }
                    }
                    return;
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    if (dy >= noseY && dy < bodyY)
                    {
                        size_t dx = 0;
                        for (; dx < noseX; dx += 1, pd += dX)
                        {
                            __m256 sum = _mm256_setzero_ps();
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * p.strideY + ky - padY;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm256_loadu_ps(pw), sum);
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                        {
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            __m256 sum2 = _mm256_setzero_ps();
                            __m256 sum3 = _mm256_setzero_ps();
                            __m256 sum4 = _mm256_setzero_ps();
                            __m256 sum5 = _mm256_setzero_ps();
                            __m256 sum6 = _mm256_setzero_ps();
                            __m256 sum7 = _mm256_setzero_ps();
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m256 w0 = _mm256_loadu_ps(pw);
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
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            __m256 sum2 = _mm256_setzero_ps();
                            __m256 sum3 = _mm256_setzero_ps();
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m256 w0 = _mm256_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(LoadSrc(ps + 3 * ssX), w0, sum3);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                            Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                            Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m256 w0 = _mm256_loadu_ps(pw);
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            __m256 sum = _mm256_setzero_ps();
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m256 w0 = _mm256_loadu_ps(pw);
                                    sum = Fmadd<nofma>(LoadSrc(ps), w0, sum);
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
                        {
                            __m256 sum = _mm256_setzero_ps();
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm256_loadu_ps(pw), sum);
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m256 sum = _mm256_setzero_ps();
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
                                            const T* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = Fmadd<nofma>(LoadSrc(ps), _mm256_loadu_ps(pw), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const T* src0,
            const T* src1, size_t sX, const __m256* weight, const __m256 * bias, const __m256* params, uint8_t* dst)
        {
            if (nofma)
            {
                __m256 sum = _mm256_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                Save1<term, type>(dst, NULL, _mm256_add_ps(sum0, sum1), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const __m256* weight, const __m256 * bias, const __m256* params, uint8_t* dst)
        {
            if (nofma)
            {
                __m256 sum = _mm256_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                Save1<term, type>(dst, NULL, _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m256* weight, const __m256 * bias, const __m256* params, uint8_t* dst)
        {
            if (nofma)
            {
                __m256 sum = _mm256_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                Save1<term, type>(dst, NULL, _mm256_add_ps(sum0, sum1), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m256* weight, const __m256 * bias, const __m256* params, uint8_t* dst)
        {
            if (nofma)
            {
                __m256 sum = _mm256_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 2 * sX), weight[8], sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                sum2 = Fmadd<false>(LoadSrc(src2 + 2 * sX), weight[8], sum2);
                Save1<term, type>(dst, NULL, _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = maC;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX)*sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            __m256 _params[2], _bias[1];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm256_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                __m256 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm256_loadu_ps(weight + i * F);
                _bias[0] = _mm256_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm256_loadu_ps(params + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst);
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
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (IsKernel(p, 3) && IsDilation(p, 1) && Aligned(p.dstC, F))
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = p.srcT == SimdTensorData16b ? 
                    DepthwiseConvolution3x3<uint16_t, term, type, true> :
                    DepthwiseConvolution3x3<float, term, type, true>;
                else
                    depthwise = p.srcT == SimdTensorData16b ? 
                    DepthwiseConvolution3x3<uint16_t, term, type, false> :
                    DepthwiseConvolution3x3<float, term, type, false>;
            }
            else
            {
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (p.srcT == SimdTensorData16b)
                        depthwise = DepthwiseConvolution<uint16_t, term, type, true>;
                    else
                        depthwise = DepthwiseConvolution<float, term, type, true>;
                }
                else
                {
                    if (p.srcT == SimdTensorData16b)
                        depthwise = DepthwiseConvolution<uint16_t, term, type, false>;
                    else
                        depthwise = DepthwiseConvolution<float, term, type, false>;
                }
            }
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (p.dstT == SimdTensorData32f)
                SetDepthwise<Term16bLast32f, type>(p, depthwise);
            else
                SetDepthwise<Term16bLast16b, type>(p, depthwise);
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
