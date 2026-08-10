/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdNeon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void DepthwiseConvolution(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
            size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
            size_t bodyS = bodyX > noseX ? bodyX - noseX : 0;
            size_t bodyX2 = AlignLo(bodyS, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyS, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyS, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F);

            float32x4_t _params[2], _bias[1];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                _bias[0] = Load<false>(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + c);
                if (c == dstCF)
                {
                    size_t tail = dstC - dstCF;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            float32x4_t sum = vdupq_n_f32(0.0f);
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
                                            sum = vmlaq_f32(sum, LoadSrc(ps), Load<false>(pw));
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
                            float32x4_t sum = vdupq_n_f32(0.0f);
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
                                        sum = vmlaq_f32(sum, LoadSrc(ps), Load<false>(pw));
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                        {
                            float32x4_t sum0 = vdupq_n_f32(0.0f);
                            float32x4_t sum1 = vdupq_n_f32(0.0f);
                            float32x4_t sum2 = vdupq_n_f32(0.0f);
                            float32x4_t sum3 = vdupq_n_f32(0.0f);
                            float32x4_t sum4 = vdupq_n_f32(0.0f);
                            float32x4_t sum5 = vdupq_n_f32(0.0f);
                            float32x4_t sum6 = vdupq_n_f32(0.0f);
                            float32x4_t sum7 = vdupq_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vmlaq_f32(sum0, LoadSrc(ps + 0 * ssX), w0);
                                    sum1 = vmlaq_f32(sum1, LoadSrc(ps + 1 * ssX), w0);
                                    sum2 = vmlaq_f32(sum2, LoadSrc(ps + 2 * ssX), w0);
                                    sum3 = vmlaq_f32(sum3, LoadSrc(ps + 3 * ssX), w0);
                                    sum4 = vmlaq_f32(sum4, LoadSrc(ps + 4 * ssX), w0);
                                    sum5 = vmlaq_f32(sum5, LoadSrc(ps + 5 * ssX), w0);
                                    sum6 = vmlaq_f32(sum6, LoadSrc(ps + 6 * ssX), w0);
                                    sum7 = vmlaq_f32(sum7, LoadSrc(ps + 7 * ssX), w0);
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
                            float32x4_t sum0 = vdupq_n_f32(0.0f);
                            float32x4_t sum1 = vdupq_n_f32(0.0f);
                            float32x4_t sum2 = vdupq_n_f32(0.0f);
                            float32x4_t sum3 = vdupq_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vmlaq_f32(sum0, LoadSrc(ps + 0 * ssX), w0);
                                    sum1 = vmlaq_f32(sum1, LoadSrc(ps + 1 * ssX), w0);
                                    sum2 = vmlaq_f32(sum2, LoadSrc(ps + 2 * ssX), w0);
                                    sum3 = vmlaq_f32(sum3, LoadSrc(ps + 3 * ssX), w0);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                            Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                            Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            float32x4_t sum0 = vdupq_n_f32(0.0f);
                            float32x4_t sum1 = vdupq_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vmlaq_f32(sum0, LoadSrc(ps + 0 * ssX), w0);
                                    sum1 = vmlaq_f32(sum1, LoadSrc(ps + 1 * ssX), w0);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            float32x4_t sum = vdupq_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum = vmlaq_f32(sum, LoadSrc(ps), w0);
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
                        {
                            float32x4_t sum = vdupq_n_f32(0.0f);
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
                                        sum = vmlaq_f32(sum, LoadSrc(ps), Load<false>(pw));
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
                            float32x4_t sum = vdupq_n_f32(0.0f);
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
                                            sum = vmlaq_f32(sum, LoadSrc(ps), Load<false>(pw));
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
            const T* src1, size_t sX, const float32x4_t* weight, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 1 * sX), weight[4]), sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
                sum0 = vmlaq_f32(sum0, LoadSrc(src0 + 0 * sX), weight[0]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src0 + 1 * sX), weight[1]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src1 + 0 * sX), weight[3]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src1 + 1 * sX), weight[4]);
                Save1<term, type>(dst, NULL, vaddq_f32(sum0, sum1), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const float32x4_t* weight, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 2 * sX), weight[2]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 2 * sX), weight[5]), sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
                sum0 = vmlaq_f32(sum0, LoadSrc(src0 + 0 * sX), weight[0]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src0 + 1 * sX), weight[1]);
                sum2 = vmlaq_f32(sum2, LoadSrc(src0 + 2 * sX), weight[2]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src1 + 0 * sX), weight[3]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src1 + 1 * sX), weight[4]);
                sum2 = vmlaq_f32(sum2, LoadSrc(src1 + 2 * sX), weight[5]);
                Save1<term, type>(dst, NULL, vaddq_f32(vaddq_f32(sum0, sum1), sum2), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const float32x4_t* weight, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src2 + 0 * sX), weight[6]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src2 + 1 * sX), weight[7]), sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
                sum0 = vmlaq_f32(sum0, LoadSrc(src0 + 0 * sX), weight[0]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src0 + 1 * sX), weight[1]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src1 + 0 * sX), weight[3]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src1 + 1 * sX), weight[4]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src2 + 0 * sX), weight[6]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src2 + 1 * sX), weight[7]);
                Save1<term, type>(dst, NULL, vaddq_f32(sum0, sum1), bias, params);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const float32x4_t* weight, const float32x4_t* bias, const float32x4_t* params, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src0 + 2 * sX), weight[2]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src1 + 2 * sX), weight[5]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src2 + 0 * sX), weight[6]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src2 + 1 * sX), weight[7]), sum);
                sum = vaddq_f32(vmulq_f32(LoadSrc(src2 + 2 * sX), weight[8]), sum);
                Save1<term, type>(dst, NULL, sum, bias, params);
            }
            else
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
                sum0 = vmlaq_f32(sum0, LoadSrc(src0 + 0 * sX), weight[0]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src0 + 1 * sX), weight[1]);
                sum2 = vmlaq_f32(sum2, LoadSrc(src0 + 2 * sX), weight[2]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src1 + 0 * sX), weight[3]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src1 + 1 * sX), weight[4]);
                sum2 = vmlaq_f32(sum2, LoadSrc(src1 + 2 * sX), weight[5]);
                sum0 = vmlaq_f32(sum0, LoadSrc(src2 + 0 * sX), weight[6]);
                sum1 = vmlaq_f32(sum1, LoadSrc(src2 + 1 * sX), weight[7]);
                sum2 = vmlaq_f32(sum2, LoadSrc(src2 + 2 * sX), weight[8]);
                Save1<term, type>(dst, NULL, vaddq_f32(vaddq_f32(sum0, sum1), sum2), bias, params);
            }
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

            float32x4_t _params[2], _bias[1];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                float32x4_t _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = Load<false>(weight + i * F);
                _bias[0] = Load<false>(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + c);

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
                if (p.srcT == SimdTensorData16b)
                    depthwise = DepthwiseConvolution<uint16_t, term, type>;
                else
                    depthwise = DepthwiseConvolution<float, term, type>;
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
