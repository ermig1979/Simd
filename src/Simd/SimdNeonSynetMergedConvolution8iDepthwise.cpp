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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam& p, const AlgParam& a, size_t dstC,
            size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX =  strideX * sX;
            size_t noseY = p.NoseH(), bodyY = p.BodyH(), noseX = p.NoseW(), bodyX = p.BodyW();
            size_t bodyS = bodyX > noseX ? bodyX - noseX : 0;
            size_t bodyX2 = AlignLo(bodyS, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyS, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyS, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F);

            uint8x8_t _upper = vdup_n_u8(a.upper);
            float32x4_t _params[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + c);
                float32x4_t _scale = Load<false>(scale + c);
                float32x4_t _shift = Load<false>(shift + c);

                if (c == dstCF)
                {
                    size_t tail = dstC - dstCF;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            float32x4_t sum = _bias;
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
                                            sum = vaddq_f32(vmulq_f32(Load<false>(ps), Load<false>(pw)), sum);
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
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    if (dy >= noseY && dy < bodyY)
                    {
                        size_t dx = 0;
                        for (; dx < noseX; dx += 1, pd += dX)
                        {
                            float32x4_t sum = _bias;
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
                                        sum = vaddq_f32(vmulq_f32(Load<false>(ps), Load<false>(pw)), sum);
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                        {
                            float32x4_t sum0 = _bias;
                            float32x4_t sum1 = _bias;
                            float32x4_t sum2 = _bias;
                            float32x4_t sum3 = _bias;
                            float32x4_t sum4 = _bias;
                            float32x4_t sum5 = _bias;
                            float32x4_t sum6 = _bias;
                            float32x4_t sum7 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vaddq_f32(vmulq_f32(Load<false>(ps + 0 * ssX), w0), sum0);
                                    sum1 = vaddq_f32(vmulq_f32(Load<false>(ps + 1 * ssX), w0), sum1);
                                    sum2 = vaddq_f32(vmulq_f32(Load<false>(ps + 2 * ssX), w0), sum2);
                                    sum3 = vaddq_f32(vmulq_f32(Load<false>(ps + 3 * ssX), w0), sum3);
                                    sum4 = vaddq_f32(vmulq_f32(Load<false>(ps + 4 * ssX), w0), sum4);
                                    sum5 = vaddq_f32(vmulq_f32(Load<false>(ps + 5 * ssX), w0), sum5);
                                    sum6 = vaddq_f32(vmulq_f32(Load<false>(ps + 6 * ssX), w0), sum6);
                                    sum7 = vaddq_f32(vmulq_f32(Load<false>(ps + 7 * ssX), w0), sum7);
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
                            float32x4_t sum0 = _bias;
                            float32x4_t sum1 = _bias;
                            float32x4_t sum2 = _bias;
                            float32x4_t sum3 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vaddq_f32(vmulq_f32(Load<false>(ps + 0 * ssX), w0), sum0);
                                    sum1 = vaddq_f32(vmulq_f32(Load<false>(ps + 1 * ssX), w0), sum1);
                                    sum2 = vaddq_f32(vmulq_f32(Load<false>(ps + 2 * ssX), w0), sum2);
                                    sum3 = vaddq_f32(vmulq_f32(Load<false>(ps + 3 * ssX), w0), sum3);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            float32x4_t sum0 = _bias;
                            float32x4_t sum1 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum0 = vaddq_f32(vmulq_f32(Load<false>(ps + 0 * ssX), w0), sum0);
                                    sum1 = vaddq_f32(vmulq_f32(Load<false>(ps + 1 * ssX), w0), sum1);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            float32x4_t sum = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    float32x4_t w0 = Load<false>(pw);
                                    sum = vaddq_f32(vmulq_f32(Load<false>(ps), w0), sum);
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
                        {
                            float32x4_t sum = _bias;
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
                                        sum = vaddq_f32(vmulq_f32(Load<false>(ps), Load<false>(pw)), sum);
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
                            float32x4_t sum = _bias;
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
                                            sum = vaddq_f32(vmulq_f32(Load<false>(ps), Load<false>(pw)), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
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
            size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, const float32x4_t& scale, const float32x4_t& shift, uint8x8_t upper, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = bias;
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum1);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum1);
                Save1<term, type>(dst, vaddq_f32(sum0, sum1), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const float* src0, const float* src1,
            size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, const float32x4_t& scale, const float32x4_t& shift, uint8x8_t upper, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = bias;
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 2 * sX), weight[2]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 2 * sX), weight[5]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0), sum2 = vdupq_n_f32(0);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum1);
                sum2 = vaddq_f32(vmulq_f32(Load<false>(src0 + 2 * sX), weight[2]), sum2);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum1);
                sum2 = vaddq_f32(vmulq_f32(Load<false>(src1 + 2 * sX), weight[5]), sum2);
                Save1<term, type>(dst, vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const float* src0, const float* src1, const float* src2, 
            size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, const float32x4_t& scale, const float32x4_t& shift, uint8x8_t upper, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = bias;
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src2 + 0 * sX), weight[6]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src2 + 1 * sX), weight[7]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum1);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum1);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src2 + 0 * sX), weight[6]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src2 + 1 * sX), weight[7]), sum1);
                Save1<term, type>(dst, vaddq_f32(sum0, sum1), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const float* src0, const float* src1, const float* src2,
            size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, const float32x4_t& scale, const float32x4_t& shift, uint8x8_t upper, uint8_t* dst)
        {
            if (nofma)
            {
                float32x4_t sum = bias;
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src0 + 2 * sX), weight[2]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src1 + 2 * sX), weight[5]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src2 + 0 * sX), weight[6]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src2 + 1 * sX), weight[7]), sum);
                sum = vaddq_f32(vmulq_f32(Load<false>(src2 + 2 * sX), weight[8]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0), sum2 = vdupq_n_f32(0);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src0 + 0 * sX), weight[0]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src0 + 1 * sX), weight[1]), sum1);
                sum2 = vaddq_f32(vmulq_f32(Load<false>(src0 + 2 * sX), weight[2]), sum2);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src1 + 0 * sX), weight[3]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src1 + 1 * sX), weight[4]), sum1);
                sum2 = vaddq_f32(vmulq_f32(Load<false>(src1 + 2 * sX), weight[5]), sum2);
                sum0 = vaddq_f32(vmulq_f32(Load<false>(src2 + 0 * sX), weight[6]), sum0);
                sum1 = vaddq_f32(vmulq_f32(Load<false>(src2 + 1 * sX), weight[7]), sum1);
                sum2 = vaddq_f32(vmulq_f32(Load<false>(src2 + 2 * sX), weight[8]), sum2);
                Save1<term, type>(dst, vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const float* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX)*sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            uint8x8_t _upper = vdup_n_u8(a.upper);
            float32x4_t _params[2];
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
                float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + c);
                float32x4_t _scale = Load<false>(scale + c);
                float32x4_t _shift = Load<false>(shift + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
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
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && Aligned(p.dstC, F))
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = DepthwiseConvolution3x3<term, type, true>;
                else
                    depthwise = DepthwiseConvolution3x3<term, type, false>;
            }
            else
                depthwise = DepthwiseConvolution<term, type>;
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.dstT == SimdTensorData32f)
                SetDepthwise<Term8iLast32f, type>(p, depthwise);
            else
                SetDepthwise<Term8iLast8u, type>(p, depthwise);
        }

        void SetDepthwise(const ConvParam& p, DepthwiseConvolutionPtr& depthwise)
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
