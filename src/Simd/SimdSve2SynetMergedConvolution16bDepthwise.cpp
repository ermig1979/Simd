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
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sve2
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template<bool nofma> SIMD_INLINE svfloat32_t Madd(const svbool_t& mask, svfloat32_t sum, svfloat32_t src, svfloat32_t weight)
        {
            if (nofma)
                return svadd_f32_x(mask, svmul_f32_x(mask, src, weight), sum);
            else
                return svmla_f32_x(mask, sum, src, weight);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0, const svfloat32_t* bias, const svfloat32_t* params, const svbool_t& mask)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, mask);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, svfloat32_t val0, const svfloat32_t* bias, const svfloat32_t* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
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

            svfloat32_t _params[2], _bias[1];
            _params[0] = svdup_n_f32(params[0]);
            _params[1] = svdup_n_f32(params[1]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = svdup_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                _bias[0] = svld1_f32(body, bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = svld1_f32(body, params + c);
                if (c == dstCF)
                {
                    size_t tail = dstC - dstCF;
                    svbool_t mask = svwhilelt_b32((size_t)0, tail);
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            svfloat32_t sum = svdup_n_f32(0.0f);
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
                                            sum = Madd<nofma>(mask, sum, LoadSrc(ps, mask), svld1_f32(mask, pw));
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, mask);
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
                            svfloat32_t sum = svdup_n_f32(0.0f);
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
                                        sum = Madd<nofma>(body, sum, LoadSrc(ps, body), svld1_f32(body, pw));
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, body);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                        {
                            svfloat32_t sum0 = svdup_n_f32(0.0f);
                            svfloat32_t sum1 = svdup_n_f32(0.0f);
                            svfloat32_t sum2 = svdup_n_f32(0.0f);
                            svfloat32_t sum3 = svdup_n_f32(0.0f);
                            svfloat32_t sum4 = svdup_n_f32(0.0f);
                            svfloat32_t sum5 = svdup_n_f32(0.0f);
                            svfloat32_t sum6 = svdup_n_f32(0.0f);
                            svfloat32_t sum7 = svdup_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    svfloat32_t w0 = svld1_f32(body, pw);
                                    sum0 = Madd<nofma>(body, sum0, LoadSrc(ps + 0 * ssX, body), w0);
                                    sum1 = Madd<nofma>(body, sum1, LoadSrc(ps + 1 * ssX, body), w0);
                                    sum2 = Madd<nofma>(body, sum2, LoadSrc(ps + 2 * ssX, body), w0);
                                    sum3 = Madd<nofma>(body, sum3, LoadSrc(ps + 3 * ssX, body), w0);
                                    sum4 = Madd<nofma>(body, sum4, LoadSrc(ps + 4 * ssX, body), w0);
                                    sum5 = Madd<nofma>(body, sum5, LoadSrc(ps + 5 * ssX, body), w0);
                                    sum6 = Madd<nofma>(body, sum6, LoadSrc(ps + 6 * ssX, body), w0);
                                    sum7 = Madd<nofma>(body, sum7, LoadSrc(ps + 7 * ssX, body), w0);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params, body);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params, body);
                            Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params, body);
                            Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params, body);
                            Save1<term, type>(pd + 4 * dX, NULL, sum4, _bias, _params, body);
                            Save1<term, type>(pd + 5 * dX, NULL, sum5, _bias, _params, body);
                            Save1<term, type>(pd + 6 * dX, NULL, sum6, _bias, _params, body);
                            Save1<term, type>(pd + 7 * dX, NULL, sum7, _bias, _params, body);
                        }
                        for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                        {
                            svfloat32_t sum0 = svdup_n_f32(0.0f);
                            svfloat32_t sum1 = svdup_n_f32(0.0f);
                            svfloat32_t sum2 = svdup_n_f32(0.0f);
                            svfloat32_t sum3 = svdup_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    svfloat32_t w0 = svld1_f32(body, pw);
                                    sum0 = Madd<nofma>(body, sum0, LoadSrc(ps + 0 * ssX, body), w0);
                                    sum1 = Madd<nofma>(body, sum1, LoadSrc(ps + 1 * ssX, body), w0);
                                    sum2 = Madd<nofma>(body, sum2, LoadSrc(ps + 2 * ssX, body), w0);
                                    sum3 = Madd<nofma>(body, sum3, LoadSrc(ps + 3 * ssX, body), w0);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params, body);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params, body);
                            Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params, body);
                            Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params, body);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            svfloat32_t sum0 = svdup_n_f32(0.0f);
                            svfloat32_t sum1 = svdup_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    svfloat32_t w0 = svld1_f32(body, pw);
                                    sum0 = Madd<nofma>(body, sum0, LoadSrc(ps + 0 * ssX, body), w0);
                                    sum1 = Madd<nofma>(body, sum1, LoadSrc(ps + 1 * ssX, body), w0);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params, body);
                            Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params, body);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            svfloat32_t sum = svdup_n_f32(0.0f);
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    svfloat32_t w0 = svld1_f32(body, pw);
                                    sum = Madd<nofma>(body, sum, LoadSrc(ps, body), w0);
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, body);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
                        {
                            svfloat32_t sum = svdup_n_f32(0.0f);
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
                                        sum = Madd<nofma>(body, sum, LoadSrc(ps, body), svld1_f32(body, pw));
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, body);
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            svfloat32_t sum = svdup_n_f32(0.0f);
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
                                            sum = Madd<nofma>(body, sum, LoadSrc(ps, body), svld1_f32(body, pw));
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, body);
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
            const T* src1, size_t sX, const svfloat32_t* weight, const svfloat32_t* bias, const svfloat32_t* params, uint8_t* dst)
        {
            const svbool_t body = svptrue_b32();
            if (nofma)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 1 * sX, body), weight[4]);
                Save1<term, type>(dst, NULL, sum, bias, params, body);
            }
            else
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                sum0 = Madd<false>(body, sum0, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src1 + 1 * sX, body), weight[4]);
                Save1<term, type>(dst, NULL, svadd_f32_x(body, sum0, sum1), bias, params, body);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const svfloat32_t* weight, const svfloat32_t* bias, const svfloat32_t* params, uint8_t* dst)
        {
            const svbool_t body = svptrue_b32();
            if (nofma)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 2 * sX, body), weight[2]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 2 * sX, body), weight[5]);
                Save1<term, type>(dst, NULL, sum, bias, params, body);
            }
            else
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f), sum2 = svdup_n_f32(0.0f);
                sum0 = Madd<false>(body, sum0, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum2 = Madd<false>(body, sum2, LoadSrc(src0 + 2 * sX, body), weight[2]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum2 = Madd<false>(body, sum2, LoadSrc(src1 + 2 * sX, body), weight[5]);
                Save1<term, type>(dst, NULL, svadd_f32_x(body, svadd_f32_x(body, sum0, sum1), sum2), bias, params, body);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const svfloat32_t* weight, const svfloat32_t* bias, const svfloat32_t* params, uint8_t* dst)
        {
            const svbool_t body = svptrue_b32();
            if (nofma)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum = Madd<true>(body, sum, LoadSrc(src2 + 0 * sX, body), weight[6]);
                sum = Madd<true>(body, sum, LoadSrc(src2 + 1 * sX, body), weight[7]);
                Save1<term, type>(dst, NULL, sum, bias, params, body);
            }
            else
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                sum0 = Madd<false>(body, sum0, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src2 + 0 * sX, body), weight[6]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src2 + 1 * sX, body), weight[7]);
                Save1<term, type>(dst, NULL, svadd_f32_x(body, sum0, sum1), bias, params, body);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const svfloat32_t* weight, const svfloat32_t* bias, const svfloat32_t* params, uint8_t* dst)
        {
            const svbool_t body = svptrue_b32();
            if (nofma)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum = Madd<true>(body, sum, LoadSrc(src0 + 2 * sX, body), weight[2]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum = Madd<true>(body, sum, LoadSrc(src1 + 2 * sX, body), weight[5]);
                sum = Madd<true>(body, sum, LoadSrc(src2 + 0 * sX, body), weight[6]);
                sum = Madd<true>(body, sum, LoadSrc(src2 + 1 * sX, body), weight[7]);
                sum = Madd<true>(body, sum, LoadSrc(src2 + 2 * sX, body), weight[8]);
                Save1<term, type>(dst, NULL, sum, bias, params, body);
            }
            else
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f), sum2 = svdup_n_f32(0.0f);
                sum0 = Madd<false>(body, sum0, LoadSrc(src0 + 0 * sX, body), weight[0]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src0 + 1 * sX, body), weight[1]);
                sum2 = Madd<false>(body, sum2, LoadSrc(src0 + 2 * sX, body), weight[2]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src1 + 0 * sX, body), weight[3]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src1 + 1 * sX, body), weight[4]);
                sum2 = Madd<false>(body, sum2, LoadSrc(src1 + 2 * sX, body), weight[5]);
                sum0 = Madd<false>(body, sum0, LoadSrc(src2 + 0 * sX, body), weight[6]);
                sum1 = Madd<false>(body, sum1, LoadSrc(src2 + 1 * sX, body), weight[7]);
                sum2 = Madd<false>(body, sum2, LoadSrc(src2 + 2 * sX, body), weight[8]);
                Save1<term, type>(dst, NULL, svadd_f32_x(body, svadd_f32_x(body, sum0, sum1), sum2), bias, params, body);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = maC;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            svfloat32_t _params[2], _bias[1];
            _params[0] = svdup_n_f32(params[0]);
            _params[1] = svdup_n_f32(params[1]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = svdup_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                svfloat32_t _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = svld1_f32(body, weight + i * F);
                _bias[0] = svld1_f32(body, bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = svld1_f32(body, params + c);

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
            const size_t F = svcntw();
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
