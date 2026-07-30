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
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;

        //---------------------------------------------------------------------

        template<bool nofma> SIMD_INLINE svfloat32_t Madd(const svbool_t& mask, const svfloat32_t& sum, const float* src, const svfloat32_t& weight)
        {
            if (nofma)
                return svadd_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src), weight), sum);
            else
                return svmla_f32_x(mask, sum, svld1_f32(mask, src), weight);
        }

        template<Term8iType term, SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam& p, const AlgParam& a, size_t dstC,
            size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            const size_t F = svcntw();
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F;
            size_t dstCF = AlignLoAny(dstC, F);
            const svbool_t body = svptrue_b32();

            svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(0.0f);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                param1 = svdup_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                size_t tail = Simd::Min(F, dstC - c);
                svbool_t mask = c == dstCF ? svwhilelt_b32((size_t)0, tail) : body;
                svfloat32_t _bias = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
                if (type == ::SimdConvolutionActivationPrelu)
                    param0 = svld1_f32(mask, params + c);
                svfloat32_t _scale = svld1_f32(mask, scale + c);
                svfloat32_t _shift = svld1_f32(mask, shift + c);

                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                    {
                        svfloat32_t sum = _bias;
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
                                        sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps), svld1_f32(mask, pw));
                                    }
                                }
                            }
                        }
                        Save1<term, type>(pd, sum, param0, param1, _scale, _shift, a.upper, tail);
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const float* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            const size_t F = svcntw();
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            const svbool_t body = svptrue_b32();

            svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(0.0f);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                param1 = svdup_n_f32(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                svfloat32_t weight0 = svld1_f32(body, weight + 0 * F);
                svfloat32_t weight1 = svld1_f32(body, weight + 1 * F);
                svfloat32_t weight2 = svld1_f32(body, weight + 2 * F);
                svfloat32_t weight3 = svld1_f32(body, weight + 3 * F);
                svfloat32_t weight4 = svld1_f32(body, weight + 4 * F);
                svfloat32_t weight5 = svld1_f32(body, weight + 5 * F);
                svfloat32_t weight6 = svld1_f32(body, weight + 6 * F);
                svfloat32_t weight7 = svld1_f32(body, weight + 7 * F);
                svfloat32_t weight8 = svld1_f32(body, weight + 8 * F);
                svfloat32_t _bias = bias ? svld1_f32(body, bias + c) : svdup_n_f32(0.0f);
                if (type == ::SimdConvolutionActivationPrelu)
                    param0 = svld1_f32(body, params + c);
                svfloat32_t _scale = svld1_f32(body, scale + c);
                svfloat32_t _shift = svld1_f32(body, shift + c);

                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                    {
                        svfloat32_t sum = _bias;
#define SIMD_SVE2_MERGED_DEPTHWISE_3X3(KY, KX, WEIGHT) \
                        { \
                            size_t sy = dy * strideY + (KY) - padY; \
                            if (sy < p.srcH) \
                            { \
                                size_t sx = dx * strideX + (KX) - padX; \
                                if (sx < p.srcW) \
                                { \
                                    const float* ps = src + (sy & sM) * sY + sx * sX; \
                                    sum = Madd<nofma>(body, sum, ps, WEIGHT); \
                                } \
                            } \
                        }
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(0, 0, weight0);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(0, 1, weight1);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(0, 2, weight2);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(1, 0, weight3);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(1, 1, weight4);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(1, 2, weight5);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(2, 0, weight6);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(2, 1, weight7);
                        SIMD_SVE2_MERGED_DEPTHWISE_3X3(2, 2, weight8);
#undef SIMD_SVE2_MERGED_DEPTHWISE_3X3
                        Save1<term, type>(pd, sum, param0, param1, _scale, _shift, a.upper);
                    }
                }
                src += sD;
                dst += dD;
                weight += 9 * F;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwiseConvolutionPtr& depthwise)
        {
            const size_t F = svcntw();
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
