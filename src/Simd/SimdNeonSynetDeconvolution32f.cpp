/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam32f & p)
            : Base::SynetDeconvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Neon::Gemm32fNN, "Neon", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (NHWC_GEMM_RUNTIME)
                {
#if defined(SIMD_ARM64_ENABLE)
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF4));
#else
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF3));
#endif
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Neon::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Neon::Gemm32fNNcbRun;
                _nhwcReorderB = Neon::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        typedef void(*DeconvolutionNhwcDirect2x2_Ptr) (const float * src0, const DeconvParam32f & p, size_t srcC, size_t dstC, 
            const float * weight, const float32x4_t * bias, const float32x4_t * params, float * ds, int first);

        template<TermType term, SimdConvolutionActivationType type, size_t tail> void DeconvolutionNhwcDirect2x2_M(const float * src0,
            const DeconvParam32f & p, size_t srcC, size_t dstC, const float * weight0, const float32x4_t * bias, const float32x4_t * params, float * dst, int first)
        {
            size_t dS = p.srcC, dD = p.dstC;
            const float * weight1 = weight0 + srcC * F, *src1, *src2, *src3, *src4, *src5;
            if (tail > 1) src1 = src0 + 1 * dS;
            if (tail > 2) src2 = src0 + 2 * dS;
            if (tail > 3) src3 = src0 + 3 * dS;
            if (tail > 4) src4 = src0 + 4 * dS;
            if (tail > 5) src5 = src0 + 5 * dS;
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            if (tail > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
            if (tail > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
            if (tail > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
            if (tail > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
            if (tail > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
            if (tail > 5) d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = Load<false>(weight0);
                w1 = Load<false>(weight1);
                if (tail > 0) s0 = vld1q_dup_f32(src0 + sc), d00 = vmlaq_f32(d00, s0, w0), d01 = vmlaq_f32(d01, s0, w1);
                if (tail > 1) s0 = vld1q_dup_f32(src1 + sc), d10 = vmlaq_f32(d10, s0, w0), d11 = vmlaq_f32(d11, s0, w1);
                if (tail > 2) s0 = vld1q_dup_f32(src2 + sc), d20 = vmlaq_f32(d20, s0, w0), d21 = vmlaq_f32(d21, s0, w1);
                if (tail > 3) s0 = vld1q_dup_f32(src3 + sc), d30 = vmlaq_f32(d30, s0, w0), d31 = vmlaq_f32(d31, s0, w1);
                if (tail > 4) s0 = vld1q_dup_f32(src4 + sc), d40 = vmlaq_f32(d40, s0, w0), d41 = vmlaq_f32(d41, s0, w1);
                if (tail > 5) s0 = vld1q_dup_f32(src5 + sc), d50 = vmlaq_f32(d50, s0, w0), d51 = vmlaq_f32(d51, s0, w1);
                weight0 += F;
                weight1 += F;
            }
            if (dstC == F)
            {
                if (tail > 0) Term<term>::template Save<type, 0>(dst + 0x0 * dD, d00, bias, params), Term<term>::template Save<type, 0>(dst + 0x1 * dD, d01, bias, params);
                if (tail > 1) Term<term>::template Save<type, 0>(dst + 0x2 * dD, d10, bias, params), Term<term>::template Save<type, 0>(dst + 0x3 * dD, d11, bias, params);
                if (tail > 2) Term<term>::template Save<type, 0>(dst + 0x4 * dD, d20, bias, params), Term<term>::template Save<type, 0>(dst + 0x5 * dD, d21, bias, params);
                if (tail > 3) Term<term>::template Save<type, 0>(dst + 0x6 * dD, d30, bias, params), Term<term>::template Save<type, 0>(dst + 0x7 * dD, d31, bias, params);
                if (tail > 4) Term<term>::template Save<type, 0>(dst + 0x8 * dD, d40, bias, params), Term<term>::template Save<type, 0>(dst + 0x9 * dD, d41, bias, params);
                if (tail > 5) Term<term>::template Save<type, 0>(dst + 0xA * dD, d50, bias, params), Term<term>::template Save<type, 0>(dst + 0xB * dD, d51, bias, params);
            }
            else
            {
                if (tail > 0) Term<term>::template Save<type, 0>(dst + 0x0 * dD, d00, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0x1 * dD, d01, bias, params, dstC);
                if (tail > 1) Term<term>::template Save<type, 0>(dst + 0x2 * dD, d10, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0x3 * dD, d11, bias, params, dstC);
                if (tail > 2) Term<term>::template Save<type, 0>(dst + 0x4 * dD, d20, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0x5 * dD, d21, bias, params, dstC);
                if (tail > 3) Term<term>::template Save<type, 0>(dst + 0x6 * dD, d30, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0x7 * dD, d31, bias, params, dstC);
                if (tail > 4) Term<term>::template Save<type, 0>(dst + 0x8 * dD, d40, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0x9 * dD, d41, bias, params, dstC);
                if (tail > 5) Term<term>::template Save<type, 0>(dst + 0xA * dD, d50, bias, params, dstC), Term<term>::template Save<type, 0>(dst + 0xB * dD, d51, bias, params, dstC);
            }
        }

        template <TermType term, SimdConvolutionActivationType type> SIMD_INLINE DeconvolutionNhwcDirect2x2_Ptr GetDeconvolutionNhwcDirect2x2(size_t tail)
        {
            switch (tail)
            {
            case 0: return NULL;
            case 1: return DeconvolutionNhwcDirect2x2_M<term, type, 1>;
            case 2: return DeconvolutionNhwcDirect2x2_M<term, type, 2>;
            case 3: return DeconvolutionNhwcDirect2x2_M<term, type, 3>;
            case 4: return DeconvolutionNhwcDirect2x2_M<term, type, 4>;
            case 5: return DeconvolutionNhwcDirect2x2_M<term, type, 5>;
            case 6: return DeconvolutionNhwcDirect2x2_M<term, type, 6>;
            default:
                assert(0);
                return NULL;
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float * src, const DeconvParam32f & p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float * weight, const float * bias, const float * params, float * dst, int first)
        {
            size_t body = 6, srcWb = AlignLoAny(p.srcW, body), tail = p.srcW - srcWb;
            DeconvolutionNhwcDirect2x2_Ptr bodyKernel = GetDeconvolutionNhwcDirect2x2<term, type>(body);
            DeconvolutionNhwcDirect2x2_Ptr tailKernel = GetDeconvolutionNhwcDirect2x2<term, type>(tail);

            float32x4_t _params[2], _bias[1];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += F)
            {
                size_t dC = Simd::Min(F, dstC - dc);
                _bias[0] = Load<false>(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = Load<false>(params + dc);
                const float * s = src + yBeg * p.srcW * p.srcC;
                float * d = dst + yBeg * p.strideY * p.dstW * p.dstC;
                const float * w0 = weight + 0 * p.kernelX * srcC * F;
                const float * w1 = weight + 1 * p.kernelX * srcC * F;
                for (size_t sy = yBeg; sy < yEnd; sy += 1, s += p.srcW * p.srcC)
                {
                    for (size_t sx = 0; sx < srcWb; sx += body)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w0, _bias, _params, d, first), d += body * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + srcWb * p.srcC, p, srcC, dC, w0, _bias, _params, d, first), d += tail * p.strideX * p.dstC;
                    for (size_t sx = 0; sx < srcWb; sx += body)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w1, _bias, _params, d, first), d += body * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + srcWb * p.srcC, p, srcC, dC, w1, _bias, _params, d, first), d += tail * p.strideX * p.dstC;
                }
                weight += p.kernelY * p.kernelX*srcC*F;
                dst += F;
            }
        }

        template<SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float * src, const DeconvParam32f & p,
            const SynetDeconvolution32fNhwcDirect2x2::AlgParam & a, const float * weight, const float * bias, const float * params, float * dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    size_t macroK = p.kernelY * p.kernelX * macroC;
                    for (size_t yBeg = 0; yBeg < p.srcH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.srcH);
                        if (a.macroC == p.srcC)
                            DeconvolutionNhwcDirect2x2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 1);
                        else if (sc == 0)
                            DeconvolutionNhwcDirect2x2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 1);
                        else if (sc + macroC == p.srcC)
                            DeconvolutionNhwcDirect2x2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 0);
                        else
                            DeconvolutionNhwcDirect2x2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD)*macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        SynetDeconvolution32fNhwcDirect2x2::SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p)
            : Base::SynetDeconvolution32fNhwcDirect2x2(p)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationRelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationLeakyRelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationRestrictRange: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationSwish>; break;
            default: assert(0);
            }
            SetAlgParam(F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
        }

        bool SynetDeconvolution32fNhwcDirect2x2::Preferable(const DeconvParam32f & p)
        {
            return p.IsPad(0) && p.IsDilation(1) && p.IsKernel(2) && p.IsStride(2) && p.group == 1 && p.trans;
        }

        //---------------------------------------------------------------------

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            DeconvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            if (SynetDeconvolution32fNhwcDirect2x2::Preferable(param))
                return new SynetDeconvolution32fNhwcDirect2x2(param);
            else
                return new SynetDeconvolution32fGemmNN(param);
        }
    }
#endif//SIMD_NEON_ENABLE
}
