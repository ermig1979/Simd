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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx2
    {
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam32f & p)
            : Avx::SynetDeconvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Avx2::Gemm32fNN, "Avx2", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx2::Gemm32fNNcbBufferSize, Avx2::Gemm32fNNcbReorderB, Avx2::Gemm32fNNcbRun, "Avx2", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Avx2::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Avx2::Gemm32fNNcbRun;
                _nhwcReorderB = Avx2::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx2::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        typedef void(*DeconvolutionNhwcDirect2x2_Ptr) (const float * src0, const DeconvParam32f & p, size_t srcC, size_t dstC, 
            const float * weight, const __m256 * bias, const __m256 * params, float * ds, int first);

        template<TermType term, SimdConvolutionActivationType type, size_t tail> void DeconvolutionNhwcDirect2x2_M(const float * src0,
            const DeconvParam32f & p, size_t srcC, size_t dstC, const float * weight0, const __m256 * bias, const __m256 * params, float * dst, int first)
        {
            size_t dS = p.srcC, dD = p.dstC;
            const float * weight1 = weight0 + srcC * F, *src1, *src2, *src3, *src4, *src5;
            if (tail > 1) src1 = src0 + 1 * dS;
            if (tail > 2) src2 = src0 + 2 * dS;
            if (tail > 3) src3 = src0 + 3 * dS;
            if (tail > 4) src4 = src0 + 4 * dS;
            if (tail > 5) src5 = src0 + 5 * dS;
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            if (first)
            {
                if (tail > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                if (tail > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                if (tail > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                if (tail > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                if (tail > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                if (tail > 5) d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
            }
            else
            {
                if (tail > 0) d00 = _mm256_loadu_ps(dst + 0x0 * dD), d01 = _mm256_loadu_ps(dst + 0x1 * dD);
                if (tail > 1) d10 = _mm256_loadu_ps(dst + 0x2 * dD), d11 = _mm256_loadu_ps(dst + 0x3 * dD);
                if (tail > 2) d20 = _mm256_loadu_ps(dst + 0x4 * dD), d21 = _mm256_loadu_ps(dst + 0x5 * dD);
                if (tail > 3) d30 = _mm256_loadu_ps(dst + 0x6 * dD), d31 = _mm256_loadu_ps(dst + 0x7 * dD);
                if (tail > 4) d40 = _mm256_loadu_ps(dst + 0x8 * dD), d41 = _mm256_loadu_ps(dst + 0x9 * dD);
                if (tail > 5) d50 = _mm256_loadu_ps(dst + 0xa * dD), d51 = _mm256_loadu_ps(dst + 0xb * dD);
            }
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = _mm256_loadu_ps(weight0);
                w1 = _mm256_loadu_ps(weight1);
                if (tail > 0) s0 = _mm256_set1_ps(src0[sc]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
                if (tail > 1) s0 = _mm256_set1_ps(src1[sc]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
                if (tail > 2) s0 = _mm256_set1_ps(src2[sc]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
                if (tail > 3) s0 = _mm256_set1_ps(src3[sc]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
                if (tail > 4) s0 = _mm256_set1_ps(src4[sc]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
                if (tail > 5) s0 = _mm256_set1_ps(src5[sc]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
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

        template<TermType term, SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float* src, const DeconvParam32f& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t body = 6, srcWb = AlignLoAny(p.srcW, body), tail = p.srcW - srcWb;
            DeconvolutionNhwcDirect2x2_Ptr bodyKernel = GetDeconvolutionNhwcDirect2x2<term, type>(body);
            DeconvolutionNhwcDirect2x2_Ptr tailKernel = GetDeconvolutionNhwcDirect2x2<term, type>(tail);

            __m256 _params[2], _bias[1];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += F)
            {
                size_t dC = Simd::Min(F, dstC - dc);
                _bias[0] = _mm256_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm256_loadu_ps(params + dc);
                const float* s = src + yBeg * p.srcW * p.srcC;
                float* d = dst + yBeg * p.strideY * p.dstW * p.dstC;
                const float* w0 = weight + 0 * p.kernelX * srcC * F;
                const float* w1 = weight + 1 * p.kernelX * srcC * F;
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
                weight += p.kernelY * p.kernelX * srcC * F;
                dst += F;
            }
        }

        template<SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float* src, const DeconvParam32f& p,
            const SynetDeconvolution32fNhwcDirect2x2::AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
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
                    weight += AlignHiAny(macroD, a.microD) * macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        SynetDeconvolution32fNhwcDirect2x2::SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p)
            : Avx::SynetDeconvolution32fNhwcDirect2x2(p)
        {
            if (p.dstC > HF)
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
#endif//SIMD_AVX2_ENABLE
}
