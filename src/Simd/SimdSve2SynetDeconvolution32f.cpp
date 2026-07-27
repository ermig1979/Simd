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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam& p)
            : Base::SynetDeconvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Sve2::Gemm32fNN, "Sve2"));
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDeconvolution32fGemmNN::RowToImg(const float* src, float* dst)
        {
            const DeconvParam& p = _param;
            assert(p.trans && p.group == 1);
            if (p.IsPad(0) && p.IsDilation(1) && p.kernelY == p.strideX && p.kernelX == p.strideX)
            {
                Base::SynetDeconvolution32fGemmNN::RowToImg(src, dst);
                return;
            }
            else
            {
                const size_t F = svcntw();
                for (size_t dy = 0; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        memset(dst + (dy * p.dstW + dx) * p.dstC, 0, p.dstC * sizeof(float));
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY - p.padY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = sx * p.strideX - p.padX;
                                for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                                {
                                    if (dx < p.dstW)
                                    {
                                        float* d = dst + (dy * p.dstW + dx) * p.dstC;
                                        for (size_t dc = 0; dc < p.dstC; dc += F)
                                        {
                                            svbool_t mask = svwhilelt_b32(dc, p.dstC);
                                            svst1_f32(mask, d + dc, svadd_f32_x(mask, svld1_f32(mask, d + dc), svld1_f32(mask, src + dc)));
                                        }
                                    }
                                    src += p.dstC;
                                }
                            }
                            else
                                src += p.kernelX * p.dstC;
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t DeconvExp2(const svbool_t& mask, svfloat32_t x)
        {
            x = svmax_f32_x(mask, svmin_f32_x(mask, x, svdup_n_f32(126.99999f)), svdup_n_f32(-126.99999f));
            svint32_t ipart = svcvt_s32_f32_x(mask, svsub_n_f32_x(mask, x, 0.5f));
            svfloat32_t fpart = svsub_f32_x(mask, x, svcvt_f32_s32_x(mask, ipart));
            svfloat32_t expipart = svreinterpret_f32_s32(svlsl_n_s32_x(mask, svadd_n_s32_x(mask, ipart, 127), 23));
            svfloat32_t p = svdup_n_f32(1.8775767e-3f);
            p = svmla_f32_x(mask, svdup_n_f32(8.9893397e-3f), fpart, p);
            p = svmla_f32_x(mask, svdup_n_f32(5.5826318e-2f), fpart, p);
            p = svmla_f32_x(mask, svdup_n_f32(2.4015361e-1f), fpart, p);
            p = svmla_f32_x(mask, svdup_n_f32(6.9315308e-1f), fpart, p);
            p = svmla_f32_x(mask, svdup_n_f32(9.9999994e-1f), fpart, p);
            return svmul_f32_x(mask, expipart, p);
        }

        SIMD_INLINE svfloat32_t DeconvExp(const svbool_t& mask, svfloat32_t value)
        {
            return DeconvExp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
        }

        SIMD_INLINE svfloat32_t DeconvLog2(const svbool_t& mask, svfloat32_t x)
        {
            svuint32_t i = svreinterpret_u32_f32(x);
            svint32_t e32 = svsub_n_s32_x(mask, svreinterpret_s32_u32(svlsr_n_u32_x(mask, svand_n_u32_x(mask, i, 0x7F800000), 23)), 127);
            svfloat32_t e = svcvt_f32_s32_x(mask, e32);
            svfloat32_t one = svdup_n_f32(1.0f);
            svfloat32_t m = svreinterpret_f32_u32(svorr_u32_x(mask, svand_n_u32_x(mask, i, 0x007FFFFF), svreinterpret_u32_f32(one)));
            svfloat32_t p = svdup_n_f32(-3.4436006e-2f);
            p = svmla_f32_x(mask, svdup_n_f32(3.1821337e-1f), m, p);
            p = svmla_f32_x(mask, svdup_n_f32(-1.2315303f), m, p);
            p = svmla_f32_x(mask, svdup_n_f32(2.5988452f), m, p);
            p = svmla_f32_x(mask, svdup_n_f32(-3.3241990f), m, p);
            p = svmla_f32_x(mask, svdup_n_f32(3.1157899f), m, p);
            return svmla_f32_x(mask, e, p, svsub_f32_x(mask, m, one));
        }

        SIMD_INLINE svfloat32_t DeconvLog(const svbool_t& mask, svfloat32_t value)
        {
            return svmul_n_f32_x(mask, DeconvLog2(mask, value), 0.693147181f);
        }

        SIMD_INLINE svfloat32_t DeconvErf(const svbool_t& mask, svfloat32_t x)
        {
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            svfloat32_t a = svmin_f32_x(mask, svabs_f32_x(mask, x), svdup_n_f32(9.0f));
            svfloat32_t p = svdup_n_f32(0.0000430638f);
            p = svmla_f32_x(mask, svdup_n_f32(0.0002765672f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0001520143f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0092705272f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0422820123f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0705230784f), a, p);
            p = svmla_f32_x(mask, _1, a, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            svfloat32_t r = svsub_f32_x(mask, _1, svdiv_f32_x(mask, _1, p));
            return svsel_f32(svcmplt_n_f32(mask, x, 0.0f), svneg_f32_x(mask, r), r);
        }

        SIMD_INLINE svfloat32_t DeconvTanh(const svbool_t& mask, svfloat32_t x)
        {
            svfloat32_t e = DeconvExp(mask, svmul_n_f32_x(mask, x, -2.0f));
            return svsub_n_f32_x(mask, svdiv_f32_x(mask, svdup_n_f32(2.0f), svadd_n_f32_x(mask, e, 1.0f)), 1.0f);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask);

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationIdentity>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return value;
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svmax_n_f32_x(mask, value, 0.0f);
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationLeakyRelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRestrictRange>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svmin_f32_x(mask, svmax_f32_x(mask, param0, value), param1);
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationPrelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationElu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            svfloat32_t neg = svmul_f32_x(mask, param0, svsub_n_f32_x(mask, DeconvExp(mask, value), 1.0f));
            return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHswish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            svfloat32_t upper = svmin_f32_x(mask, value, param0);
            svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, param0), 0.0f);
            return svmul_f32_x(mask, svmul_f32_x(mask, positive, param1), value);
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationMish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            svfloat32_t exp = svmin_f32_x(mask, DeconvExp(mask, value), param0);
            return svmul_f32_x(mask, value, DeconvTanh(mask, DeconvLog(mask, svadd_n_f32_x(mask, exp, 1.0f))));
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, param1, value, param0), 1.0f), 0.0f);
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, DeconvExp(mask, svmul_f32_x(mask, svneg_f32_x(mask, value), param0)), 1.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationGelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
        {
            svfloat32_t t = svmul_n_f32_x(mask, value, 0.70710678118654752440f);
            return svmul_f32_x(mask, svmul_n_f32_x(mask, t, 0.70710678118654752440f), svadd_n_f32_x(mask, DeconvErf(mask, t), 1.0f));
        }

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask);
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
            {
                svst1_f32(mask, ptr, Activate<type>(svadd_f32_x(mask, value, bias), param0, param1, mask));
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
            {
                svst1_f32(mask, ptr, value);
            }
        };

        typedef void(*DeconvolutionNhwcDirect2x2_Ptr) (const float* src0, const DeconvParam& p, size_t srcC, size_t dstC,
            const float* weight, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, float* ds, int first);

        template<TermType term, SimdConvolutionActivationType type, size_t tail> void DeconvolutionNhwcDirect2x2_M(const float* src0,
            const DeconvParam& p, size_t srcC, size_t dstC, const float* weight0, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, float* dst, int first)
        {
            const size_t dS = p.srcC, dD = p.dstC, F = svcntw();
            const float* weight1 = weight0 + srcC * F, * src1, * src2, * src3, * src4, * src5;
            if (tail > 1) src1 = src0 + 1 * dS;
            if (tail > 2) src2 = src0 + 2 * dS;
            if (tail > 3) src3 = src0 + 3 * dS;
            if (tail > 4) src4 = src0 + 4 * dS;
            if (tail > 5) src5 = src0 + 5 * dS;
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, w0, w1, s0;
            svbool_t mask = svwhilelt_b32(size_t(0), dstC);
            if (first)
            {
                if (tail > 0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                if (tail > 1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                if (tail > 2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                if (tail > 3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                if (tail > 4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                if (tail > 5) d50 = svdup_n_f32(0.0f), d51 = svdup_n_f32(0.0f);
            }
            else
            {
                if (tail > 0) d00 = svld1_f32(mask, dst + 0x0 * dD), d01 = svld1_f32(mask, dst + 0x1 * dD);
                if (tail > 1) d10 = svld1_f32(mask, dst + 0x2 * dD), d11 = svld1_f32(mask, dst + 0x3 * dD);
                if (tail > 2) d20 = svld1_f32(mask, dst + 0x4 * dD), d21 = svld1_f32(mask, dst + 0x5 * dD);
                if (tail > 3) d30 = svld1_f32(mask, dst + 0x6 * dD), d31 = svld1_f32(mask, dst + 0x7 * dD);
                if (tail > 4) d40 = svld1_f32(mask, dst + 0x8 * dD), d41 = svld1_f32(mask, dst + 0x9 * dD);
                if (tail > 5) d50 = svld1_f32(mask, dst + 0xA * dD), d51 = svld1_f32(mask, dst + 0xB * dD);
            }
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = svld1_f32(mask, weight0);
                w1 = svld1_f32(mask, weight1);
                if (tail > 0) s0 = svdup_n_f32(src0[sc]), d00 = svmla_f32_x(mask, d00, s0, w0), d01 = svmla_f32_x(mask, d01, s0, w1);
                if (tail > 1) s0 = svdup_n_f32(src1[sc]), d10 = svmla_f32_x(mask, d10, s0, w0), d11 = svmla_f32_x(mask, d11, s0, w1);
                if (tail > 2) s0 = svdup_n_f32(src2[sc]), d20 = svmla_f32_x(mask, d20, s0, w0), d21 = svmla_f32_x(mask, d21, s0, w1);
                if (tail > 3) s0 = svdup_n_f32(src3[sc]), d30 = svmla_f32_x(mask, d30, s0, w0), d31 = svmla_f32_x(mask, d31, s0, w1);
                if (tail > 4) s0 = svdup_n_f32(src4[sc]), d40 = svmla_f32_x(mask, d40, s0, w0), d41 = svmla_f32_x(mask, d41, s0, w1);
                if (tail > 5) s0 = svdup_n_f32(src5[sc]), d50 = svmla_f32_x(mask, d50, s0, w0), d51 = svmla_f32_x(mask, d51, s0, w1);
                weight0 += F;
                weight1 += F;
            }
            if (tail > 0) Term<term>::template Save<type>(dst + 0x0 * dD, d00, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0x1 * dD, d01, bias, param0, param1, mask);
            if (tail > 1) Term<term>::template Save<type>(dst + 0x2 * dD, d10, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0x3 * dD, d11, bias, param0, param1, mask);
            if (tail > 2) Term<term>::template Save<type>(dst + 0x4 * dD, d20, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0x5 * dD, d21, bias, param0, param1, mask);
            if (tail > 3) Term<term>::template Save<type>(dst + 0x6 * dD, d30, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0x7 * dD, d31, bias, param0, param1, mask);
            if (tail > 4) Term<term>::template Save<type>(dst + 0x8 * dD, d40, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0x9 * dD, d41, bias, param0, param1, mask);
            if (tail > 5) Term<term>::template Save<type>(dst + 0xA * dD, d50, bias, param0, param1, mask), Term<term>::template Save<type>(dst + 0xB * dD, d51, bias, param0, param1, mask);
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

        template<TermType term, SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float* src, const DeconvParam& p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            const size_t F = svcntw(), body = 6, srcWb = AlignLoAny(p.srcW, body), tail = p.srcW - srcWb;
            DeconvolutionNhwcDirect2x2_Ptr bodyKernel = GetDeconvolutionNhwcDirect2x2<term, type>(body);
            DeconvolutionNhwcDirect2x2_Ptr tailKernel = GetDeconvolutionNhwcDirect2x2<term, type>(tail);
            svbool_t mask = svwhilelt_b32(size_t(0), dstC);
            svfloat32_t _param0, _param1, _bias;

            _param0 = svdup_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _param1 = svdup_n_f32(params[1]);
            else
                _param1 = svdup_n_f32(0.0f);

            for (size_t dc = 0; dc < dstC; dc += F)
            {
                size_t dC = Simd::Min(F, dstC - dc);
                mask = svwhilelt_b32(size_t(0), dC);
                _bias = svld1_f32(mask, bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _param0 = svld1_f32(mask, params + dc);
                const float* s = src + yBeg * p.srcW * p.srcC;
                float* d = dst + yBeg * p.strideY * p.dstW * p.dstC;
                const float* w0 = weight + 0 * p.kernelX * srcC * F;
                const float* w1 = weight + 1 * p.kernelX * srcC * F;
                for (size_t sy = yBeg; sy < yEnd; sy += 1, s += p.srcW * p.srcC)
                {
                    for (size_t sx = 0; sx < srcWb; sx += body)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w0, _bias, _param0, _param1, d, first), d += body * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + srcWb * p.srcC, p, srcC, dC, w0, _bias, _param0, _param1, d, first), d += tail * p.strideX * p.dstC;
                    for (size_t sx = 0; sx < srcWb; sx += body)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w1, _bias, _param0, _param1, d, first), d += body * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + srcWb * p.srcC, p, srcC, dC, w1, _bias, _param0, _param1, d, first), d += tail * p.strideX * p.dstC;
                }
                weight += p.kernelY * p.kernelX * srcC * F;
                dst += F;
            }
        }

        template<SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float* src, const DeconvParam& p,
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

        SynetDeconvolution32fNhwcDirect2x2::SynetDeconvolution32fNhwcDirect2x2(const DeconvParam& p)
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
            case SimdConvolutionActivationGelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationGelu>; break;
            default: assert(0);
            }
            SetAlgParam(svcntw(), Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
        }

        bool SynetDeconvolution32fNhwcDirect2x2::Preferable(const DeconvParam& p)
        {
            return p.IsPad(0) && p.IsDilation(1) && p.IsKernel(2) && p.IsStride(2) && p.group == 1 && p.trans;
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            DeconvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f))
                return NULL;
            if (SynetDeconvolution32fNhwcDirect2x2::Preferable(param))
                return new SynetDeconvolution32fNhwcDirect2x2(param);
            else
                return new SynetDeconvolution32fGemmNN(param);
        }
    }
#endif
}
