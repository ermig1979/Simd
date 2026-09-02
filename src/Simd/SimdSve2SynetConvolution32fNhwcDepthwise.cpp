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
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        namespace
        {
            SIMD_INLINE svfloat32_t Exp2(const svbool_t& mask, svfloat32_t x)
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

            SIMD_INLINE svfloat32_t Exp(const svbool_t& mask, svfloat32_t value)
            {
                return Exp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
            }

            SIMD_INLINE svfloat32_t Log2(const svbool_t& mask, svfloat32_t x)
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

            SIMD_INLINE svfloat32_t Log(const svbool_t& mask, svfloat32_t value)
            {
                return svmul_n_f32_x(mask, Log2(mask, value), 0.693147181f);
            }

            SIMD_INLINE svfloat32_t Erf(const svbool_t& mask, svfloat32_t x)
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

            SIMD_INLINE svfloat32_t Tanh(const svbool_t& mask, svfloat32_t x)
            {
                svfloat32_t e = Exp(mask, svmul_n_f32_x(mask, x, -2.0f));
                return svsub_n_f32_x(mask, svdiv_f32_x(mask, svdup_n_f32(2.0f), svadd_n_f32_x(mask, e, 1.0f)), 1.0f);
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask);

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationIdentity>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return value;
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmax_n_f32_x(mask, value, 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationLeakyRelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmla_n_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), svmin_n_f32_x(mask, value, 0.0f), params[0]);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRestrictRange>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmin_n_f32_x(mask, svmax_n_f32_x(mask, value, params[0]), params[1]);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationPrelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), svld1_f32(mask, params + offset), svmin_n_f32_x(mask, value, 0.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationElu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t neg = svmul_n_f32_x(mask, svsub_n_f32_x(mask, Exp(mask, value), 1.0f), params[0]);
                return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHswish>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t shift = svdup_n_f32(params[0]);
                svfloat32_t scale = svdup_n_f32(params[1]);
                svfloat32_t upper = svmin_f32_x(mask, value, shift);
                svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, shift), 0.0f);
                return svmul_f32_x(mask, svmul_f32_x(mask, positive, scale), value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationMish>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t exp = svmin_f32_x(mask, Exp(mask, value), svdup_n_f32(params[0]));
                return svmul_f32_x(mask, value, Tanh(mask, Log(mask, svadd_n_f32_x(mask, exp, 1.0f))));
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_n_f32_x(mask, svdup_n_f32(params[1]), value, params[0]), 1.0f), 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t exp = Exp(mask, svneg_f32_x(mask, svmul_n_f32_x(mask, value, params[0])));
                return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationGelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t t = svmul_n_f32_x(mask, value, 0.70710678118654752440f);
                return svmul_f32_x(mask, svmul_n_f32_x(mask, t, 0.70710678118654752440f), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwiseDefault(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            size_t size = p.group;
            size_t size2F = AlignLo(size, 2 * F);
            size_t size4F = AlignLo(size, 4 * F);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < size4F; i += 4 * F)
                    {
                        svfloat32_t sum0 = bias ? svld1_f32(body, bias + i + 0 * F) : svdup_n_f32(0.0f);
                        svfloat32_t sum1 = bias ? svld1_f32(body, bias + i + 1 * F) : svdup_n_f32(0.0f);
                        svfloat32_t sum2 = bias ? svld1_f32(body, bias + i + 2 * F) : svdup_n_f32(0.0f);
                        svfloat32_t sum3 = bias ? svld1_f32(body, bias + i + 3 * F) : svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * size + i;
                                        const float* ps = src + (sy * p.srcW + sx) * size + i;
                                        sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps + 0 * F), svld1_f32(body, pw + 0 * F));
                                        sum1 = svmla_f32_x(body, sum1, svld1_f32(body, ps + 1 * F), svld1_f32(body, pw + 1 * F));
                                        sum2 = svmla_f32_x(body, sum2, svld1_f32(body, ps + 2 * F), svld1_f32(body, pw + 2 * F));
                                        sum3 = svmla_f32_x(body, sum3, svld1_f32(body, ps + 3 * F), svld1_f32(body, pw + 3 * F));
                                    }
                                }
                            }
                        }
                        svst1_f32(body, dst + i + 0 * F, Activate<type>(sum0, params, i + 0 * F, body));
                        svst1_f32(body, dst + i + 1 * F, Activate<type>(sum1, params, i + 1 * F, body));
                        svst1_f32(body, dst + i + 2 * F, Activate<type>(sum2, params, i + 2 * F, body));
                        svst1_f32(body, dst + i + 3 * F, Activate<type>(sum3, params, i + 3 * F, body));
                    }
                    for (; i < size2F; i += 2 * F)
                    {
                        svfloat32_t sum0 = bias ? svld1_f32(body, bias + i + 0 * F) : svdup_n_f32(0.0f);
                        svfloat32_t sum1 = bias ? svld1_f32(body, bias + i + 1 * F) : svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * size + i;
                                        const float* ps = src + (sy * p.srcW + sx) * size + i;
                                        sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps + 0 * F), svld1_f32(body, pw + 0 * F));
                                        sum1 = svmla_f32_x(body, sum1, svld1_f32(body, ps + 1 * F), svld1_f32(body, pw + 1 * F));
                                    }
                                }
                            }
                        }
                        svst1_f32(body, dst + i + 0 * F, Activate<type>(sum0, params, i + 0 * F, body));
                        svst1_f32(body, dst + i + 1 * F, Activate<type>(sum1, params, i + 1 * F, body));
                    }
                    for (; i < size; i += F)
                    {
                        svbool_t mask = svwhilelt_b32((uint32_t)i, (uint32_t)size);
                        svfloat32_t sum = bias ? svld1_f32(mask, bias + i) : svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * size + i;
                                        const float* ps = src + (sy * p.srcW + sx) * size + i;
                                        sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps), svld1_f32(mask, pw));
                                    }
                                }
                            }
                        }
                        svst1_f32(mask, dst + i, Activate<type>(sum, params, i, mask));
                    }
                    dst += p.dstC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Edge(const float* src, const ConvParam& p, size_t dy, size_t dx, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            size_t srcC = p.srcC;
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t sum = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float* pw = weight + (ky * 3 + kx) * srcC + c;
                                const float* ps = src + (sy * p.srcW + sx) * srcC + c;
                                sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps), svld1_f32(mask, pw));
                            }
                        }
                    }
                }
                svst1_f32(mask, dst + c, Activate<type>(sum, params, c, mask));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main1(const float* src, size_t srcS, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t sum = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float* ps = src + ky * srcS + c;
                    const float* pw = weight + ky * 3 * srcC + c;
                    sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps + 0 * srcC), svld1_f32(mask, pw + 0 * srcC));
                    sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps + 1 * srcC), svld1_f32(mask, pw + 1 * srcC));
                    sum = svmla_f32_x(mask, sum, svld1_f32(mask, ps + 2 * srcC), svld1_f32(mask, pw + 2 * srcC));
                }
                svst1_f32(mask, dst + c, Activate<type>(sum, params, c, mask));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main2(const float* src, size_t srcS, size_t srcX, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t sum0 = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
                svfloat32_t sum1 = sum0;
                const float* pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float* ps0 = src + ky * srcS + c;
                    const float* ps1 = ps0 + srcX;
                    svfloat32_t w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + 0 * srcC), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + 0 * srcC), w0);
                    pw += srcC;
                    w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + 1 * srcC), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + 1 * srcC), w0);
                    pw += srcC;
                    w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + 2 * srcC), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + 2 * srcC), w0);
                    pw += srcC;
                }
                svst1_f32(mask, dst + c, Activate<type>(sum0, params, c, mask));
                svst1_f32(mask, dst + c + srcC, Activate<type>(sum1, params, c, mask));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main4(const float* src, size_t srcS, size_t srcX, size_t srcC, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t sum0 = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
                svfloat32_t sum1 = sum0;
                svfloat32_t sum2 = sum0;
                svfloat32_t sum3 = sum0;
                const float* pw = weight + c;
                const float* ps0 = src + 0 * srcX + c;
                const float* ps1 = src + 1 * srcX + c;
                const float* ps2 = src + 2 * srcX + c;
                const float* ps3 = src + 3 * srcX + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    svfloat32_t w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + offset), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + offset), w0);
                    sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, ps2 + offset), w0);
                    sum3 = svmla_f32_x(mask, sum3, svld1_f32(mask, ps3 + offset), w0);
                    pw += srcC;
                    offset += srcC;
                    w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + offset), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + offset), w0);
                    sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, ps2 + offset), w0);
                    sum3 = svmla_f32_x(mask, sum3, svld1_f32(mask, ps3 + offset), w0);
                    pw += srcC;
                    offset += srcC;
                    w0 = svld1_f32(mask, pw);
                    sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps0 + offset), w0);
                    sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, ps1 + offset), w0);
                    sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, ps2 + offset), w0);
                    sum3 = svmla_f32_x(mask, sum3, svld1_f32(mask, ps3 + offset), w0);
                    pw += srcC;
                }
                svst1_f32(mask, dst + 0 * srcC + c, Activate<type>(sum0, params, c, mask));
                svst1_f32(mask, dst + 1 * srcC + c, Activate<type>(sum1, params, c, mask));
                svst1_f32(mask, dst + 2 * srcC + c, Activate<type>(sum2, params, c, mask));
                svst1_f32(mask, dst + 3 * srcC + c, Activate<type>(sum3, params, c, mask));
            }
        }

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise3x3(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t srcS = p.srcC * p.srcW;
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;

            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW4; dx += 4)
                    Convolution32fNhwcDepthwise3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * p.dstC, offset += 4 * srcX;
                for (; dx < dstW2; dx += 2)
                    Convolution32fNhwcDepthwise3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * p.dstC, offset += 2 * srcX;
                for (; dx < dstW; ++dx)
                    Convolution32fNhwcDepthwise3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += p.dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
        }

        //-------------------------------------------------------------------------------------------------

        template <::SimdConvolutionActivationType type> SynetConvolution32fNhwcDepthwise::ConvolutionPtr Get(const ConvParam& p)
        {
            if (p.IsKernel(3) && p.IsDilation(1))
                return Convolution32fNhwcDepthwise3x3<type>;
            else
                return Convolution32fNhwcDepthwiseDefault<type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fNhwcDepthwise::SynetConvolution32fNhwcDepthwise(const ConvParam& p)
            : Neon::SynetConvolution32fNhwcDepthwise(p)
        {
            const size_t F = svcntw();
            if (p.dstC >= F && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: _convolution = Get<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: _convolution = Get<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: _convolution = Get<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: _convolution = Get<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: _convolution = Get<::SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: _convolution = Get<::SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: _convolution = Get<::SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: _convolution = Get<::SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: _convolution = Get<::SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: _convolution = Get<::SimdConvolutionActivationSwish>(p); break;
                case ::SimdConvolutionActivationGelu: _convolution = Get<::SimdConvolutionActivationGelu>(p); break;
                }
            }
        }
    }
#endif
}
