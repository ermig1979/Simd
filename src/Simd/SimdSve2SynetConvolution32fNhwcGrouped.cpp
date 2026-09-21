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
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        namespace
        {
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
                svfloat32_t neg = svmul_n_f32_x(mask, svsub_n_f32_x(mask, Exponent(mask, value), 1.0f), params[0]);
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
                svfloat32_t exp = svmin_f32_x(mask, Exponent(mask, value), svdup_n_f32(params[0]));
                return svmul_f32_x(mask, value, Tanh(mask, Logarithm(mask, svadd_n_f32_x(mask, exp, 1.0f))));
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_n_f32_x(mask, svdup_n_f32(params[1]), value, params[0]), 1.0f), 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t exp = Exponent(mask, svneg_f32_x(mask, svmul_n_f32_x(mask, value, params[0])));
                return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationGelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return Gelu(mask, value);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveResult(svfloat32_t sum0, svfloat32_t sum1, const float* bias, const float* params, size_t offset, float* dst)
        {
            const svbool_t mask = svptrue_b32();
            const size_t F = svcntw();
            svst1_f32(mask, dst + offset + 0, Activate<type>(svadd_f32_x(mask, svzip1_f32(sum0, sum1), svld1_f32(mask, bias + offset + 0)), params, offset + 0, mask));
            svst1_f32(mask, dst + offset + F, Activate<type>(svadd_f32_x(mask, svzip2_f32(sum0, sum1), svld1_f32(mask, bias + offset + F)), params, offset + F, mask));
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcGroupedBlock1x2Default(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            const size_t F = svcntw();
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcC2F = AlignLo(srcC, 2 * F);
            size_t srcC4F = AlignLo(srcC, 4 * F);
            size_t dW = p.kernelY * p.kernelX * p.srcC;
            const svbool_t mask = svptrue_b32();
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t c = 0;
                    for (; c < srcC4F; c += 4 * F)
                    {
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        svfloat32_t sum00 = svdup_n_f32(0.0f);
                        svfloat32_t sum01 = svdup_n_f32(0.0f);
                        svfloat32_t sum02 = svdup_n_f32(0.0f);
                        svfloat32_t sum03 = svdup_n_f32(0.0f);
                        svfloat32_t sum10 = svdup_n_f32(0.0f);
                        svfloat32_t sum11 = svdup_n_f32(0.0f);
                        svfloat32_t sum12 = svdup_n_f32(0.0f);
                        svfloat32_t sum13 = svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, * pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        svfloat32_t s0 = svld1_f32(mask, ps0 + 0 * F);
                                        sum00 = svmla_f32_x(mask, sum00, s0, svld1_f32(mask, pw0 + 0 * F));
                                        sum10 = svmla_f32_x(mask, sum10, s0, svld1_f32(mask, pw1 + 0 * F));
                                        svfloat32_t s1 = svld1_f32(mask, ps0 + 1 * F);
                                        sum01 = svmla_f32_x(mask, sum01, s1, svld1_f32(mask, pw0 + 1 * F));
                                        sum11 = svmla_f32_x(mask, sum11, s1, svld1_f32(mask, pw1 + 1 * F));
                                        svfloat32_t s2 = svld1_f32(mask, ps0 + 2 * F);
                                        sum02 = svmla_f32_x(mask, sum02, s2, svld1_f32(mask, pw0 + 2 * F));
                                        sum12 = svmla_f32_x(mask, sum12, s2, svld1_f32(mask, pw1 + 2 * F));
                                        svfloat32_t s3 = svld1_f32(mask, ps0 + 3 * F);
                                        sum03 = svmla_f32_x(mask, sum03, s3, svld1_f32(mask, pw0 + 3 * F));
                                        sum13 = svmla_f32_x(mask, sum13, s3, svld1_f32(mask, pw1 + 3 * F));
                                    }
                                }
                            }
                        }
                        size_t d = 2 * c;
                        SaveResult<type>(sum00, sum10, bias, params, d + 0 * F, dst);
                        SaveResult<type>(sum01, sum11, bias, params, d + 2 * F, dst);
                        SaveResult<type>(sum02, sum12, bias, params, d + 4 * F, dst);
                        SaveResult<type>(sum03, sum13, bias, params, d + 6 * F, dst);
                    }
                    for (; c < srcC2F; c += 2 * F)
                    {
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        svfloat32_t sum00 = svdup_n_f32(0.0f);
                        svfloat32_t sum01 = svdup_n_f32(0.0f);
                        svfloat32_t sum10 = svdup_n_f32(0.0f);
                        svfloat32_t sum11 = svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, *pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        svfloat32_t s0 = svld1_f32(mask, ps0 + 0 * F);
                                        sum00 = svmla_f32_x(mask, sum00, s0, svld1_f32(mask, pw0 + 0 * F));
                                        sum10 = svmla_f32_x(mask, sum10, s0, svld1_f32(mask, pw1 + 0 * F));
                                        svfloat32_t s1 = svld1_f32(mask, ps0 + 1 * F);
                                        sum01 = svmla_f32_x(mask, sum01, s1, svld1_f32(mask, pw0 + 1 * F));
                                        sum11 = svmla_f32_x(mask, sum11, s1, svld1_f32(mask, pw1 + 1 * F));
                                    }
                                }
                            }
                        }
                        size_t d = 2 * c;
                        SaveResult<type>(sum00, sum10, bias, params, d + 0 * F, dst);
                        SaveResult<type>(sum01, sum11, bias, params, d + 2 * F, dst);
                    }
                    for (; c < srcC; c += F)
                    {
                        c = c >= srcCF ? srcC - F : c;
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        svfloat32_t sum00 = svdup_n_f32(0.0f);
                        svfloat32_t sum10 = svdup_n_f32(0.0f);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, *pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        svfloat32_t s0 = svld1_f32(mask, ps0);
                                        sum00 = svmla_f32_x(mask, sum00, s0, svld1_f32(mask, pw0));
                                        sum10 = svmla_f32_x(mask, sum10, s0, svld1_f32(mask, pw1));
                                    }
                                }
                            }
                        }
                        SaveResult<type>(sum00, sum10, bias, params, c * 2, dst);
                    }
                    dst += p.dstC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SynetConvolution32fNhwcGroupedBlock1x2::ConvolutionPtr GetConvolution(const ConvParam& p)
        {
            return ConvolutionNhwcGroupedBlock1x2Default<type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fNhwcGroupedBlock1x2::SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p)
            : Base::SynetConvolution32fNhwcGroupedBlock1x2(p)
        {
            const size_t F = svcntw();
            if (p.srcC >= F)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: _convolution = GetConvolution<SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: _convolution = GetConvolution<SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: _convolution = GetConvolution<SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: _convolution = GetConvolution<SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: _convolution = GetConvolution<SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: _convolution = GetConvolution<SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: _convolution = GetConvolution<SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: _convolution = GetConvolution<SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: _convolution = GetConvolution<SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: _convolution = GetConvolution<SimdConvolutionActivationSwish>(p); break;
                case ::SimdConvolutionActivationGelu: _convolution = GetConvolution<SimdConvolutionActivationGelu>(p); break;
                }
            }
        }
    }
#endif
}
