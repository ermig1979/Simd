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

            template<::SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params);

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationIdentity>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return value;
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRelu>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return svmax_n_f32_x(mask, value, 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationLeakyRelu>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), params[0], svmin_n_f32_x(mask, value, 0.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRestrictRange>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return svmin_f32_x(mask, svmax_f32_x(mask, params[0], value), params[1]);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationPrelu>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), params[0], svmin_n_f32_x(mask, value, 0.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationElu>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                svfloat32_t neg = svmul_f32_x(mask, params[0], svsub_n_f32_x(mask, Exp(mask, value), 1.0f));
                return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHswish>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                svfloat32_t upper = svmin_f32_x(mask, value, params[0]);
                svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, params[0]), 0.0f);
                return svmul_f32_x(mask, svmul_f32_x(mask, positive, params[1]), value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationMish>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                svfloat32_t _1 = svdup_n_f32(1.0f);
                svfloat32_t mish = svadd_f32_x(mask, Exp(mask, value), _1);
                mish = svmla_f32_x(mask, _1, mish, mish);
                mish = svmul_f32_x(mask, value, svsub_f32_x(mask, _1, svdiv_f32_x(mask, svdup_n_f32(2.0f), mish)));
                return svsel_f32(svcmpgt_f32(mask, params[0], value), mish, value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHardSigmoid>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, params[1], value, params[0]), 1.0f), 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationSwish>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                svfloat32_t exp = Exp(mask, svneg_f32_x(mask, svmul_f32_x(mask, value, params[0])));
                return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationGelu>(const svbool_t& mask, svfloat32_t value, const svfloat32_t* params)
            {
                svfloat32_t t = svmul_n_f32_x(mask, value, 0.70710678118654752440f);
                return svmul_f32_x(mask, svmul_n_f32_x(mask, t, 0.70710678118654752440f), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam & p)
            : Neon::SynetConvolution32fDirectNchw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, svfloat32_t * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = svdup_n_f32(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svmul_f32_x(mask, svld1_f32(mask, src), weight[0]);
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const svfloat32_t * weight)
            {
                return svmla_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src + 0), weight[0]), svld1_f32(mask, src + 1), weight[1]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight), RowConv(mask, src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const svfloat32_t * weight)
            {
                svuint32_t index = svindex_u32(0, 2);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                return svmla_f32_x(mask, svmul_f32_x(mask, s0, weight[0]), s1, weight[1]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight), RowConv(mask, src + step, weight + 2));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const svfloat32_t * weight)
            {
                return svmla_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src), weight[0]),
                    svld1_f32(mask, src + 1), weight[1]), svld1_f32(mask, src + 2), weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight),
                    svadd_f32_x(mask, RowConv(mask, src + step, weight + 3),
                        RowConv(mask, src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const svfloat32_t * weight)
            {
                svuint32_t index = svindex_u32(0, 2);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                svfloat32_t s2 = svld1_gather_u32index_f32(mask, src + 2, index);
                return svmla_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, s0, weight[0]), s1, weight[1]), s2, weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight),
                    svadd_f32_x(mask, RowConv(mask, src + step, weight + 3),
                        RowConv(mask, src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 3>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const svfloat32_t * weight)
            {
                svuint32_t index = svindex_u32(0, 3);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                svfloat32_t s2 = svld1_gather_u32index_f32(mask, src + 2, index);
                return svmla_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, s0, weight[0]), s1, weight[1]), s2, weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const svfloat32_t * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight),
                    svadd_f32_x(mask, RowConv(mask, src + step, weight + 3),
                        RowConv(mask, src + 2 * step, weight + 6)));
            }
        };

        template<int kernel, int stride, ::SimdConvolutionActivationType type>
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight,
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            const size_t F = svcntw();
            svfloat32_t _weight[kernel * kernel];
            svfloat32_t _params[2];
            _params[0] = svdup_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = svdup_n_f32(params[1]);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = svdup_n_f32(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel * kernel>(weight, _weight);
                    svfloat32_t _bias = bias ? svdup_n_f32(bias[dc]) : svdup_n_f32(0.0f);
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        for (size_t x = 0; x < dstW; x += F)
                        {
                            svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                            svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, _weight);
                            svst1_f32(mask, pd + x, Activate<type>(mask, svadd_f32_x(mask, _bias, conv), _params));
                        }
                        ps += srcW * stride;
                        pd += dstW;
                    }
                    weight += kernel * kernel;
                }
                else
                {
                    size_t sc = 0;
                    for (; sc < 1; ++sc)
                    {
                        const float * ps = src;
                        float * pd = dst;
                        LoadWeight<kernel * kernel>(weight, _weight);
                        svfloat32_t _bias = bias ? svdup_n_f32(bias[dc]) : svdup_n_f32(0.0f);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, _weight);
                                svst1_f32(mask, pd + x, svadd_f32_x(mask, _bias, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC - 1; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel * kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t _dst = svld1_f32(mask, pd + x);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, _weight);
                                svst1_f32(mask, pd + x, svadd_f32_x(mask, _dst, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel * kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t _dst = svld1_f32(mask, pd + x);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, _weight);
                                svst1_f32(mask, pd + x, Activate<type>(mask, svadd_f32_x(mask, _dst, conv), _params));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                }
                dst += dstH * dstW;
            }
        }

        bool SynetConvolution32fDirectNchw::Preferable(const ConvParam & p)
        {
            return Neon::SynetConvolution32fDirectNchw::Preferable(p);
        }

        template <int kernel, int stride> SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationIdentity>;
            case ::SimdConvolutionActivationRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRelu>;
            case ::SimdConvolutionActivationLeakyRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationLeakyRelu>;
            case ::SimdConvolutionActivationRestrictRange: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRestrictRange>;
            case ::SimdConvolutionActivationPrelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationPrelu>;
            case ::SimdConvolutionActivationElu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationElu>;
            case ::SimdConvolutionActivationHswish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationHswish>;
            case ::SimdConvolutionActivationMish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationMish>;
            case ::SimdConvolutionActivationHardSigmoid: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationHardSigmoid>;
            case ::SimdConvolutionActivationSwish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationSwish>;
            case ::SimdConvolutionActivationGelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationGelu>;
            default:
                assert(0);
                return NULL;
            }
        }

        SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SynetConvolution32fDirectNchw::SetConvolutionBiasActivation()
        {
            const ConvParam & p = _param;
            const size_t F = svcntw();
            if (p.dstW < F)
                return Neon::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Sve2::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Sve2::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Sve2::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Sve2::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Sve2::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Sve2::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            default:
                return Neon::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
            }
            assert(0);
            return NULL;
        }
    }
#endif
}
