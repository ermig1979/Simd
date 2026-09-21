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
            template<::SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1);

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationIdentity>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return value;
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRelu>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return svmax_n_f32_x(mask, value, 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationLeakyRelu>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRestrictRange>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return svmin_f32_x(mask, svmax_f32_x(mask, param0, value), param1);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationPrelu>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationElu>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                svfloat32_t neg = svmul_f32_x(mask, param0, svsub_n_f32_x(mask, Exponent(mask, value), 1.0f));
                return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHswish>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                svfloat32_t upper = svmin_f32_x(mask, value, param0);
                svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, param0), 0.0f);
                return svmul_f32_x(mask, svmul_f32_x(mask, positive, param1), value);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationMish>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return Mish(mask, value, param0);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHardSigmoid>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, param1, value, param0), 1.0f), 0.0f);
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationSwish>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                svfloat32_t exp = Exponent(mask, svneg_f32_x(mask, svmul_f32_x(mask, value, param0)));
                return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
            }

            template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationGelu>(const svbool_t& mask, svfloat32_t value, svfloat32_t param0, svfloat32_t param1)
            {
                return Gelu(mask, value);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam & p)
            : Neon::SynetConvolution32fDirectNchw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template<int kernel, int stride> struct Kernel
        {
            static svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
            {
                return svmul_n_f32_x(mask, svld1_f32(mask, src), weight[0]);
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const float * weight)
            {
                return svmla_n_f32_x(mask, svmul_n_f32_x(mask, svld1_f32(mask, src + 0), weight[0]), svld1_f32(mask, src + 1), weight[1]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight), RowConv(mask, src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const float * weight)
            {
                svuint32_t index = svindex_u32(0, 2);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                return svmla_n_f32_x(mask, svmul_n_f32_x(mask, s0, weight[0]), s1, weight[1]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight), RowConv(mask, src + step, weight + 2));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const float * weight)
            {
                return svmla_n_f32_x(mask, svmla_n_f32_x(mask, svmul_n_f32_x(mask, svld1_f32(mask, src), weight[0]),
                    svld1_f32(mask, src + 1), weight[1]), svld1_f32(mask, src + 2), weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight),
                    svadd_f32_x(mask, RowConv(mask, src + step, weight + 3),
                        RowConv(mask, src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const float * weight)
            {
                svuint32_t index = svindex_u32(0, 2);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                svfloat32_t s2 = svld1_gather_u32index_f32(mask, src + 2, index);
                return svmla_n_f32_x(mask, svmla_n_f32_x(mask, svmul_n_f32_x(mask, s0, weight[0]), s1, weight[1]), s2, weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
            {
                return svadd_f32_x(mask, RowConv(mask, src, weight),
                    svadd_f32_x(mask, RowConv(mask, src + step, weight + 3),
                        RowConv(mask, src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 3>
        {
            static SIMD_INLINE svfloat32_t RowConv(const svbool_t & mask, const float * src, const float * weight)
            {
                svuint32_t index = svindex_u32(0, 3);
                svfloat32_t s0 = svld1_gather_u32index_f32(mask, src + 0, index);
                svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, index);
                svfloat32_t s2 = svld1_gather_u32index_f32(mask, src + 2, index);
                return svmla_n_f32_x(mask, svmla_n_f32_x(mask, svmul_n_f32_x(mask, s0, weight[0]), s1, weight[1]), s2, weight[2]);
            }

            static SIMD_INLINE svfloat32_t SynetConvolution32f(const svbool_t & mask, const float * src, size_t step, const float * weight)
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
            svfloat32_t param0 = svdup_n_f32(params[0]);
            svfloat32_t param1 = svdup_n_f32(0.0f);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                param1 = svdup_n_f32(params[1]);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    param0 = svdup_n_f32(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    const float * pw = weight;
                    float * pd = dst;
                    svfloat32_t _bias = bias ? svdup_n_f32(bias[dc]) : svdup_n_f32(0.0f);
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        for (size_t x = 0; x < dstW; x += F)
                        {
                            svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                            svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, pw);
                            svst1_f32(mask, pd + x, Activate<type>(mask, svadd_f32_x(mask, _bias, conv), param0, param1));
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
                        const float * pw = weight;
                        float * pd = dst;
                        svfloat32_t _bias = bias ? svdup_n_f32(bias[dc]) : svdup_n_f32(0.0f);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, pw);
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
                        const float * pw = weight;
                        float * pd = dst;
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t _dst = svld1_f32(mask, pd + x);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, pw);
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
                        const float * pw = weight;
                        float * pd = dst;
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstW; x += F)
                            {
                                svbool_t mask = svwhilelt_b32((uint32_t)x, (uint32_t)dstW);
                                svfloat32_t _dst = svld1_f32(mask, pd + x);
                                svfloat32_t conv = Kernel<kernel, stride>::SynetConvolution32f(mask, ps + x * stride, srcW, pw);
                                svst1_f32(mask, pd + x, Activate<type>(mask, svadd_f32_x(mask, _dst, conv), param0, param1));
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

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam& p)
            : Neon::SynetConvolution32fDepthwiseDotProduct(p)
        {
        }

        SIMD_INLINE void DotProduct(const float* a, const float* b, const svbool_t& mask, svfloat32_t& sum)
        {
            svfloat32_t _a = svld1_f32(mask, a);
            svfloat32_t _b = svld1_f32(mask, b);
            sum = svmla_f32_m(mask, sum, _a, _b);
        }

        SIMD_INLINE float DotProduct(const float* a, const float* b, size_t size)
        {
            const size_t F = svcntw();
            const size_t QF = 4 * F;
            const svbool_t body = svptrue_b32();
            size_t sizeQF = AlignLo(size, QF);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            svfloat32_t sums0 = svdup_n_f32(0.0f), sums1 = svdup_n_f32(0.0f);
            svfloat32_t sums2 = svdup_n_f32(0.0f), sums3 = svdup_n_f32(0.0f);
            for (; i < sizeQF; i += QF)
            {
                DotProduct(a + i + 0 * F, b + i + 0 * F, body, sums0);
                DotProduct(a + i + 1 * F, b + i + 1 * F, body, sums1);
                DotProduct(a + i + 2 * F, b + i + 2 * F, body, sums2);
                DotProduct(a + i + 3 * F, b + i + 3 * F, body, sums3);
            }
            sums0 = svadd_f32_x(body, svadd_f32_x(body, sums0, sums1), svadd_f32_x(body, sums2, sums3));
            for (; i < sizeF; i += F)
                DotProduct(a + i, b + i, body, sums0);
            if (i < size)
                DotProduct(a + i, b + i, svwhilelt_b32(i, size), sums0);
            return svaddv_f32(body, sums0);
        }

        void SynetConvolution32fDepthwiseDotProduct::Forward(const float* src, float* buf, float* dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_bias)
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size) + _bias[i];
                }
                else
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size);
                }
                if (_param.activation)
                    Neon::ConvolutionBiasAndActivation(NULL, _count, 1, _param.activation, _params, ::SimdFalse, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }
    }
#endif
}
