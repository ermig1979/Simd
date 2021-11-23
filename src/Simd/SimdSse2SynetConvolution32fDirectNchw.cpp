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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse2
    {
        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam32f & p)
            : Base::SynetConvolution32fDirectNchw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m128 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_mul_ps(_mm_loadu_ps(src), weight[0]);
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE __m128 RowConv(const float * src, const __m128  * weight)
            {
                return _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0), weight[0]), _mm_mul_ps(_mm_loadu_ps(src + 1), weight[1]));
            }

            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE __m128 RowConv(const float * src, const __m128  * weight)
            {
                __m128 s0 = _mm_loadu_ps(src + 0);
                __m128 s1 = _mm_loadu_ps(src + F);
                return _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s0, s1, 0x88), weight[0]), 
                    _mm_mul_ps(_mm_shuffle_ps(s0, s1, 0xDD), weight[1]));
            }

            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE __m128 RowConv(const float * src, const __m128  * weight)
            {
                return _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src), weight[0]),
                    _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 1), weight[1]),
                        _mm_mul_ps(_mm_loadu_ps(src + 2), weight[2])));
            }

            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight),
                    _mm_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE __m128 RowConv(const float * src, const __m128  * weight)
            {
                __m128 s00 = _mm_loadu_ps(src);
                __m128 s10 = _mm_loadu_ps(src + F);
                __m128 s02 = _mm_loadu_ps(src + 2);
                __m128 s12 = _mm_loadu_ps(src + 2 + F);
                return _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s00, s10, 0x88), weight[0]),
                    _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s00, s10, 0xDD), weight[1]),
                        _mm_mul_ps(_mm_shuffle_ps(s02, s12, 0x88), weight[2])));
            }

            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight),
                    _mm_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 3>
        {
            static SIMD_INLINE __m128 RowConv(const float * src, const __m128  * weight)
            {
                __m128 s0 = _mm_shuffle_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(src + 6), 0xCC);
                __m128 s1 = _mm_shuffle_ps(_mm_loadu_ps(src + 1), _mm_loadu_ps(src + 7), 0xCC);
                __m128 s2 = _mm_shuffle_ps(_mm_loadu_ps(src + 2), _mm_loadu_ps(src + 8), 0xCC);
                return _mm_add_ps(_mm_mul_ps(s0, weight[0]), _mm_add_ps(_mm_mul_ps(s1, weight[1]), _mm_mul_ps(s2, weight[2])));
            }

            static SIMD_INLINE __m128 SynetConvolution32f(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight), _mm_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const __m128 * params);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const __m128 * params)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const __m128 * params)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const __m128 * params)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const __m128 * params)
        {
            return _mm_min_ps(_mm_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const __m128 * params)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationElu>(__m128 value, const __m128 * params)
        {
            return Sse2::Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHswish>(__m128 value, const __m128 * params)
        {
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationMish>(__m128 value, const __m128* params)
        {
            return Sse2::Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHardSigmoid>(__m128 value, const __m128* params)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationSwish>(__m128 value, const __m128* params)
        {
            return Sse2::Swish(value, params[0]);
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m128 _weight[kernel*kernel];
            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm_set1_ps(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            __m128 tail = RightNotZero32f(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_set1_ps(params[dc]);
                if(srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel*kernel>(weight, _weight);
                    __m128 _bias = bias ? _mm_set1_ps(bias[dc]) : _mm_setzero_ps();
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        for (size_t x = 0; x < dstWF; x += F)
                        {
                            __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                            _mm_storeu_ps(pd + x, Activate<type>(_mm_add_ps(_bias, conv), _params));
                        }
                        if (dstWF < dstW)
                        {
                            size_t x = dstW - F;
                            __m128 _dst = _mm_loadu_ps(pd + x);
                            __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                            _mm_storeu_ps(pd + x, Combine(tail, Activate<type>(_mm_add_ps(_bias, conv), _params), _dst));
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
                        LoadWeight<kernel*kernel>(weight, _weight);
                        __m128 _bias = bias ? _mm_set1_ps(bias[dc]) : _mm_setzero_ps();
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstWF; x += F)
                            {
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, _mm_add_ps(_bias, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, Combine(tail, _mm_add_ps(_bias, conv), _dst));
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
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstWF; x += F)
                            {
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, _mm_add_ps(_dst, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, _mm_add_ps(_dst, _mm_and_ps(conv, tail)));
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
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstWF; x += F)
                            {
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, Activate<type>(_mm_add_ps(_dst, conv), _params));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, Combine(tail, Activate<type>(_mm_add_ps(_dst, conv), _params), _dst));
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

        bool SynetConvolution32fDirectNchw::Preferable(const ConvParam32f & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && ((p.IsStride(1) && p.IsKernel(1)) || p.IsKernel(2) || p.IsKernel(3)) && p.trans == 0;
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
            default:
                assert(0);
                return NULL;
            }
        }

        SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SynetConvolution32fDirectNchw::SetConvolutionBiasActivation()
        {
            const ConvParam32f & p = _param;
            if (p.dstW < F)
                return Base::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Sse2::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Sse2::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Sse2::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Sse2::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Sse2::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Sse2::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            default:
                return Base::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
            }
            return NULL;
        }
    }
#endif//SIMD_SSE2_ENABLE
}
