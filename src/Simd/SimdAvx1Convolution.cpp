/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdConvolution.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx1.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    SynetAddBias(bias, count, size, dst, trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m256 _0 = _mm256_set1_ps(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 _dst = _mm256_loadu_ps(dst + i);
                                __m256 _bias = _mm256_loadu_ps(bias + i);
                                _mm256_storeu_ps(dst + i, _mm256_max_ps(_0, _mm256_add_ps(_dst, _bias)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::Max(0.0f, dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 _dst = _mm256_loadu_ps(dst + j);
                                _mm256_storeu_ps(dst + j, _mm256_max_ps(_0, _mm256_add_ps(_dst, _bias)));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::Max(0.0f, dst[j] + bias[i]);
                            dst += size;
                        }
                    }
                }
                else
                {
                    float slope = 0;
                    NeuralRelu(dst, size*count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    __m256 _slope = _mm256_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(bias + i));
                                _mm256_storeu_ps(dst + i, SynetPreluLayerForward(value, _slope));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetPreluLayerForward(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, SynetPreluLayerForward(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetPreluLayerForward(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    NeuralRelu(dst, size*count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    __m256 _lower = _mm256_set1_ps(lower);
                    __m256 _upper = _mm256_set1_ps(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(dst + i));
                                _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_max_ps(_lower, value), _upper));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::RestrictRange(dst[i] + bias[i], lower, upper);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, _mm256_min_ps(_mm256_max_ps(_lower, value), _upper));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
                            dst += size;
                        }
                    }
                }
                else
                    SynetRestrictRange(dst, size*count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(bias + i));
                                _mm256_storeu_ps(dst + i, SynetPreluLayerForward(value, _mm256_loadu_ps(params + i)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetPreluLayerForward(dst[i] + bias[i], params[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            __m256 _slope = _mm256_set1_ps(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, SynetPreluLayerForward(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetPreluLayerForward(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    Avx::SynetPreluLayerForward(dst, params, count, size, dst, trans);
            }
        }

        //---------------------------------------------------------------------

        ConvolutionGemmNN::ConvolutionGemmNN(const ConvParam & p)
            : Sse::ConvolutionGemmNN(p)
        {
        }

        void ConvolutionGemmNN::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
            {
                if (p.srcT)
                    Avx::Gemm32fNN(_M, _N, _K, &_1, src + _grS * g, _ldS, _weight + _grW * g, _ldW, &_0, dst + _grD * g, _ldD);
                else
                    Avx::Gemm32fNN(_M, _N, _K, &_1, _weight + _grW * g, _ldW, src + _grS * g, _ldS, &_0, dst + _grD * g, _ldD);
            }
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.dstT, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionGemmNT::ConvolutionGemmNT(const ConvParam & p)
            : Sse3::ConvolutionGemmNT(p)
        {
        }

        void ConvolutionGemmNT::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Sse::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Avx::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Avx::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Avx::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirectChw::ConvolutionDirectChw(const ConvParam & p)
            : Sse::ConvolutionDirectChw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static __m256 Convolution(const float * src, size_t step, const __m256  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_mul_ps(_mm256_loadu_ps(src), weight[0]);
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                return _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(src + 0), weight[0]), _mm256_mul_ps(_mm256_loadu_ps(src + 1), weight[1]));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                return _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(src), weight[0]),
                    _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(src + 1), weight[1]),
                        _mm256_mul_ps(_mm256_loadu_ps(src + 2), weight[2])));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_add_ps(RowConv(src, weight),
                    _mm256_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const __m256 * params);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const __m256 * params)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const __m256 * params)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const __m256 * params)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(params[0], _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const __m256 * params)
        {
            return _mm256_min_ps(_mm256_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const __m256 * params)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(params[0], _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m256 _weight[kernel*kernel];
            __m256 _params[2];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm256_set1_ps(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            __m256 tail = RightNotZero(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm256_set1_ps(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel*kernel>(weight, _weight);
                    __m256 _bias = bias ? _mm256_set1_ps(bias[dc]) : _mm256_setzero_ps();
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        for (size_t x = 0; x < dstWF; x += F)
                        {
                            __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm256_storeu_ps(pd + x, Activate<type>(_mm256_add_ps(_bias, conv), _params));
                        }
                        if (dstWF < dstW)
                        {
                            size_t x = dstW - F;
                            __m256 _dst = _mm256_loadu_ps(pd + x);
                            __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm256_storeu_ps(pd + x, _mm256_blendv_ps(_dst, Activate<type>(_mm256_add_ps(_bias, conv), _params), tail));
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
                        __m256 _bias = bias ? _mm256_set1_ps(bias[dc]) : _mm256_setzero_ps();
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstWF; x += F)
                            {
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_add_ps(_bias, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_blendv_ps(_dst, _mm256_add_ps(_bias, conv), tail));
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
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_add_ps(_dst, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_add_ps(_dst, _mm256_and_ps(conv, tail)));
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
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, Activate<type>(_mm256_add_ps(_dst, conv), _params));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_blendv_ps(_dst, Activate<type>(_mm256_add_ps(_dst, conv), _params), tail));
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

        template <int kernel, int stride> ConvolutionDirectChw::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationIdentity>;
            case ::SimdConvolutionActivationRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRelu>;
            case ::SimdConvolutionActivationLeakyRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationLeakyRelu>;
            case ::SimdConvolutionActivationRestrictRange: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRestrictRange>;
            case ::SimdConvolutionActivationPrelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationPrelu>;
            default:
                assert(0);
                return NULL;
            }
        }

        ConvolutionDirectChw::ConvolutionBiasActivationPtr ConvolutionDirectChw::SetConvolutionBiasActivation()
        {
            const ConvParam & p = _param;
            if (p.dstW < F)
                return Sse::ConvolutionDirectChw::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Avx::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Avx::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Avx::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            }
            return Sse::ConvolutionDirectChw::SetConvolutionBiasActivation();
        }

        //---------------------------------------------------------------------

        ConvolutionDirectHwc::ConvolutionDirectHwc(const ConvParam & p)
            : Sse::ConvolutionDirectHwc(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        bool ConvolutionDirectHwc::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            if (!(p.group == 1 || p.IsDepthwise()))
                return false;
            double k = double(p.srcC) / p.group / p.kernelX / p.kernelY;
            return k < 2.0 && (p.IsKernel(1) || p.IsKernel(2) || p.IsKernel(3)) && p.IsHwc();
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const float * params, size_t offset);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_set1_ps(params[0]), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_min_ps(_mm256_max_ps(_mm256_set1_ps(params[0]), value), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_loadu_ps(params + offset), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<size_t stride, size_t height, size_t width>
        SIMD_INLINE void KernelHwcDefaultEdge1(const float * src, size_t srcW, size_t srcC, size_t dstC, const float * weight, __m256 & sum)
        {
            for (size_t ky = 0; ky < height; ++ky)
            {
                for (size_t kx = 0; kx < width; ++kx)
                {
                    const float * pw = weight + (ky*stride + kx)*srcC*dstC;
                    const float * ps = src + (ky*srcW + kx)*srcC;
                    for (size_t sc = 0; sc < srcC; ++sc, pw += dstC)
                        sum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ps[sc]), _mm256_loadu_ps(pw)), sum);
                }
            }
        }

        template<> SIMD_INLINE void KernelHwcDefaultEdge1<3, 3, 3>(const float * src, size_t srcW, size_t srcC, size_t dstC, const float * weight, __m256 & sum)
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (size_t ky = 0; ky < 3; ++ky)
            {
                for (size_t i = 0, n = 3 * srcC; i < n; i += 3, weight += 3 * dstC)
                {
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(src[i + 0]), _mm256_loadu_ps(weight + 0 * dstC)), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(src[i + 1]), _mm256_loadu_ps(weight + 1 * dstC)), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(src[i + 2]), _mm256_loadu_ps(weight + 2 * dstC)), sum2);
                }
                src += srcW * srcC;
            }
            sum = _mm256_add_ps(_mm256_add_ps(sum, sum0), _mm256_add_ps(sum1, sum2));
        }

        template<size_t stride, size_t height, size_t width, ::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, size_t srcW, size_t srcC, size_t dstC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dc = 0;
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m256 conv = bias ? _mm256_loadu_ps(bias + dc) : _mm256_setzero_ps();
                KernelHwcDefaultEdge1<stride, height, width>(src, srcW, srcC, dstC, weight + dc, conv);
                _mm256_storeu_ps(dst + dc, Activate<type>(conv, params, dc));
            }
            if (dc < dstC)
            {
                __m256 conv = bias ? _mm256_loadu_ps(bias + dstC - F) : _mm256_setzero_ps();
                KernelHwcDefaultEdge1<stride, height, width>(src, srcW, srcC, dstC, weight + dstC - F, conv);
                _mm256_storeu_ps(dst + dstC - F, Activate<type>(conv, params, dstC - F));
            }
        }

        template<size_t kernel>
        SIMD_INLINE void KernelHwcDefaultMain2x2(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, __m256 sums[2][2])
        {
            __m256 w0, w1, s0;
            for (size_t ky = 0; ky < kernel; ++ky)
            {
                for (size_t i = 0, n = kernel * srcC; i < n; ++i)
                {
                    w0 = _mm256_loadu_ps(weight + 0 * F);
                    w1 = _mm256_loadu_ps(weight + 1 * F);
                    s0 = _mm256_set1_ps(src[i + 0 * srcC*strideX]);
                    sums[0][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[0][0]);
                    sums[0][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[0][1]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[1][0]);
                    sums[1][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[1][1]);
                    weight += dstC;
                }
                src += srcW * srcC;
            }
        }

        template<size_t kernel>
        SIMD_INLINE void KernelHwcDefaultMain2x1(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, __m256 sums[2][1])
        {
            __m256 w0, s0;
            for (size_t ky = 0; ky < kernel; ++ky)
            {
                for (size_t i = 0, n = kernel * srcC; i < n; ++i)
                {
                    w0 = _mm256_loadu_ps(weight + 0 * F);
                    s0 = _mm256_set1_ps(src[i + 0 * srcC*strideX]);
                    sums[0][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[0][0]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[1][0]);
                    weight += dstC;
                }
                src += srcW * srcC;
            }
        }

        template<size_t kernel, ::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultMain2(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m256 sums[2][2];
                if (bias)
                {
                    sums[0][0] = _mm256_loadu_ps(bias + dc + 0 * F);
                    sums[0][1] = _mm256_loadu_ps(bias + dc + 1 * F);
                    sums[1][0] = _mm256_loadu_ps(bias + dc + 0 * F);
                    sums[1][1] = _mm256_loadu_ps(bias + dc + 1 * F);
                }
                else
                {
                    sums[0][0] = _mm256_setzero_ps();
                    sums[0][1] = _mm256_setzero_ps();
                    sums[1][0] = _mm256_setzero_ps();
                    sums[1][1] = _mm256_setzero_ps();
                }
                KernelHwcDefaultMain2x2<kernel>(src, srcW, srcC, dstC, strideX, weight + dc, sums);
                _mm256_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m256 sums[2][1];
                if (bias)
                {
                    __m256 _bias = _mm256_loadu_ps(bias + dc);
                    sums[0][0] = _bias;
                    sums[1][0] = _bias;
                }
                else
                {
                    sums[0][0] = _mm256_setzero_ps();
                    sums[1][0] = _mm256_setzero_ps();
                }
                KernelHwcDefaultMain2x1<kernel>(src, srcW, srcC, dstC, strideX, weight + dc, sums);
                _mm256_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm256_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
            if (dc < dstC)
            {
                __m256 sums[2][1];
                if (bias)
                {
                    __m256 _bias = _mm256_loadu_ps(bias + dstC - F);
                    sums[0][0] = _bias;
                    sums[1][0] = _bias;
                }
                else
                {
                    sums[0][0] = _mm256_setzero_ps();
                    sums[1][0] = _mm256_setzero_ps();
                }
                KernelHwcDefaultMain2x1<kernel>(src, srcW, srcC, dstC, strideX, weight + dstC - F, sums);
                _mm256_storeu_ps(dst + dstC - F + 0 * dstC, Activate<type>(sums[0][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 1 * dstC, Activate<type>(sums[1][0], params, dstC - F));
            }
        }

        template<size_t kernel>
        SIMD_INLINE void KernelHwcDefaultMain6x2(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, __m256 sums[6][2])
        {
            const float * src0 = src + 0 * srcC*strideX;
            const float * src1 = src + 1 * srcC*strideX;
            const float * src2 = src + 2 * srcC*strideX;
            const float * src3 = src + 3 * srcC*strideX;
            const float * src4 = src + 4 * srcC*strideX;
            const float * src5 = src + 5 * srcC*strideX;
            __m256 w0, w1, s0;
            for (size_t ky = 0; ky < kernel; ++ky)
            {
                size_t offset = ky * srcW * srcC;
                for (size_t end = offset + kernel * srcC; offset < end; ++offset)
                {
                    w0 = _mm256_loadu_ps(weight + 0 * F);
                    w1 = _mm256_loadu_ps(weight + 1 * F);
                    s0 = _mm256_set1_ps(src0[offset]);
                    sums[0][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[0][0]);
                    sums[0][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[0][1]);
                    s0 = _mm256_set1_ps(src1[offset]);
                    sums[1][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[1][0]);
                    sums[1][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[1][1]);
                    s0 = _mm256_set1_ps(src2[offset]);
                    sums[2][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[2][0]);
                    sums[2][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[2][1]);
                    s0 = _mm256_set1_ps(src3[offset]);
                    sums[3][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[3][0]);
                    sums[3][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[3][1]);
                    s0 = _mm256_set1_ps(src4[offset]);
                    sums[4][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[4][0]);
                    sums[4][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[4][1]);
                    s0 = _mm256_set1_ps(src5[offset]);
                    sums[5][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[5][0]);
                    sums[5][1] = _mm256_add_ps(_mm256_mul_ps(s0, w1), sums[5][1]);
                    weight += dstC;
                }
            }
        }

        template<size_t kernel>
        SIMD_INLINE void KernelHwcDefaultMain6x1(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, __m256 sums[6][1])
        {
            __m256 w0, s0;
            for (size_t ky = 0; ky < kernel; ++ky)
            {
                for (size_t i = 0, n = kernel * srcC; i < n; ++i)
                {
                    w0 = _mm256_loadu_ps(weight + 0 * F);
                    s0 = _mm256_set1_ps(src[i + 0 * srcC*strideX]);
                    sums[0][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[0][0]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[1][0]);
                    s0 = _mm256_set1_ps(src[i + 2 * srcC*strideX]);
                    sums[2][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[2][0]);
                    s0 = _mm256_set1_ps(src[i + 3 * srcC*strideX]);
                    sums[3][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[3][0]);
                    s0 = _mm256_set1_ps(src[i + 4 * srcC*strideX]);
                    sums[4][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[4][0]);
                    s0 = _mm256_set1_ps(src[i + 5 * srcC*strideX]);
                    sums[5][0] = _mm256_add_ps(_mm256_mul_ps(s0, w0), sums[5][0]);
                    weight += dstC;
                }
                src += srcW * srcC;
            }
        }

        template<size_t kernel, ::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultMain6(const float * src, size_t srcW, size_t srcC, size_t dstC, size_t strideX, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m256 sums[6][2];
                __m256 bias0 = bias ? _mm256_loadu_ps(bias + dc + 0 * F) : _mm256_setzero_ps();
                __m256 bias1 = bias ? _mm256_loadu_ps(bias + dc + 1 * F) : _mm256_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                sums[2][0] = bias0;
                sums[2][1] = bias1;
                sums[3][0] = bias0;
                sums[3][1] = bias1;
                sums[4][0] = bias0;
                sums[4][1] = bias1;
                sums[5][0] = bias0;
                sums[5][1] = bias1;
                KernelHwcDefaultMain6x2<kernel>(src, srcW, srcC, dstC, strideX, weight + dc, sums);
                _mm256_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 1 * F));
                _mm256_storeu_ps(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 0 * F));
                _mm256_storeu_ps(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m256 sums[6][1];
                __m256 bias0 = bias ? _mm256_loadu_ps(bias + dc) : _mm256_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultMain6x1<kernel>(src, srcW, srcC, dstC, strideX, weight + dc, sums);
                _mm256_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm256_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                _mm256_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                _mm256_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                _mm256_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                _mm256_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
            }
            if (dc < dstC)
            {
                __m256 sums[6][1];
                __m256 bias0 = bias ? _mm256_loadu_ps(bias + dstC - F) : _mm256_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultMain6x1<kernel>(src, srcW, srcC, dstC, strideX, weight + dstC - F, sums);
                _mm256_storeu_ps(dst + dstC - F + 0 * dstC, Activate<type>(sums[0][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 1 * dstC, Activate<type>(sums[1][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 2 * dstC, Activate<type>(sums[2][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 3 * dstC, Activate<type>(sums[3][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 4 * dstC, Activate<type>(sums[4][0], params, dstC - F));
                _mm256_storeu_ps(dst + dstC - F + 5 * dstC, Activate<type>(sums[5][0], params, dstC - F));
            }
        }

        template<size_t kernel, ::SimdConvolutionActivationType type> void ConvolutionDirectHwcConvolutionBiasActivationDefault(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstH = p.dstH - p.padH, dstW = p.dstW - p.padW;
            size_t wS = p.srcC*p.dstC, sS = p.strideX*p.srcC;
            size_t dstW2 = AlignLoAny(dstW - p.padX, 2) + p.padX;
            size_t dstW6 = AlignLoAny(dstW - p.padX, 6) + p.padX;
            if (p.padY)
            {
                if (p.padX)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel - 1, type>(src, p.srcW, p.srcC, p.dstC, weight + (kernel + 1)*wS, bias, params, dst);
                for (size_t dx = p.padX; dx < dstW; ++dx)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel, type>(src + (dx - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight + kernel * wS, bias, params, dst + dx * p.dstC);
                if (p.padW)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel - 1, type>(src + (dstW - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight + kernel * wS, bias, params, dst + dstW * p.dstC);
                dst += p.dstW*p.dstC;
            }
            for (size_t dy = p.padY; dy < dstH; ++dy)
            {
                if (p.padX)
                    KernelHwcDefaultEdge<kernel, kernel, kernel - 1, type>(src, p.srcW, p.srcC, p.dstC, weight + wS, bias, params, dst);
                size_t dx = p.padX;
                for (; dx < dstW6; dx += 6)
                    KernelHwcDefaultMain6<kernel, type>(src + (dx - p.padX) * sS, p.srcW, p.srcC, p.dstC, p.strideX, weight, bias, params, dst + dx * p.dstC);
                for (; dx < dstW2; dx += 2)
                    KernelHwcDefaultMain2<kernel, type>(src + (dx - p.padX) * sS, p.srcW, p.srcC, p.dstC, p.strideX, weight, bias, params, dst + dx * p.dstC);
                for (; dx < dstW; ++dx)
                    KernelHwcDefaultEdge<kernel, kernel, kernel, type>(src + (dx - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight, bias, params, dst + dx * p.dstC);
                if (p.padW)
                    KernelHwcDefaultEdge<kernel, kernel, kernel - 1, type>(src + (dstW - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight, bias, params, dst + dstW * p.dstC);
                src += p.strideY*p.srcW*p.srcC;
                dst += p.dstW*p.dstC;
            }
            if (p.padH)
            {
                if (p.padX)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel - 1, type>(src, p.srcW, p.srcC, p.dstC, weight + wS, bias, params, dst);
                for (size_t dx = p.padX; dx < dstW; ++dx)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel, type>(src + (dx - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight, bias, params, dst + dx * p.dstC);
                if (p.padW)
                    KernelHwcDefaultEdge<kernel, kernel - 1, kernel - 1, type>(src + (dstW - p.padX) * sS, p.srcW, p.srcC, p.dstC, weight, bias, params, dst + dstW * p.dstC);
            }
        }

        static void ConvolutionDirectHwcConvolutionBiasActivationDefault(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    memset(dst, 0, p.dstC * sizeof(float));
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx - p.padX;
                                if (sx < p.srcW)
                                {
                                    const float * pw = weight + (ky*p.kernelX + kx)*p.srcC*p.dstC;
                                    const float * ps = src + (sy*p.srcW + sx)*p.srcC;
                                    for (size_t sc = 0; sc < p.srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < p.dstC; ++dc)
                                            dst[dc] += ps[0] * pw[dc];
                                        ps += 1;
                                        pw += p.dstC;
                                    }
                                }
                            }
                        }
                    }
                    ConvolutionBiasAndActivation(bias, p.dstC, 1, p.activation, params, ::SimdTrue, dst);
                    dst += p.dstC;
                }
            }
        }

        template <::SimdConvolutionActivationType type> ConvolutionDirectHwc::ConvolutionBiasActivationPtr GetConvolutionBiasActivation(const ConvParam & p)
        {
            if (p.IsKernel(3))
                return ConvolutionDirectHwcConvolutionBiasActivationDefault<3, type>;
            return ConvolutionDirectHwcConvolutionBiasActivationDefault;
        }

        ConvolutionDirectHwc::ConvolutionBiasActivationPtr ConvolutionDirectHwc::SetConvolutionBiasActivation()
        {
            const ConvParam & p = _param;
            ConvolutionDirectHwc::ConvolutionBiasActivationPtr func = NULL;
            if (p.dstC >= F && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: func = GetConvolutionBiasActivation<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationPrelu>(p); break;
                }
            }
            return func ? func : Sse::ConvolutionDirectHwc::SetConvolutionBiasActivation();
        };

        //---------------------------------------------------------------------

        ConvolutionDepthwiseDotProduct::ConvolutionDepthwiseDotProduct(const ConvParam & p)
            : Sse::ConvolutionDepthwiseDotProduct(p)
        {
        }

        SIMD_INLINE void DotProduct(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = _mm256_loadu_ps(a + offset);
            __m256 _b = _mm256_loadu_ps(b + offset);
            sum = _mm256_add_ps(_mm256_mul_ps(_a, _b), sum);
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            float sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        DotProduct(a, b, i + F * 0, sums[0]);
                        DotProduct(a, b, i + F * 1, sums[1]);
                        DotProduct(a, b, i + F * 2, sums[2]);
                        DotProduct(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    DotProduct(a, b, i, sums[0]);
                sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                sum += a[i] * b[i];
            return sum;
        }

        void ConvolutionDepthwiseDotProduct::Forward(const float * src, float * buf, float * dst)
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
                ConvolutionBiasAndActivation(NULL, _count, 1, _param.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation)
        {
            ConvParam param(srcC, srcH, srcW, srcT, dstC, dstT, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, activation);
            if (!param.Valid())
                return NULL;
            else if (ConvolutionDepthwiseDotProduct::Preferable(param))
                return new ConvolutionDepthwiseDotProduct(param);
            else if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionGemmNT::Preferable(param))
                return new ConvolutionGemmNT(param);
            else if (ConvolutionDirectChw::Preferable(param))
                return new Avx::ConvolutionDirectChw(param);
            else if (ConvolutionDirectHwc::Preferable(param))
                return new ConvolutionDirectHwc(param);
            else
                return new ConvolutionGemmNN(param);
        }
    }
#endif//SIMD_AVX_ENABLE
}
