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
#include "Simd/SimdSse1.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, float * dst)
        {
            size_t aligned = AlignLo(size, F);
            if (type == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    SynetAddBias(bias, count, size, dst);
            }
            else if (type == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m128 _0 = _mm_set1_ps(0.0f);
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        __m128 _shift = _mm_set1_ps(shift);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m128 _dst = _mm_loadu_ps(dst + j);
                            _mm_storeu_ps(dst + j, _mm_max_ps(_0, _mm_add_ps(_dst, _shift)));
                        }
                        for (; j < size; ++j)
                            dst[j] = Simd::Max(0.0f, dst[j] + shift);
                        dst += size;
                    }
                }
                else
                {
                    float slope = 0;
                    NeuralRelu(dst, size*count, &slope, dst);
                }
            }
            else if (type == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    __m128 _slope = _mm_set1_ps(slope);
                    __m128 _0 = _mm_set1_ps(0.0f);
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        __m128 _shift = _mm_set1_ps(shift);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _shift);
                            _mm_storeu_ps(dst + j, _mm_add_ps(_mm_max_ps(_0, value), _mm_mul_ps(_slope, _mm_min_ps(_0, value))));
                        }
                        for (; j < size; ++j)
                        {
                            float value = dst[j] + shift;
                            dst[i] = Simd::Max(0.0f, value) + slope * Simd::Min(value, 0.0f);
                        }
                        dst += size;
                    }
                }
                else
                    NeuralRelu(dst, size*count, &slope, dst);
            }
            else if (type == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    __m128 _lower = _mm_set1_ps(lower);
                    __m128 _upper = _mm_set1_ps(upper);
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        __m128 _shift = _mm_set1_ps(shift);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _shift);
                            _mm_storeu_ps(dst + j, _mm_min_ps(_mm_max_ps(_lower, value), _upper));
                        }
                        for (; j < size; ++j)
                            dst[j] = Simd::RestrictRange(dst[j] + shift, lower, upper);
                        dst += size;
                    }
                }
                else
                    SynetRestrictRange(dst, size*count, &lower, &upper, dst);
            }
            else if (type == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    __m128 _0 = _mm_set1_ps(0.0f);
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        float slope = params[i];
                        __m128 _shift = _mm_set1_ps(shift);
                        __m128 _slope = _mm_set1_ps(slope);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _shift);
                            _mm_storeu_ps(dst + j, _mm_add_ps(_mm_max_ps(_0, value), _mm_mul_ps(_slope, _mm_min_ps(_0, value))));
                        }
                        for (; j < size; ++j)
                        {
                            float value = dst[j] + shift;
                            dst[j] = Simd::Max(0.0f, value) + slope*Simd::Min(value, 0.0f);
                        }
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                        NeuralRelu(dst + i*size, size, params + i, dst + i*size);
                }
            }
        }

        //---------------------------------------------------------------------

        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Base::ConvolutionImgToCol(p)
        {
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Sse::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            Sse::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Base::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::SetWeight(const float * weight, const float * bias, SimdBool * internal)
        {
            const ConvParam & p = _param;
            _weight.Resize(_strideW*_count);
            Sse::Winograd2x3pSetFilter(weight, p.srcC*p.dstC, _weight.data);
            _bias = bias;
            if (internal)
                *internal = SimdTrue;
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Sse::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Sse::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Sse::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            Sse::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Base::ConvolutionDirect(p)
        {
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m128 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static __m128 Convolution(const float * src, size_t step, const __m128  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE __m128 Convolution(const float * src, size_t step, const __m128  * weight)
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

            static SIMD_INLINE __m128 Convolution(const float * src, size_t step, const __m128  * weight)
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

            static SIMD_INLINE __m128 Convolution(const float * src, size_t step, const __m128  * weight)
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

            static SIMD_INLINE __m128 Convolution(const float * src, size_t step, const __m128  * weight)
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

            static SIMD_INLINE __m128 Convolution(const float * src, size_t step, const __m128  * weight)
            {
                return _mm_add_ps(RowConv(src, weight),
                    _mm_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
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

        template<int kernel, int stride, ::SimdConvolutionActivationType type> void ConvolutionAndBias(const float * src, size_t srcC, size_t srcH, size_t srcW,
            const float * weight, const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m128 _weight[kernel*kernel];
            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm_set1_ps(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            __m128 tail = RightNotZero(dstW - dstWF);
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
                            __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm_storeu_ps(pd + x, Activate<type>(_mm_add_ps(_bias, conv), _params));
                        }
                        if (dstWF < dstW)
                        {
                            size_t x = dstW - F;
                            __m128 _dst = _mm_loadu_ps(pd + x);
                            __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
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
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, _mm_add_ps(_bias, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
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
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, _mm_add_ps(_dst, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
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
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm_storeu_ps(pd + x, Activate<type>(_mm_add_ps(_dst, conv), _params));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m128 _dst = _mm_loadu_ps(pd + x);
                                __m128 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
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

        template<int kernel, int stride> void ConvolutionAndBias(const float * src, size_t srcC, size_t srcH, size_t srcW,
            const float * weight, const float * bias, ::SimdConvolutionActivationType type, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity:
                ConvolutionAndBias<kernel, stride, ::SimdConvolutionActivationIdentity>(src, srcC, srcH, srcW, weight, bias, params, dst, dstC, dstH, dstW);
                break;
            case ::SimdConvolutionActivationRelu:
                ConvolutionAndBias<kernel, stride, ::SimdConvolutionActivationRelu>(src, srcC, srcH, srcW, weight, bias, params, dst, dstC, dstH, dstW);
                break;
            case ::SimdConvolutionActivationLeakyRelu:
                ConvolutionAndBias<kernel, stride, ::SimdConvolutionActivationLeakyRelu>(src, srcC, srcH, srcW, weight, bias, params, dst, dstC, dstH, dstW);
                break;
            case ::SimdConvolutionActivationRestrictRange:
                ConvolutionAndBias<kernel, stride, ::SimdConvolutionActivationRestrictRange>(src, srcC, srcH, srcW, weight, bias, params, dst, dstC, dstH, dstW);
                break;
            case ::SimdConvolutionActivationPrelu:
                ConvolutionAndBias<kernel, stride, ::SimdConvolutionActivationPrelu>(src, srcC, srcH, srcW, weight, bias, params, dst, dstC, dstH, dstW);
                break;
            default:
                assert(0);
            }
        }

        bool ConvolutionDirect::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && ((p.IsStride(1) && p.IsKernel(1)) || p.IsKernel(2) || p.IsKernel(3));
        }

        void ConvolutionDirect::ConvolutionAndBias(const float * src, const float * weight, const float * bias, const float * params, float * dst) const
        {
            const ConvParam & p = _param;
            if (p.dstW >= F)
            {
                switch (p.kernelX)
                {
                case 1:
                    Sse::ConvolutionAndBias<1, 1>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, params, dst, _dstC, p.dstH, p.dstW);
                    return;
                case 2:
                    if (p.IsStride(2))
                        Sse::ConvolutionAndBias<2, 2>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, params, dst, _dstC, p.dstH, p.dstW);
                    else
                        Sse::ConvolutionAndBias<2, 1>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, params, dst, _dstC, p.dstH, p.dstW);
                    return;
                case 3:
                    if (p.IsStride(2))
                        Sse::ConvolutionAndBias<3, 2>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, params, dst, _dstC, p.dstH, p.dstW);
                    else
                        Sse::ConvolutionAndBias<3, 1>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, params, dst, _dstC, p.dstH, p.dstW);
                    return;
                default:
                    break;
                };
            }
            Base::ConvolutionDirect::ConvolutionAndBias(src, weight, bias, params, dst);
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionDirect::Preferable(param))
                return new ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
#endif//SIMD_SSE_ENABLE
}
