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
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Avx::ConvolutionImgToCol(p)
        {
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx2::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionImgToRow::ConvolutionImgToRow(const ConvParam & p)
            : Avx::ConvolutionImgToRow(p)
        {
        }

        void ConvolutionImgToRow::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx2::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Avx::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Avx::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Avx2::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Avx::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Avx::ConvolutionDirect(p)
        {
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

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                return _mm256_fmadd_ps(_mm256_loadu_ps(src), weight[0],
                    _mm256_mul_ps(_mm256_loadu_ps(src + 1), weight[1]));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                __m256 s0 = _mm256_loadu_ps(src + 0);
                __m256 s1 = _mm256_loadu_ps(src + F);
                return _mm256_fmadd_ps(_mm256_shuffle_ps(s0, s1, 0x88), weight[0], 
                    _mm256_mul_ps(_mm256_shuffle_ps(s0, s1, 0xDD), weight[1]));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return Permute4x64<0xD8>(_mm256_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2)));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                return _mm256_fmadd_ps(_mm256_loadu_ps(src), weight[0],
                    _mm256_fmadd_ps(_mm256_loadu_ps(src + 1), weight[1],
                        _mm256_mul_ps(_mm256_loadu_ps(src + 2), weight[2])));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_add_ps(RowConv(src, weight),
                    _mm256_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                __m256 s00 = _mm256_loadu_ps(src);
                __m256 s10 = _mm256_loadu_ps(src + F);
                __m256 s02 = _mm256_loadu_ps(src + 2);
                __m256 s12 = _mm256_loadu_ps(src + 2 + F);
                return _mm256_fmadd_ps(_mm256_shuffle_ps(s00, s10, 0x88), weight[0],
                    _mm256_fmadd_ps(_mm256_shuffle_ps(s00, s10, 0xDD), weight[1],
                        _mm256_mul_ps(_mm256_shuffle_ps(s02, s12, 0x88), weight[2])));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return Permute4x64<0xD8>(_mm256_add_ps(RowConv(src, weight), 
                    _mm256_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6))));
            }
        };

        template<::SimdConvolutionActivationType type> struct Activation
        {
            static SIMD_INLINE __m256 Apply(const __m256 & value, const __m256 * params);
        };

        template<> struct Activation<::SimdConvolutionActivationIdentity>
        {
            static SIMD_INLINE __m256 Apply(const __m256 & value, const __m256 * params)
            {
                return value;
            }
        };

        template<> struct Activation<::SimdConvolutionActivationRelu>
        {
            static SIMD_INLINE __m256 Apply(const __m256 & value, const __m256 * params)
            {
                return _mm256_max_ps(_mm256_setzero_ps(), value);
            }
        };

        template<> struct Activation<::SimdConvolutionActivationLeakyRelu>
        {
            static SIMD_INLINE __m256 Apply(const __m256 & value, const __m256 * params)
            {
                return _mm256_max_ps(_mm256_mul_ps(params[0], value), value);
            }
        };

        template<> struct Activation<::SimdConvolutionActivationRestrictRange>
        {
            static SIMD_INLINE __m256 Apply(const __m256 & value, const __m256 * params)
            {
                return _mm256_min_ps(_mm256_max_ps(params[0], value), params[1]);
            }
        };

        template<int kernel, int stride, ::SimdConvolutionActivationType type> void ConvolutionAndBias(const float * src, size_t srcC, size_t srcH, size_t srcW,
            const float * weight, const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m256 _weight[kernel*kernel];
            __m256 _params[2] = { _mm256_set1_ps(params[0]), _mm256_set1_ps(params[1]) };
            size_t dstWF = Simd::AlignLo(dstW, F);
            __m256 tail = RightNotZero(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
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
                            _mm256_storeu_ps(pd + x, Activation<type>::Apply(_mm256_add_ps(_bias, conv), _params));
                        }
                        if (dstWF < dstW)
                        {
                            size_t x = dstW - F;
                            __m256 _dst = _mm256_loadu_ps(pd + x);
                            __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm256_storeu_ps(pd + x, _mm256_blendv_ps(_dst, Activation<type>::Apply(_mm256_add_ps(_bias, conv), _params), tail));
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
                                _mm256_storeu_ps(pd + x, Activation<type>::Apply(_mm256_add_ps(_dst, conv), _params));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                __m256 _dst = _mm256_loadu_ps(pd + x);
                                __m256 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm256_storeu_ps(pd + x, _mm256_blendv_ps(_dst, Activation<type>::Apply(_mm256_add_ps(_dst, conv), _params), tail));
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
            default:
                assert(0);
            }
        }

        void ConvolutionDirect::ConvolutionAndBias(const float * src, const float * weight, const float * bias, float * dst) const
        {
            const ConvParam & p = _param;
            if (p.dstW >= F)
            {
                switch (p.kernelX)
                {
                case 1:
                    break;
                case 2:
                    if (p.IsStride(2))
                        Avx2::ConvolutionAndBias<2, 2>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, _activationParams, dst, _dstC, p.dstH, p.dstW);
                    else
                        Avx2::ConvolutionAndBias<2, 1>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, _activationParams, dst, _dstC, p.dstH, p.dstW);
                    return;
                case 3:
                    if (p.IsStride(2))
                        Avx2::ConvolutionAndBias<3, 2>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, _activationParams, dst, _dstC, p.dstH, p.dstW);
                    else
                        Avx2::ConvolutionAndBias<3, 1>(src, _srcC, _srcH, _srcW, weight, bias, _activationType, _activationParams, dst, _dstC, p.dstH, p.dstW);
                    return;
                default:
                    break;
                };
            }
            Sse::ConvolutionDirect::ConvolutionAndBias(src, weight, bias, dst);
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionImgToRow::Preferable(param))
                return new ConvolutionImgToRow(param);
            else if (ConvolutionDirect::Preferable(param))
                return new Avx2::ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
