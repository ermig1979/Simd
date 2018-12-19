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
#include "Simd/SimdSet.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        ConvolutionGemmNN::ConvolutionGemmNN(const ConvParam & p)
            : Avx::ConvolutionGemmNN(p)
        {
            _index.Resize(F);
            for (size_t i = 0; i < F; ++i)
                _index[i] = int(i * p.strideX);
            _nose.Resize(p.kernelX);
            _tail.Resize(p.kernelX);
            _start.Resize(p.kernelX);
            for (size_t kx = 0; kx < p.kernelX; ++kx)
            {
                _nose[kx] = 0;
                _tail[kx] = int(p.dstW);
                ptrdiff_t sx = kx * p.dilationX - p.padX;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    if (sx < 0)
                        _nose[kx]++;
                    if (sx >= ptrdiff_t(p.srcW))
                        _tail[kx]--;
                    sx += p.strideX;
                }
                _start[kx] = int(kx * p.dilationX - p.padX + _nose[kx] * p.strideX);
            }
        }

        void ConvolutionGemmNN::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
            {
                if (p.srcT)
                    Avx2::Gemm32fNN(_M, _N, _K, &_1, src + _grS * g, _ldS, _weight + _grW * g, _ldW, &_0, dst + _grD * g, _ldD);
                else
                    Avx2::Gemm32fNN(_M, _N, _K, &_1, _weight + _grW * g, _ldW, src + _grS * g, _ldS, &_0, dst + _grD * g, _ldD);
            }
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.dstT, dst);
        }

        void ConvolutionGemmNN::ImgToCol(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy*p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX*p.dilationY*p.strideX*p.strideY != 1)
            {
                __m256i index = _mm256_loadu_si256((__m256i*)_index.data);
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t noseDx = _nose[kx];
                            size_t tailDx = _tail[kx];
                            size_t bodyDx = AlignLo(tailDx - noseDx, F) + noseDx;
                            size_t sx0 = _start[kx];
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t dx = 0, sx = sx0 + sy * p.srcW;
                                    for (; dx < noseDx; ++dx)
                                        *(dst++) = 0;
                                    for (; dx < bodyDx; dx += F, sx += p.strideX*F, dst += F)
                                        _mm256_storeu_ps(dst, _mm256_i32gather_ps(src + sx, index, 4));
                                    for (; dx < tailDx; ++dx, sx += p.strideX)
                                        *(dst++) = src[sx];
                                    for (; dx < p.dstW; ++dx)
                                        *(dst++) = 0;
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                    dst += p.dstW;
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                Base::ConvolutionGemmNN::ImgToCol(src, dst);
            }
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
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _param.activation, _params, ::SimdFalse, dst);
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
            Avx::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _param.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Avx::ConvolutionDirect(p)
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

        template<> struct Kernel<3, 3>
        {
            static SIMD_INLINE __m256 Gather(const float * src)
            {
                return _mm256_shuffle_ps(Avx::Load<false>(src + 0, src + 12), Avx::Load<false>(src + 6, src + 18), 0xCC);
            }

            static SIMD_INLINE __m256 RowConv(const float * src, const __m256  * weight)
            {
                return _mm256_fmadd_ps(Gather(src + 0), weight[0],
                    _mm256_fmadd_ps(Gather(src + 1), weight[1],
                        _mm256_mul_ps(Gather(src + 2), weight[2])));
            }

            static SIMD_INLINE __m256 Convolution(const float * src, size_t step, const __m256  * weight)
            {
                return _mm256_add_ps(RowConv(src, weight), _mm256_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6)));
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

        template <int kernel, int stride> ConvolutionDirect::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
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

        ConvolutionDirect::ConvolutionBiasActivationPtr ConvolutionDirect::SetConvolutionBiasActivation()
        {
            const ConvParam & p = _param;
            if (p.dstW < F)
                return Sse::ConvolutionDirect::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Avx2::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Avx2::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Avx2::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Avx2::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Avx2::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Avx2::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            }
            return Sse::ConvolutionDirect::SetConvolutionBiasActivation();
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation)
        {
            ConvParam param(srcC, srcH, srcW, srcT, dstC, dstT, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, activation);
            if (!param.Valid())
                return NULL;
            else if (Avx::ConvolutionDepthwiseDotProduct::Preferable(param))
                return new Avx::ConvolutionDepthwiseDotProduct(param);
            else if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionImgToRow::Preferable(param))
                return new ConvolutionImgToRow(param);
            else if (ConvolutionDirect::Preferable(param))
                return new Avx2::ConvolutionDirect(param);
            else
                return new ConvolutionGemmNN(param);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
