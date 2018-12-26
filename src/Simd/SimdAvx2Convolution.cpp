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

        ConvolutionGemmNT::ConvolutionGemmNT(const ConvParam & p)
            : Avx::ConvolutionGemmNT(p)
        {
        }

        void ConvolutionGemmNT::GemmAndBias(const float * src, float * dst)
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

        ConvolutionDirectChw::ConvolutionDirectChw(const ConvParam & p)
            : Avx::ConvolutionDirectChw(p)
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
            return Sse::ConvolutionDirectChw::SetConvolutionBiasActivation();
        }

        //---------------------------------------------------------------------

        ConvolutionDirectHwc::ConvolutionDirectHwc(const ConvParam & p)
            : Avx::ConvolutionDirectHwc(p)
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
                        sum = _mm256_fmadd_ps(_mm256_set1_ps(ps[sc]), _mm256_loadu_ps(pw), sum);
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
                    sum0 = _mm256_fmadd_ps(_mm256_set1_ps(src[i + 0]), _mm256_loadu_ps(weight + 0 * dstC), sum0);
                    sum1 = _mm256_fmadd_ps(_mm256_set1_ps(src[i + 1]), _mm256_loadu_ps(weight + 1 * dstC), sum1);
                    sum2 = _mm256_fmadd_ps(_mm256_set1_ps(src[i + 2]), _mm256_loadu_ps(weight + 2 * dstC), sum2);
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
                    sums[0][0] = _mm256_fmadd_ps(s0, w0, sums[0][0]);
                    sums[0][1] = _mm256_fmadd_ps(s0, w1, sums[0][1]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_fmadd_ps(s0, w0, sums[1][0]);
                    sums[1][1] = _mm256_fmadd_ps(s0, w1, sums[1][1]);
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
                    sums[0][0] = _mm256_fmadd_ps(s0, w0, sums[0][0]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_fmadd_ps(s0, w0, sums[1][0]);
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
                _mm256_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 0 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 1 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * dstC + 1 * F));
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
                _mm256_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc + 0 * dstC));
                _mm256_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc + 1 * dstC));
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
                _mm256_storeu_ps(dst + dstC - F + 0 * dstC, Activate<type>(sums[0][0], params, dstC - F + 0 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 1 * dstC, Activate<type>(sums[1][0], params, dstC - F + 1 * dstC));
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
                    sums[0][0] = _mm256_fmadd_ps(s0, w0, sums[0][0]);
                    sums[0][1] = _mm256_fmadd_ps(s0, w1, sums[0][1]);
                    s0 = _mm256_set1_ps(src1[offset]);
                    sums[1][0] = _mm256_fmadd_ps(s0, w0, sums[1][0]);
                    sums[1][1] = _mm256_fmadd_ps(s0, w1, sums[1][1]);
                    s0 = _mm256_set1_ps(src2[offset]);
                    sums[2][0] = _mm256_fmadd_ps(s0, w0, sums[2][0]);
                    sums[2][1] = _mm256_fmadd_ps(s0, w1, sums[2][1]);
                    s0 = _mm256_set1_ps(src3[offset]);
                    sums[3][0] = _mm256_fmadd_ps(s0, w0, sums[3][0]);
                    sums[3][1] = _mm256_fmadd_ps(s0, w1, sums[3][1]);
                    s0 = _mm256_set1_ps(src4[offset]);
                    sums[4][0] = _mm256_fmadd_ps(s0, w0, sums[4][0]);
                    sums[4][1] = _mm256_fmadd_ps(s0, w1, sums[4][1]);
                    s0 = _mm256_set1_ps(src5[offset]);
                    sums[5][0] = _mm256_fmadd_ps(s0, w0, sums[5][0]);
                    sums[5][1] = _mm256_fmadd_ps(s0, w1, sums[5][1]);
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
                    sums[0][0] = _mm256_fmadd_ps(s0, w0, sums[0][0]);
                    s0 = _mm256_set1_ps(src[i + 1 * srcC*strideX]);
                    sums[1][0] = _mm256_fmadd_ps(s0, w0, sums[1][0]);
                    s0 = _mm256_set1_ps(src[i + 2 * srcC*strideX]);
                    sums[2][0] = _mm256_fmadd_ps(s0, w0, sums[2][0]);
                    s0 = _mm256_set1_ps(src[i + 3 * srcC*strideX]);
                    sums[3][0] = _mm256_fmadd_ps(s0, w0, sums[3][0]);
                    s0 = _mm256_set1_ps(src[i + 4 * srcC*strideX]);
                    sums[4][0] = _mm256_fmadd_ps(s0, w0, sums[4][0]);
                    s0 = _mm256_set1_ps(src[i + 5 * srcC*strideX]);
                    sums[5][0] = _mm256_fmadd_ps(s0, w0, sums[5][0]);
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
                _mm256_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 0 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 1 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 2 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 2 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 3 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 3 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 4 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 4 * dstC + 1 * F));
                _mm256_storeu_ps(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 5 * dstC + 0 * F));
                _mm256_storeu_ps(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 5 * dstC + 1 * F));
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
                _mm256_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc + 0 * dstC));
                _mm256_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc + 1 * dstC));
                _mm256_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc + 2 * dstC));
                _mm256_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc + 3 * dstC));
                _mm256_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc + 4 * dstC));
                _mm256_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc + 5 * dstC));
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
                _mm256_storeu_ps(dst + dstC - F + 0 * dstC, Activate<type>(sums[0][0], params, dstC - F + 0 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 1 * dstC, Activate<type>(sums[1][0], params, dstC - F + 1 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 2 * dstC, Activate<type>(sums[2][0], params, dstC - F + 2 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 3 * dstC, Activate<type>(sums[3][0], params, dstC - F + 3 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 4 * dstC, Activate<type>(sums[4][0], params, dstC - F + 4 * dstC));
                _mm256_storeu_ps(dst + dstC - F + 5 * dstC, Activate<type>(sums[5][0], params, dstC - F + 5 * dstC));
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
            return func ? func : Avx::ConvolutionDirectHwc::SetConvolutionBiasActivation();
        };

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
            else if (ConvolutionGemmNT::Preferable(param))
                return new ConvolutionGemmNT(param);
            else if (ConvolutionDirectChw::Preferable(param))
                return new Avx2::ConvolutionDirectChw(param);
            else if (ConvolutionDirectHwc::Preferable(param))
                return new ConvolutionDirectHwc(param);
            else
                return new ConvolutionGemmNN(param);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
