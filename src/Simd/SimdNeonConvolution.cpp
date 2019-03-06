/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    Neon::SynetAddBias(bias, count, size, dst, trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t _dst = Load<false>(dst + i);
                                float32x4_t _bias = Load<false>(bias + i);
                                Store<false>(dst + i, vmaxq_f32(_0, vaddq_f32(_dst, _bias)));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t _dst = Load<false>(dst + j);
                                Store<false>(dst + j, vmaxq_f32(_0, vaddq_f32(_dst, _bias)));
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
                    Neon::NeuralRelu(dst, size*count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    float32x4_t _slope = vdupq_n_f32(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetPreluLayerForward(value, _slope, _0));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetPreluLayerForward(value, _slope, _0));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetPreluLayerForward(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::NeuralRelu(dst, size*count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    float32x4_t _lower = vdupq_n_f32(lower);
                    float32x4_t _upper = vdupq_n_f32(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, vminq_f32(vmaxq_f32(_lower, value), _upper));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, vminq_f32(vmaxq_f32(_lower, value), _upper));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetRestrictRange(dst, size*count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetPreluLayerForward(value, Load<false>(params + i), _0));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            float32x4_t _slope = vdupq_n_f32(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetPreluLayerForward(value, _slope, _0));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetPreluLayerForward(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetPreluLayerForward(dst, params, count, size, dst, trans);
            }
        }

        //---------------------------------------------------------------------

        ConvolutionGemmNN::ConvolutionGemmNN(const ConvParam & p)
            : Base::ConvolutionGemmNN(p)
        {
            _gemm.Init(Neon::Gemm32fNN, "Neon", p.gemm, "Ext");
        }

        void ConvolutionGemmNN::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
            {
                if (p.srcT)
                    _gemm.Run(_M, _N, _K, &_1, src + _grS * g, _ldS, _weight + _grW * g, _ldW, &_0, dst + _grD * g, _ldD);
                else
                    _gemm.Run(_M, _N, _K, &_1, _weight + _grW * g, _ldW, src + _grS * g, _ldS, &_0, dst + _grD * g, _ldD);
            }
            Neon::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.dstT, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionGemmNT::ConvolutionGemmNT(const ConvParam & p)
            : Base::ConvolutionGemmNT(p)
        {
        }

        bool ConvolutionGemmNT::Preferable(const ConvParam & p)
        {
            return p.srcH < 4 && p.srcW < 4 && p.group == 1 && p.srcT == 0 && p.dstT == 0;
        }

        void ConvolutionGemmNT::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Base::ConvolutionWinograd2x3p(p)
        {
            _setFilter = Neon::Winograd2x3SetFilter;
            _gemm.Init(Neon::Gemm32fNN, "Neon", p.gemm, "Ext");
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Neon::Winograd2x3SetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad, p.srcT);
            for (size_t i = 0; i < _count; ++i)
            {
                if (p.srcT)
                    _gemm.Run(_M, _N, _K, &_1, bufS + i * _strideS, _K, _weight.data + i * _strideW, _N, &_0, bufD + i * _strideD, _N);
                else
                    _gemm.Run(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            }
            Neon::Winograd2x3SetOutput(bufD, dst, p.dstC, p.dstH, p.dstW, p.dstT);
            Neon::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.dstT, dst);
        }

        bool ConvolutionWinograd2x3p::Preferable(const ConvParam & p)
        {
            return p.IsKernel(3) && p.IsDilation(1) && p.IsStride(1) && (p.IsPad(0) || p.IsPad(1)) && p.group == 1 && p.srcC >= 10 &&
                (p.srcT ? (p.srcH >= 12 && p.srcW >= 12) : (p.srcH >= 6 && p.srcW >= 6));
        }

        //---------------------------------------------------------------------

        ConvolutionDirectChw::ConvolutionDirectChw(const ConvParam & p)
            : Base::ConvolutionDirectChw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, float32x4_t * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = vdupq_n_f32(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vmulq_f32(Load<false>(src), weight[0]);
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE float32x4_t RowConv(const float * src, const float32x4_t  * weight)
            {
                return vmlaq_f32(vmulq_f32(Load<false>(src + 0), weight[0]), Load<false>(src + 1), weight[1]);
            }

            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vaddq_f32(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE float32x4_t RowConv(const float * src, const float32x4_t  * weight)
            {
                float32x4x2_t s = Load2<false>(src);
                return vmlaq_f32(vmulq_f32(s.val[0], weight[0]), s.val[1], weight[1]);
            }

            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vaddq_f32(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE float32x4_t RowConv(const float * src, const float32x4_t  * weight)
            {
                return vmlaq_f32(vmlaq_f32(vmulq_f32(Load<false>(src), weight[0]), 
                    Load<false>(src + 1), weight[1]), Load<false>(src + 2), weight[2]);
            }

            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vaddq_f32(RowConv(src, weight),
                    vaddq_f32(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE float32x4_t RowConv(const float * src, const float32x4_t  * weight)
            {
                float32x4x2_t s0 = Load2<false>(src + 0);
                float32x4x2_t s2 = Load2<false>(src + 2);
                return vmlaq_f32(vmlaq_f32(vmulq_f32(s0.val[0], weight[0]), 
                    s0.val[1], weight[1]), s2.val[0], weight[2]);
            }

            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vaddq_f32(RowConv(src, weight),
                    vaddq_f32(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 3>
        {
            static SIMD_INLINE float32x4_t RowConv(const float * src, const float32x4_t  * weight)
            {
                float32x4x3_t s = Load3<false>(src);
                return vmlaq_f32(vmlaq_f32(vmulq_f32(s.val[0], weight[0]),
                    s.val[1], weight[1]), s.val[2], weight[2]);
            }

            static SIMD_INLINE float32x4_t Convolution(const float * src, size_t step, const float32x4_t  * weight)
            {
                return vaddq_f32(RowConv(src, weight), 
                    vaddq_f32(RowConv(src + step, weight + 3), 
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<::SimdConvolutionActivationType type> SIMD_INLINE float32x4_t Activate(float32x4_t value, const float32x4_t * params);

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationIdentity>(float32x4_t value, const float32x4_t * params)
        {
            return value;
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRelu>(float32x4_t value, const float32x4_t * params)
        {
            return vmaxq_f32(vdupq_n_f32(0.0f), value);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationLeakyRelu>(float32x4_t value, const float32x4_t * params)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[0], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRestrictRange>(float32x4_t value, const float32x4_t * params)
        {
            return vminq_f32(vmaxq_f32(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationPrelu>(float32x4_t value, const float32x4_t * params)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[0], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type>
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight,
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            float32x4_t _weight[kernel*kernel];
            float32x4_t _params[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = vdupq_n_f32(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            float32x4_t tail = RightNotZero(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = vdupq_n_f32(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel*kernel>(weight, _weight);
                    float32x4_t _bias = bias ? vdupq_n_f32(bias[dc]) : vdupq_n_f32(0.0f);
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        for (size_t x = 0; x < dstWF; x += F)
                        {
                            float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            Store<false>(pd + x, Activate<type>(vaddq_f32(_bias, conv), _params));
                        }
                        if (dstWF < dstW)
                        {
                            size_t x = dstW - F;
                            float32x4_t _dst = Load<false>(pd + x);
                            float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            Store<false>(pd + x, vbslq_f32(vreinterpretq_u32_f32(tail), Activate<type>(vaddq_f32(_bias, conv), _params), _dst));
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
                        float32x4_t _bias = bias ? vdupq_n_f32(bias[dc]) : vdupq_n_f32(0.0f);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            for (size_t x = 0; x < dstWF; x += F)
                            {
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, vaddq_f32(_bias, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                float32x4_t _dst = Load<false>(pd + x);
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, vbslq_f32(vreinterpretq_u32_f32(tail), vaddq_f32(_bias, conv), _dst));
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
                                float32x4_t _dst = Load<false>(pd + x);
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, vaddq_f32(_dst, conv));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                float32x4_t _dst = Load<false>(pd + x);
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, vaddq_f32(_dst, And(conv, tail)));
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
                                float32x4_t _dst = Load<false>(pd + x);
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, Activate<type>(vaddq_f32(_dst, conv), _params));
                            }
                            if (dstWF < dstW)
                            {
                                size_t x = dstW - F;
                                float32x4_t _dst = Load<false>(pd + x);
                                float32x4_t conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                Store<false>(pd + x, vbslq_f32(vreinterpretq_u32_f32(tail), Activate<type>(vaddq_f32(_dst, conv), _params), _dst));
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

        bool ConvolutionDirectChw::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && ((p.IsStride(1) && p.IsKernel(1)) || p.IsKernel(2) || p.IsKernel(3)) && p.IsChw();
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
                return Base::ConvolutionDirectChw::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Neon::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Neon::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Neon::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Neon::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Neon::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Neon::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            default:
                return Base::ConvolutionDirectChw::SetConvolutionBiasActivation();
            }
            assert(0);
            return NULL;
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm)
        {
            ConvParam param(srcC, srcH, srcW, srcT, dstC, dstT, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, activation, gemm);
            if (!param.Valid())
                return NULL;
            else if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionDirectChw::Preferable(param))
                return new ConvolutionDirectChw(param);
            else if (ConvolutionGemmNT::Preferable(param))
                return new ConvolutionGemmNT(param);
            else
                return new ConvolutionGemmNN(param);
        }
    }
#endif// SIMD_NEON_ENABLE
}
