/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            _srcC = p.srcC / p.group;
            _srcH = p.padY + p.srcH + p.padH;
            _srcW = p.padX + p.srcW + p.padW;
            _dstC = p.dstC / p.group;
            _grW = _srcC * _dstC * p.kernelY * p.kernelX;
            _grS = _srcC * p.srcH * p.srcW;
            _grD = _dstC * p.dstH  * p.dstW;
            _pad = p.IsPad(0) ? 0 : 1;
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        size_t SynetConvolution32fDirectNchw::ExternalBufferSize() const
        {
            if (_pad)
                return _srcC*_srcH*_srcW;
            else
                return 1;
        }

        void SynetConvolution32fDirectNchw::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            if(_pad)
                buf = Buffer(buf);
            for (size_t b = 0; b < p.batch; ++b)
            {
                const float * weight = _weight;
                const float * bias = _bias;
                const float * params = _params;
                for (size_t g = 0; g < p.group; ++g)
                {
                    if (_pad)
                    {
                        Pad(src, buf);
                        _convolutionBiasActivation(buf, _srcC, _srcH, _srcW, weight, bias, params, dst, _dstC, p.dstH, p.dstW);
                    }
                    else
                        _convolutionBiasActivation(src, _srcC, _srcH, _srcW, weight, bias, params, dst, _dstC, p.dstH, p.dstW);
                    weight += _grW;
                    if (bias)
                        bias += _dstC;
                    if (p.activation == ::SimdConvolutionActivationPrelu)
                        params += _dstC;
                    src += _grS;
                    dst += _grD;
                }
            }
        }

        bool SynetConvolution32fDirectNchw::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && (p.IsKernel(2) || p.IsKernel(3)) && p.trans == 0;
        }

        void SynetConvolution32fDirectNchw::Pad(const float * src, float * dst) const
        {
            const ConvParam & p = _param;
            for (size_t c = 0; c < _srcC; ++c)
            {
                if (p.padY)
                {
                    memset(dst, 0, p.padY*_srcW * sizeof(float));
                    dst += p.padY*_srcW;
                }
                for (size_t row = 0; row < p.srcH; ++row)
                {
                    for (size_t col = 0; col < p.padX; ++col)
                        *dst++ = 0;
                    memcpy(dst, src, p.srcW * sizeof(float));
                    dst += p.srcW;
                    src += p.srcW;
                    for (size_t col = 0; col < p.padW; ++col)
                        *dst++ = 0;
                }
                if (p.padH)
                {
                    memset(dst, 0, p.padH*_srcW * sizeof(float));
                    dst += p.padH*_srcW;
                }
            }
        }

        SIMD_INLINE void AddConvolutionKernel1x1(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += src[sx]*weight[0];
                src += srcW * strideY;
                dst += dstW;
            }
        }

        SIMD_INLINE float ConvolutionKernel2(const float * src, const float * weight)
        {
            return src[0] * weight[0] + src[1] * weight[1];
        }

        SIMD_INLINE float ConvolutionKernel2x2(const float * src, size_t srcW, const float * weight)
        {
            return
                ConvolutionKernel2(src, weight) +
                ConvolutionKernel2(src + srcW, weight + 2);
        }

        SIMD_INLINE void AddConvolutionKernel2x2(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += ConvolutionKernel2x2(src + sx, srcW, weight);
                src += srcW * strideY;
                dst += dstW;
            }
        }

        SIMD_INLINE float ConvolutionKernel3(const float * src, const float * weight)
        {
            return src[0] * weight[0] + src[1] * weight[1] + src[2] * weight[2];
        }

        SIMD_INLINE float ConvolutionKernel3x3(const float * src, size_t srcW, const float * weight)
        {
            return
                ConvolutionKernel3(src, weight) +
                ConvolutionKernel3(src + srcW, weight + 3) +
                ConvolutionKernel3(src + 2 * srcW, weight + 6);
        }

        SIMD_INLINE void AddConvolutionKernel3x3(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += ConvolutionKernel3x3(src + sx, srcW, weight);
                src += srcW * strideY;
                dst += dstW;
            }
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                Fill32f(dst, dstW * dstH, bias ? bias + dc : NULL);
                for (size_t sc = 0; sc < srcC; ++sc)
                {
                    const float * ps = src + sc * srcW * srcH;
                    const float * pw = weight + (dc*srcC + sc)*kernel*kernel;
                    float * pd = dst;
                    if (kernel == 1)
                        AddConvolutionKernel1x1(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else if (kernel == 2)
                        AddConvolutionKernel2x2(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else if (kernel == 3)
                        AddConvolutionKernel3x3(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else
                    {
                        for (size_t dy = 0; dy < dstH; ++dy)
                        {
                            for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += stride)
                            {
                                float sum = 0;
                                for (size_t ky = 0; ky < kernel; ++ky)
                                {
                                    const float * s = ps + ky * srcW + sx;
                                    const float * w = pw + kernel*ky;
                                    for (size_t kx = 0; kx < kernel; ++kx)
                                        sum += s[kx] * w[kx];
                                }
                                pd[dx] += sum;
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                    }
                }
                ConvolutionBiasAndActivation(NULL, 1, dstH*dstW, type, params, ::SimdFalse, dst);
                if (type == ::SimdConvolutionActivationPrelu)
                    params++;
                dst += dstW * dstH;
            }
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
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Base::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Base::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2: 
                if (p.kernelX == 2)
                    return Base::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3: 
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            }
            return NULL;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            _count = p.srcC;
            _size = p.srcH*p.srcW;
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            size_t i = 0, aligned = size&(~3);
            float sums[4] = { 0, 0, 0, 0 };
            for (; i < aligned; i += 4)
            {
                sums[0] += a[i + 0] * b[i + 0];
                sums[1] += a[i + 1] * b[i + 1];
                sums[2] += a[i + 2] * b[i + 2];
                sums[3] += a[i + 3] * b[i + 3];
            }
            for (; i < size; ++i)
                sums[0] += a[i] * b[i];
            return sums[0] + sums[1] + sums[2] + sums[3];
        }
       
        void SynetConvolution32fDepthwiseDotProduct::Forward(const float * src, float * buf, float * dst)
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
                    ConvolutionBiasAndActivation(NULL, _count, 1, _param.activation, _params, ::SimdFalse, dst);                
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fDepthwiseDotProduct::Preferable(const ConvParam & p)
        {
            if (!(p.IsPad(0) && p.IsDilation(1) && p.IsStride(1)))
                return false;
            if (!(p.dstC == p.srcC && p.dstC == p.group && p.srcW == p.kernelX && p.srcH == p.kernelY))
                return false;
            return p.trans == 0;
        }
    }
#endif
}
