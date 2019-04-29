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
#include "Simd/SimdMergedConvolution.h"
#include "Simd/SimdConvolution.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float * params, size_t offset);

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<::SimdConvolutionActivationType type> void DepthwiseConvolutionBiasActivation(const float * src,
            size_t srcH, size_t srcW, size_t srcC, size_t dstH, size_t dstW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX,
            const float * weight, const float * bias, const float * params, float * dst)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float * pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float * ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum, params, c);
                    }
                    dst += srcC;
                }
            }
        }

         MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Simd::MergedConvolution(p)
        {
             _srcSize = p.srcH*p.srcW*p.srcC;
             _dstSize = p.dstH*p.dstW*p.dstC;
             _gemm.Init(Base::Gemm32fNN, "Base", p.gemm, "Ext");
             switch (p.activation0)
             {
             case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationIdentity>; break;
             case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRelu>; break;
             case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu>; break;
             case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRestrictRange>; break;
             case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationPrelu>; break;
             default: assert(0);
             }
        }

        size_t MergedConvolution::ExternalBufferSize() const
        {
            const MergConvParam & p = _param;
            return p.dstH*p.dstW*p.srcC;
        }

        void MergedConvolution::Forward(const float * src, float * buf, float * dst)
        {
            const MergConvParam & p = _param;
            for (size_t n = 0; n < p.batch; ++n)
            {
                _depthwise(src, p.srcH, p.srcW, p.srcC, p.dstH, p.dstW, p.kernelY, p.kernelX, p.strideY, p.strideX, p.padY, p.padX, _weight0, _bias0, _params0, buf);
                _gemm.Run(p.dstH * p.dstW, p.dstC, p.srcC, &_1, buf, p.srcC, _weight1, p.dstC, &_0, dst, p.dstC);
                Base::ConvolutionBiasAndActivation(_bias1, p.dstC, p.dstH * p.dstW, p.activation1, _params1, SimdTrue, dst);
                src += _srcSize;
                dst += _dstSize;
            }
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm)
        {
            MergConvParam param(batch, srcC, srcH, srcW, dstC, kernelY, kernelX, strideY, strideX, padY, padX, padH, padW, activation0, activation1, gemm);
            if (!param.Valid())
                return NULL;
            return new MergedConvolution(param);
        }
    }
}
