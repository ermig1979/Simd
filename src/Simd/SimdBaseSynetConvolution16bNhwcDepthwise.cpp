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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template <class T> SIMD_INLINE float ToFloat(T src);

        template <> SIMD_INLINE float ToFloat<float>(float src)
        {
            return src;
        }

        template <> SIMD_INLINE float ToFloat<uint16_t>(uint16_t src)
        {
            return BFloat16ToFloat32(src);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwiseDefault(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.trans && p.IsDepthwise());
            const T* src = (T*)src8;
            size_t group = p.group, elem = (term == Term16bLast16b ? 2 : 4);
            Array32f buf(group);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    buf.Clear();
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {

                                    const float* pw = weight + (ky * p.kernelX + kx) * group;
                                    const T* ps = src + (sy * p.srcW + sx) * group;
                                    for (size_t g = 0; g < group; ++g)
                                        buf[g] += ToFloat(ps[g]) * pw[g];
                                }
                            }
                        }
                    }
                    for (size_t g = 0; g < group; ++g)
                        Base::Save1<term, type>(dst, buf[g], bias, params, g);
                    dst += group * elem;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.dstT == SimdTensorData32f)
                convolution = Convolution16bNhwcDepthwiseDefault<T, Term16bLast32f, type>;
            else
                convolution = Convolution16bNhwcDepthwiseDefault<T, Term16bLast16b, type>;
        }

        template<SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.srcT == SimdTensorData16b)
                SetConvolution<uint16_t, type>(p, convolution);
            else
                SetConvolution<float, type>(p, convolution);
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcDepthwise::SynetConvolution16bNhwcDepthwise(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _stepS = p.srcC * p.srcH * p.srcW * _elemS;
            _stepD = p.dstC * p.dstH * p.dstW * _elemD;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationRelu: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationRestrictRange: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationPrelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationElu: SetConvolution<SimdConvolutionActivationElu>(p, _convolution); break;
            case SimdConvolutionActivationHswish: SetConvolution<SimdConvolutionActivationHswish>(p, _convolution); break;
            case SimdConvolutionActivationMish: SetConvolution<SimdConvolutionActivationMish>(p, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: SetConvolution<SimdConvolutionActivationHardSigmoid>(p, _convolution); break;
            case SimdConvolutionActivationSwish: SetConvolution<SimdConvolutionActivationSwish>(p, _convolution); break;
            case SimdConvolutionActivationGelu: SetConvolution<SimdConvolutionActivationGelu>(p, _convolution); break;
            }
        }

        String SynetConvolution16bNhwcDepthwise::Desc() const 
        {
            return Ext() + "::NhwcDepthwise"; 
        }
        size_t SynetConvolution16bNhwcDepthwise::InternalBufferSize() const
        {
            return SynetConvolution16b::InternalBufferSize() + _weight.RawSize();
        }

        void SynetConvolution16bNhwcDepthwise::SetParams(const float* weight, const float* bias, const float* params)
        {
            const ConvParam& p = _param;
            _weight.Assign(weight, p.kernelX * p.kernelY * p.srcC);
            SynetConvolution16b::SetBias(bias, SIMD_ALIGN);
            SynetConvolution16b::SetParams(params, SIMD_ALIGN);
        }

        void SynetConvolution16bNhwcDepthwise::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            for (size_t b = 0; b < p.batch; b += 1)
            {
                _convolution(src, _param, _weight.data, _bias.data, _params.data, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        bool SynetConvolution16bNhwcDepthwise::Preferable(const ConvParam& p)
        {
            return p.trans && p.IsDepthwise();
        }
    }
#endif
}
