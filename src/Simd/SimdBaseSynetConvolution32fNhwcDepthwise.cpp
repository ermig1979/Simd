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
        SynetConvolution32fDirectNhwc::SynetConvolution32fDirectNhwc(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        void SynetConvolution32fDirectNhwc::Forward(const float * src, float * buf, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _convolutionBiasActivation(src, _param, _weight, _bias, _params, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fDirectNhwc::Preferable(const ConvParam & p)
        {
            if (p.trans == 0)
                return false;
            if (p.group == 1)
            {
                double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
                return k < 2.0;
            }
            return p.IsDepthwise();
        }

        static void ConvolutionDirectNhwcConvolutionBiasActivationDefault(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t group = p.group;
            size_t srcC = p.srcC / group;
            size_t dstC = p.dstC / group;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    memset(dst, 0, p.dstC * sizeof(float));
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

                                    const float * pw = weight + (ky*p.kernelX + kx)*srcC*p.dstC;
                                    const float * ps = src + (sy*p.srcW + sx)*p.srcC;
                                    if (group == 1)
                                    {
                                        for (size_t sc = 0; sc < srcC; ++sc)
                                        {
                                            for (size_t dc = 0; dc < dstC; ++dc)
                                                dst[dc] += ps[0] * pw[dc];
                                            ps += 1;
                                            pw += dstC;
                                        }
                                    }
                                    else
                                    {
                                        for (size_t g = 0; g < group; ++g)
                                            dst[g] += ps[g] * pw[g];
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

        SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation()
        {
            return ConvolutionDirectNhwcConvolutionBiasActivationDefault;
        }
    }
#endif
}
