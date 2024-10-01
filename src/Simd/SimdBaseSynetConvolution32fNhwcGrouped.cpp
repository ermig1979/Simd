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
        static void ConvolutionNhwcGroupedBlock1x2(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t dW = p.kernelY * p.kernelX * p.srcC, srcC = p.srcC;
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
                                    const float* pw0 = weight + (ky * p.kernelX + kx) * srcC, *pw1 = pw0 + dW;
                                    const float* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    float* pd = dst;
                                    for (size_t c = 0; c < srcC; ++c, pd += 2)
                                    {
                                        pd[0] += ps[c] * pw0[c];
                                        pd[1] += ps[c] * pw1[c];
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

        SynetConvolution32fNhwcGroupedBlock1x2::SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p)
            : SynetConvolution32f(p)
        {
            _batch = p.batch;
            _sizeS = p.srcC * p.srcH * p.srcW;
            _sizeD = p.dstC * p.dstH * p.dstW;
            _convolution = ConvolutionNhwcGroupedBlock1x2;
        }

        void SynetConvolution32fNhwcGroupedBlock1x2::SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params)
        {
            SynetConvolution32f::SetParams(weight, internal, bias, params);
            const ConvParam& p = _param;
            size_t size = p.kernelY * p.kernelX * p.srcC;
            _rWeight.Resize(size * 2);
            const float* src = _weight;
            float* dst0 = _rWeight.data, *dst1 = dst0 + size;
            for (size_t i = 0; i < size; ++i)
            {
                dst0[i] = src[0];
                dst1[i] = src[1];
                src += 2;
            }
            _weight = _rWeight.data;
            if (_bias == NULL)
            {
                _rBias.Resize(p.dstC, true);
                _bias = _rBias.data;
            }
        }

        void SynetConvolution32fNhwcGroupedBlock1x2::Forward(const float* src, float* buf, float* dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _convolution(src, _param, _weight, _bias, _params, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fNhwcGroupedBlock1x2::Preferable(const ConvParam& p)
        {
            if (p.trans == 0 || p.group == 1 || p.IsDepthwise())
                return false;
            return p.group == p.srcC && p.dstC == 2 * p.srcC;
        }
    }
#endif
}
