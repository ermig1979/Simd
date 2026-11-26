/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution16bNhwcSpecV4::SynetConvolution16bNhwcSpecV4(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _preprocess = 0;
            _convolution = 0;
        }

        String SynetConvolution16bNhwcSpecV4::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcSpecV4";
            if (_param.dstW < 32)
                desc << "-n";
            return desc.str();
        }

        void SynetConvolution16bNhwcSpecV4::SetAlgParam()
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.srcH = p.srcH + p.padY + p.padH;
            a.srcW = p.srcW + p.padX + p.padW;
            a.bufS = a.srcH * a.srcW * p.srcC;
            a.weightStep = p.kernelY * p.kernelX * (p.srcC / 32) * 1024;
            a.elem = _elemD;

            _stepS = p.srcH * p.srcW * p.srcC * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * _elemD;
        }

        size_t SynetConvolution16bNhwcSpecV4::ExternalBufferSize() const
        {
            return _alg.bufS * sizeof(uint16_t);
        }

        void SynetConvolution16bNhwcSpecV4::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, 32);
            SynetConvolution16b::SetParams(params, 32);
        }

        void SynetConvolution16bNhwcSpecV4::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.weightStep * (p.dstC / 32), true);
            uint16_t* dst = _weight.data;
            for (size_t oc0 = 0; oc0 < p.dstC; oc0 += 32)
            {
                for (size_t y = 0; y < p.kernelY; ++y)
                {
                    for (size_t x = 0; x < p.kernelX; ++x)
                    {
                        for (size_t ic0 = 0; ic0 < p.srcC; ic0 += 32)
                        {
                            for (size_t tile = 0; tile < 2; ++tile)
                            {
                                for (size_t pair = 0; pair < 16; ++pair)
                                {
                                    for (size_t n = 0; n < 16; ++n)
                                    {
                                        size_t c0 = ic0 + pair * 2;
                                        size_t o = oc0 + tile * 16 + n;
                                        const float* src = weight + ((y * p.kernelX + x) * p.srcC + c0) * p.dstC + o;
                                        *dst++ = Float32ToBFloat16(src[0]);
                                        *dst++ = Float32ToBFloat16(src[p.dstC]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        void SynetConvolution16bNhwcSpecV4::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* buf = Allocate<uint16_t>(buf8, a.bufS);
            for (size_t b = 0; b < p.batch; ++b)
            {
                _preprocess(src, p, a, buf);
                _convolution(buf, p, a, _weight.data, _bias.data, _params.data, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        bool SynetConvolution16bNhwcSpecV4::Preferable(const ConvParam& p)
        {
            return false && p.trans != 0 && p.group == 1 && p.IsDilation(1) && p.IsStride(1) && !p.IsKernel(1) &&
                p.srcC >= 32 && p.dstC >= 32 && (p.srcC % 32) == 0 && (p.dstC % 32) == 0;
        }
    }
#endif
}
