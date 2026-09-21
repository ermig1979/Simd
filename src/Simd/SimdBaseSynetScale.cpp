/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _bias = bias[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale + _bias;
                        dst[s + 1] = src[s + 1] * _scale + _bias;
                        dst[s + 2] = src[s + 2] * _scale + _bias;
                        dst[s + 3] = src[s + 3] * _scale + _bias;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale + _bias;
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale;
                        dst[s + 1] = src[s + 1] * _scale;
                        dst[s + 2] = src[s + 2] * _scale;
                        dst[s + 3] = src[s + 3] * _scale;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale;
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0] + bias[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1] + bias[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2] + bias[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3] + bias[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] + bias[c];
                    src += channels;
                    dst += channels;

                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, spatial, dst);
            else
                assert(0);
        }
    }
#endif
}
