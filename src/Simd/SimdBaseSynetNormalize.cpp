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
#include "Simd/SimdArray.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetNormalizeLayerForward(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, 
            const float* eps, SimdBool acrossSpatial, SimdTensorFormatType format, float* buf, float* dst)
        {
            float _eps = eps[0];
            if (format == SimdTensorFormatNchw)
            {
                if (acrossSpatial)
                {
                    size_t size = channels * spatial;
                    for (size_t b = 0; b < batch; ++b)
                    {
                        float sum = _eps;
                        for (size_t i = 0; i < size; ++i)
                            sum += Simd::Square(src[i]);
                        float k0 = 1.0f / ::sqrt(sum);
                        for (size_t c = 0; c < channels; ++c)
                        {
                            float k = scale[c] * k0;
                            for (size_t s = 0; s < spatial; ++s)
                                dst[s] = src[s] * k;
                            dst += spatial;
                            src += spatial;
                        }
                    }
                }
                else
                {
                    Array32f _buf;
                    if (buf == NULL)
                    {
                        _buf.Resize(spatial);
                        buf = _buf.data;
                    }
                    for (size_t b = 0; b < batch; ++b)
                    {
                        for (size_t s = 0; s < spatial; ++s)
                            buf[s] = _eps;
                        for (size_t c = 0; c < channels; ++c)
                        {
                            const float* ps = src + c * spatial;
                            for (size_t s = 0; s < spatial; ++s)
                                buf[s] += Simd::Square(ps[s]);
                        }
                        for (size_t s = 0; s < spatial; ++s)
                            buf[s] = 1.0f / ::sqrt(buf[s]);
                        for (size_t c = 0; c < channels; ++c)
                        {
                            float k = scale[c];
                            for (size_t s = 0; s < spatial; ++s)
                                dst[s] = src[s] * buf[s] * k;
                            dst += spatial;
                            src += spatial;
                        }
                    }
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                if (acrossSpatial)
                {
                    size_t size = channels * spatial;
                    for (size_t b = 0; b < batch; ++b)
                    {
                        float sum = _eps;
                        for (size_t i = 0; i < size; ++i)
                            sum += Simd::Square(src[i]);
                        float k = 1.0f / ::sqrt(sum);
                        for (size_t s = 0; s < spatial; ++s)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c] * k;
                            dst += channels;
                            src += channels;
                        }
                    }
                }
                else
                {
                    for (size_t b = 0; b < batch; ++b)
                    {
                        for (size_t s = 0; s < spatial; ++s)
                        {
                            float sum = _eps;
                            for (size_t c = 0; c < channels; ++c)
                                sum += Simd::Square(src[c]);
                            float k = 1.0f / ::sqrt(sum);
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c] * k;
                            dst += channels;
                            src += channels;
                        }
                    }
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetNormalizeLayerForwardV2(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            float k = 1.0f / float(channels), e = *eps;
            if (format == SimdTensorFormatNchw)
            {
                Array32f _buf;
                if (buf == NULL)
                {
                    _buf.Resize(spatial);
                    buf = _buf.data;
                }
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        buf[s] = 0;
                    for (size_t c = 0, o = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s, ++o)
                            buf[s] += src[o];
                    }
                    for (size_t s = 0; s < spatial; ++s)
                        buf[s] = buf[s] * k;
                    for (size_t c = 0, o = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s, ++o)
                            dst[o] = src[o] - buf[s];
                    }

                    for (size_t s = 0; s < spatial; ++s)
                        buf[s] = 0;
                    for (size_t c = 0, o = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s, ++o)
                            buf[s] += Simd::Square(dst[o]);
                    }
                    for (size_t s = 0; s < spatial; ++s)
                        buf[s] = 1.0f / ::sqrt(buf[s] * k + e);
                    for (size_t c = 0, o = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s, ++o)
                            dst[o] = dst[o] * buf[s] * scale[c] + shift[c];
                    }

                    src += channels * spatial;
                    dst += channels * spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        float sum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sum += src[c];
                        float mean = sum * k;
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] - mean;

                        float sqsum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sqsum += Simd::Square(dst[c]);
                        float norm = 1.0f / ::sqrt(sqsum * k + e);
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = dst[c] * norm * scale[c] + shift[c];

                        dst += channels;
                        src += channels;
                    }
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetNormalizeLayerForwardV3(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            float k = 1.0f / float(spatial), e = *eps;

            if (format == SimdTensorFormatNchw)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float sum = 0;
                        for (size_t s = 0; s < spatial; ++s)
                            sum += src[s];
                        float mean = sum * k;
                        for (size_t s = 0; s < spatial; ++s)
                            dst[s] = src[s] - mean;

                        float sqsum = 0;
                        for (size_t s = 0; s < spatial; ++s)
                            sqsum += Simd::Square(dst[s]);
                        float norm = 1.0f / ::sqrt(sqsum * k + e);
                        for (size_t s = 0; s < spatial; ++s)
                            dst[s] = dst[s] * norm * scale[c] + shift[c];

                        dst += spatial;
                        src += spatial;
                    }
                }                

            }
            else if (format == SimdTensorFormatNhwc)
            {
                Array32f _buf;
                if (buf == NULL)
                {
                    _buf.Resize(channels);
                    buf = _buf.data;
                }
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t c = 0; c < channels; ++c)
                        buf[c] = 0;
                    for (size_t s = 0, o = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c, ++o)
                            buf[c] += src[o];
                    }
                    for (size_t c = 0; c < channels; ++c)
                        buf[c] = buf[c] * k;
                    for (size_t s = 0, o = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c, ++o)
                            dst[o] = src[o] - buf[c];
                    }

                    for (size_t c = 0; c < channels; ++c)
                        buf[c] = 0;
                    for (size_t s = 0, o = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c, ++o)
                            buf[c] += Simd::Square(dst[o]);
                    }
                    for (size_t c = 0; c < channels; ++c)
                        buf[c] = 1.0f / ::sqrt(buf[c] * k + e);
                    for (size_t s = 0, o = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c, ++o)
                            dst[o] = dst[o] * buf[c] * scale[c] + shift[c];
                    }

                    src += channels * spatial;
                    dst += channels * spatial;
                }
            }
            else
                assert(0);
        }
    }
#endif
}
