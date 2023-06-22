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
#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        void NormalizeNchw1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            __mmask16 sizeMask = TailMask16(size - sizeF);
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 spatialMask = TailMask16(spatial - spatialF);
            for (size_t b = 0; b < batch; ++b)
            {
                __m512 _sum = _mm512_setzero_ps();
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    __m512 _src = _mm512_loadu_ps(src + i);
                    _sum = _mm512_fmadd_ps(_src, _src, _sum);
                }
                if (i < size)
                {
                    __m512 _src = _mm512_maskz_loadu_ps(sizeMask, src + i);
                    _sum = _mm512_fmadd_ps(_src, _src, _sum);
                }
                float k0 = 1.0f / ::sqrt(ExtractSum(_sum) + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _k = _mm512_set1_ps(scale[c] * k0);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        _mm512_storeu_ps(dst + s, _mm512_mul_ps(_mm512_loadu_ps(src + s), _k));
                    if(s < spatial)
                        _mm512_mask_storeu_ps(dst + s, spatialMask, _mm512_mul_ps(_mm512_maskz_loadu_ps(spatialMask, src + s), _k));
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNchw0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* buf, float* dst)
        {
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 spatialMask = TailMask16(spatial - spatialF);
            __m512 _eps = _mm512_set1_ps(eps), _1 = _mm512_set1_ps(1.0f);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_setzero_ps());
                if(s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialMask, _mm512_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _src = _mm512_loadu_ps(ps + s);
                        _mm512_storeu_ps(buf + s, _mm512_fmadd_ps(_src, _src, _mm512_loadu_ps(buf + s)));
                    }
                    if(s < spatial)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(spatialMask, ps + s);
                        _mm512_mask_storeu_ps(buf + s, spatialMask, _mm512_fmadd_ps(_src, _src, _mm512_maskz_loadu_ps(spatialMask, buf + s)));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_div_ps(_1, _mm512_sqrt_ps(_mm512_add_ps(_mm512_loadu_ps(buf + s), _eps))));
                if (s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialMask, _mm512_div_ps(_1, 
                        _mm512_sqrt_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(spatialMask, buf + s), _eps))));
                 for (size_t c = 0; c < channels; ++c)
                {
                    float k = scale[c];
                    __m512 _k = _mm512_set1_ps(k);
                    for (s = 0; s < spatialF; s += F)
                        _mm512_storeu_ps(dst + s, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(src + s), _mm512_loadu_ps(buf + s)), _k));
                    if(s < spatial)
                        _mm512_mask_storeu_ps(dst + s, spatialMask, _mm512_mul_ps(_mm512_mul_ps
                        (_mm512_maskz_loadu_ps(spatialMask, src + s), _mm512_maskz_loadu_ps(spatialMask, buf + s)), _k));
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwc1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            __mmask16 sizeMask = TailMask16(size - sizeF);
            size_t channelsF = AlignLo(channels, F);
            __mmask16 channelsMask = TailMask16(channels - channelsF);
            for (size_t b = 0; b < batch; ++b)
            {
                __m512 _sum = _mm512_setzero_ps();
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    __m512 _src = _mm512_loadu_ps(src + i);
                    _sum = _mm512_fmadd_ps(_src, _src, _sum);
                }
                if (i < size)
                {
                    __m512 _src = _mm512_maskz_loadu_ps(sizeMask, src + i);
                    _sum = _mm512_fmadd_ps(_src, _src, _sum);
                }
                __m512 _k = _mm512_set1_ps(1.0f / ::sqrt(ExtractSum(_sum) + eps));
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        _mm512_storeu_ps(dst + c, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(src + c), _mm512_loadu_ps(scale + c)), _k));
                    if(c < channels)
                        _mm512_mask_storeu_ps(dst + c, channelsMask, _mm512_mul_ps(_mm512_mul_ps(
                            _mm512_maskz_loadu_ps(channelsMask, src + c), _mm512_maskz_loadu_ps(channelsMask, scale + c)), _k));
                    dst += channels;
                    src += channels;
                }
            }
        }

        void NormalizeNhwc0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            __mmask16 channelsMask = TailMask16(channels - channelsF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    __m512 _sum = _mm512_setzero_ps();
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                    {
                        __m512 _src = _mm512_loadu_ps(src + c);
                        _sum = _mm512_fmadd_ps(_src, _src, _sum);
                    }
                    if (c < channels)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(channelsMask, src + c);
                        _sum = _mm512_fmadd_ps(_src, _src, _sum);
                    }
                    __m512 _k = _mm512_set1_ps(1.0f / ::sqrt(ExtractSum(_sum) + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm512_storeu_ps(dst + c, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(src + c), _mm512_loadu_ps(scale + c)), _k));
                    if (c < channels)
                        _mm512_mask_storeu_ps(dst + c, channelsMask, _mm512_mul_ps(_mm512_mul_ps(
                            _mm512_maskz_loadu_ps(channelsMask, src + c), _mm512_maskz_loadu_ps(channelsMask, scale + c)), _k));
                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForward(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale,
            const float* eps, SimdBool acrossSpatial, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                if (acrossSpatial)
                    NormalizeNchw1(src, batch, channels, spatial, scale, eps[0], dst);
                else
                    NormalizeNchw0(src, batch, channels, spatial, scale, eps[0], buf, dst);
            }
            else if (format == SimdTensorFormatNhwc)
            {
                if (acrossSpatial)
                    NormalizeNhwc1(src, batch, channels, spatial, scale, eps[0], dst);
                else
                    NormalizeNhwc0(src, batch, channels, spatial, scale, eps[0], dst);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV2(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 spatialM = TailMask16(spatial - spatialF);
            __m512 _eps = _mm512_set1_ps(eps), _k = _mm512_set1_ps(k), _1 = _mm512_set1_ps(1.0f);
            for (size_t b = 0, s; b < batch; ++b)
            {
                for (s = 0; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_setzero_ps());
                if(s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _src = _mm512_loadu_ps(ps + s);
                        __m512 _sum = _mm512_loadu_ps(buf + s);
                        _mm512_storeu_ps(buf + s, _mm512_add_ps(_sum, _src));
                    }
                    if(s < spatial)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(spatialM, ps + s);
                        __m512 _sum = _mm512_maskz_loadu_ps(spatialM, buf + s);
                        _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_add_ps(_sum, _src));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_mul_ps(_mm512_loadu_ps(buf + s), _k));
                if(s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_mul_ps(_mm512_maskz_loadu_ps(spatialM, buf + s), _k));
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _src = _mm512_loadu_ps(ps + s);
                        __m512 mean = _mm512_loadu_ps(buf + s);
                        _mm512_storeu_ps(pd + s, _mm512_sub_ps(_src, mean));
                    }
                    if(s < spatial)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(spatialM, ps + s);
                        __m512 mean = _mm512_maskz_loadu_ps(spatialM, buf + s);
                        _mm512_mask_storeu_ps(pd + s, spatialM, _mm512_sub_ps(_src, mean));
                    }
                }

                for (s = 0; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_setzero_ps());
                if(s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_setzero_ps());
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(pd + s);
                        __m512 _sum = _mm512_loadu_ps(buf + s);
                        _mm512_storeu_ps(buf + s, _mm512_fmadd_ps(_dst, _dst, _sum));
                    }
                    if(s < spatial)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(spatialM, pd + s);
                        __m512 _sum = _mm512_maskz_loadu_ps(spatialM, buf + s);
                        _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_fmadd_ps(_dst, _dst, _sum));
                    }
                }
                for (s = 0; s < spatialF; s += F)
                    _mm512_storeu_ps(buf + s, _mm512_div_ps(_1, _mm512_sqrt_ps(_mm512_fmadd_ps(_mm512_loadu_ps(buf + s), _k, _eps))));
                if (s < spatial)
                    _mm512_mask_storeu_ps(buf + s, spatialM, _mm512_div_ps(_1, _mm512_sqrt_ps(_mm512_fmadd_ps(_mm512_maskz_loadu_ps(spatialM, buf + s), _k, _eps))));
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(pd + s);
                        __m512 norm = _mm512_loadu_ps(buf + s);
                        _mm512_storeu_ps(pd + s, _mm512_fmadd_ps(_mm512_mul_ps(_dst, norm), _scale, _shift));
                    }
                    if(s < spatial)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(spatialM, pd + s);
                        __m512 norm = _mm512_maskz_loadu_ps(spatialM, buf + s);
                        _mm512_mask_storeu_ps(pd + s, spatialM, _mm512_fmadd_ps(_mm512_mul_ps(_dst, norm), _scale, _shift));
                    }
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }

        void NormalizeNhwcV2(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            __mmask16 channelsM = TailMask16(channels - channelsF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    __m512 _sum = _mm512_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                        _sum = _mm512_add_ps(_mm512_loadu_ps(src + c), _sum);
                    if(c < channels)
                        _sum = _mm512_add_ps(_mm512_maskz_loadu_ps(channelsM, src + c), _sum);
                    __m512 mean = _mm512_set1_ps(ExtractSum(_sum) * k);
                    for (c = 0; c < channelsF; c += F)
                        _mm512_storeu_ps(dst + c, _mm512_sub_ps(_mm512_loadu_ps(src + c), mean));
                    if(c < channels)
                        _mm512_mask_storeu_ps(dst + c, channelsM, _mm512_sub_ps(_mm512_maskz_loadu_ps(channelsM, src + c), mean));

                    __m512 _sqsum = _mm512_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(dst + c);
                        _sqsum = _mm512_fmadd_ps(_dst, _dst, _sqsum);
                    }
                    if(c < channels)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(channelsM, dst + c);
                        _sqsum = _mm512_fmadd_ps(_dst, _dst, _sqsum);
                    }
                    __m512 norm = _mm512_set1_ps(1.0f / ::sqrt(ExtractSum(_sqsum) * k + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm512_storeu_ps(dst + c, _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(dst + c), norm), _mm512_loadu_ps(scale + c), _mm512_loadu_ps(shift + c)));
                    if(c < channels)
                        _mm512_mask_storeu_ps(dst + c, channelsM, _mm512_fmadd_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(channelsM, dst + c), norm), 
                            _mm512_maskz_loadu_ps(channelsM, scale + c), _mm512_maskz_loadu_ps(channelsM, shift + c)));

                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForwardV2(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV2(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV2(src, batch, channels, spatial, scale, shift, *eps, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV3(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* dst)
        {
            float k = 1.0f / float(spatial);
            size_t spatialF = AlignLo(spatial, F), s;
            __mmask16 spatialM = TailMask16(spatial - spatialF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _sum = _mm512_setzero_ps();
                    for (s = 0; s < spatialF; s += F)
                        _sum = _mm512_add_ps(_mm512_loadu_ps(src + s), _sum);
                    if(s < spatial)
                        _sum = _mm512_add_ps(_mm512_maskz_loadu_ps(spatialM, src + s), _sum);
                    float sum = ExtractSum(_sum);
                    __m512 mean = _mm512_set1_ps(sum * k);
                    for (s = 0; s < spatialF; s += F)
                        _mm512_storeu_ps(dst + s, _mm512_sub_ps(_mm512_loadu_ps(src + s), mean));
                    if (s < spatial)
                        _mm512_mask_storeu_ps(dst + s, spatialM, _mm512_sub_ps(_mm512_maskz_loadu_ps(spatialM, src + s), mean));

                    __m512 _sqsum = _mm512_setzero_ps();
                    for (s = 0; s < spatialF; s += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(dst + s);
                        _sqsum = _mm512_fmadd_ps(_dst, _dst, _sqsum);
                    }
                    if (s < spatial)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(spatialM, dst + s);
                        _sqsum = _mm512_fmadd_ps(_dst, _dst, _sqsum);
                    }
                    float sqsum = ExtractSum(_sqsum);
                    __m512 norm = _mm512_set1_ps(1.0f / ::sqrt(sqsum * k + eps));
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    for (s = 0; s < spatialF; s += F)
                        _mm512_storeu_ps(dst + s, _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(dst + s), norm), _scale), _shift));
                    if (s < spatial)
                        _mm512_mask_storeu_ps(dst + s, spatialM, _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(spatialM, dst + s), norm), _scale), _shift));

                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwcV3(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(spatial);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(spatial);
                buf = _buf.data;
            }
            size_t channelsF = AlignLo(channels, F);
            __mmask16 channelsM = TailMask16(channels - channelsF);
            __m512 _eps = _mm512_set1_ps(eps), _k = _mm512_set1_ps(k), _1 = _mm512_set1_ps(1.0f);
            for (size_t b = 0, c; b < batch; ++b)
            {
                for (c = 0; c < channelsF; c += F)
                    _mm512_storeu_ps(buf + c, _mm512_setzero_ps());
                if(c < channels)
                    _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_setzero_ps());
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _src = _mm512_loadu_ps(ps + c);
                        __m512 _sum = _mm512_loadu_ps(buf + c);
                        _mm512_storeu_ps(buf + c, _mm512_add_ps(_sum, _src));
                    }
                    if (c < channels)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(channelsM, ps + c);
                        __m512 _sum = _mm512_maskz_loadu_ps(channelsM, buf + c);
                        _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_add_ps(_sum, _src));
                    }
                }
                for (c = 0; c < channelsF; c += F)
                    _mm512_storeu_ps(buf + c, _mm512_mul_ps(_mm512_loadu_ps(buf + c), _k));
                if (c < channels)
                    _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_mul_ps(_mm512_maskz_loadu_ps(channelsM, buf + c), _k));
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _src = _mm512_loadu_ps(ps + c);
                        __m512 mean = _mm512_loadu_ps(buf + c);
                        _mm512_storeu_ps(pd + c, _mm512_sub_ps(_src, mean));
                    }
                    if (c < channels)
                    {
                        __m512 _src = _mm512_maskz_loadu_ps(channelsM, ps + c);
                        __m512 mean = _mm512_maskz_loadu_ps(channelsM, buf + c);
                        _mm512_mask_storeu_ps(pd + c, channelsM, _mm512_sub_ps(_src, mean));
                    }
                }

                for (c = 0; c < channelsF; c += F)
                    _mm512_storeu_ps(buf + c, _mm512_setzero_ps());
                if (c < channels)
                    _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_setzero_ps());
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(pd + c);
                        __m512 _sum = _mm512_loadu_ps(buf + c);
                        _mm512_storeu_ps(buf + c, _mm512_fmadd_ps(_dst, _dst, _sum));
                    }
                    if (c < channels)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(channelsM, pd + c);
                        __m512 _sum = _mm512_maskz_loadu_ps(channelsM, buf + c);
                        _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_fmadd_ps(_dst, _dst, _sum));
                    }
                }
                for (c = 0; c < channelsF; c += F)
                    _mm512_storeu_ps(buf + c, _mm512_div_ps(_1, _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(buf + c), _k), _eps))));
                if (c < channels)
                    _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_div_ps(_1, _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(channelsM, buf + c), _k), _eps))));
                for (size_t s = 0; s < spatial; ++s)
                {
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _dst = _mm512_loadu_ps(pd + c);
                        __m512 norm = _mm512_loadu_ps(buf + c);
                        _mm512_storeu_ps(pd + c, _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_dst, norm), _mm512_loadu_ps(scale + c)), _mm512_loadu_ps(shift + c)));
                    }
                    if (c < channels)
                    {
                        __m512 _dst = _mm512_maskz_loadu_ps(channelsM, pd + c);
                        __m512 norm = _mm512_maskz_loadu_ps(channelsM, buf + c);
                        _mm512_mask_storeu_ps(pd + c, channelsM, _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_dst, norm), _mm512_maskz_loadu_ps(channelsM, scale + c)), _mm512_maskz_loadu_ps(channelsM, shift + c)));
                    }
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }

        void SynetNormalizeLayerForwardV3(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV3(src, batch, channels, spatial, scale, shift, *eps, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV3(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
