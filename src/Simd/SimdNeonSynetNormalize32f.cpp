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
#include "Simd/SimdSynet.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        void NormalizeNchw1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            size_t spatialF = AlignLo(spatial, F);
            for (size_t b = 0; b < batch; ++b)
            {
                float32x4_t _sum = vdupq_n_f32(0.0f);
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    float32x4_t _src = vld1q_f32(src + i);
                    _sum = vaddq_f32(_sum, vmulq_f32(_src, _src));
                }
                float sum = ExtractSum32f(_sum);
                for (; i < size; ++i)
                    sum += Simd::Square(src[i]);
                float k0 = 1.0f / ::sqrt(sum + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _k = vdupq_n_f32(scale[c] * k0);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        vst1q_f32(dst + s, vmulq_f32(vld1q_f32(src + s), _k));
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * scale[c] * k0;
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
            float32x4_t _eps = vdupq_n_f32(eps), _1 = vdupq_n_f32(1.0f);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF; s += F)
                    vst1q_f32(buf + s, vdupq_n_f32(0.0f));
                for (; s < spatial; ++s)
                    buf[s] = 0.0f;
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t _src = vld1q_f32(ps + s);
                        float32x4_t _sum = vld1q_f32(buf + s);
                        vst1q_f32(buf + s, vaddq_f32(_sum, vmulq_f32(_src, _src)));
                    }
                    for (; s < spatial; ++s)
                        buf[s] += Simd::Square(ps[s]);
                }
                for (s = 0; s < spatialF; s += F)
                    vst1q_f32(buf + s, ReciprocalSqrt<1>(vaddq_f32(vld1q_f32(buf + s), _eps)));
                for (; s < spatial; ++s)
                    buf[s] = 1.0f / ::sqrt(buf[s] + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    float k = scale[c];
                    float32x4_t _k = vdupq_n_f32(k);
                    for (s = 0; s < spatialF; s += F)
                        vst1q_f32(dst + s, vmulq_f32(vmulq_f32(vld1q_f32(src + s), vld1q_f32(buf + s)), _k));
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * buf[s] * k;
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwc1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t size = channels * spatial;
            size_t sizeF = AlignLo(size, F);
            size_t channelsF = AlignLo(channels, F);
            for (size_t b = 0; b < batch; ++b)
            {
                float32x4_t _sum = vdupq_n_f32(0.0f);
                size_t i = 0;
                for (; i < sizeF; i += F)
                {
                    float32x4_t _src = vld1q_f32(src + i);
                    _sum = vaddq_f32(_sum, vmulq_f32(_src, _src));
                }
                float sum = ExtractSum32f(_sum);
                for (; i < size; ++i)
                    sum += Simd::Square(src[i]);
                float32x4_t _k = vdupq_n_f32(1.0f / ::sqrt(sum + eps));
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        vst1q_f32(dst + c, vmulq_f32(vmulq_f32(vld1q_f32(src + c), vld1q_f32(scale + c)), _k));
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] * vgetq_lane_f32(_k, 0);
                    dst += channels;
                    src += channels;
                }
            }
        }

        void NormalizeNhwc0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    float32x4_t _sum = vdupq_n_f32(0.0f);
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                    {
                        float32x4_t _src = vld1q_f32(src + c);
                        _sum = vaddq_f32(_sum, vmulq_f32(_src, _src));
                    }
                    float sum = ExtractSum32f(_sum);
                    for (; c < channels; ++c)
                        sum += Simd::Square(src[c]);
                    float32x4_t _k = vdupq_n_f32(1.0f / ::sqrt(sum + eps));
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(dst + c, vmulq_f32(vmulq_f32(vld1q_f32(src + c), vld1q_f32(scale + c)), _k));
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] * vgetq_lane_f32(_k, 0);
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
            float32x4_t _eps = vdupq_n_f32(eps), _k = vdupq_n_f32(k), _1 = vdupq_n_f32(1.0f);
            for (size_t b = 0, s; b < batch; ++b)
            {
                for (s = 0; s < spatialF; s += F)
                    vst1q_f32(buf + s, vdupq_n_f32(0.0f));
                for (; s < spatial; ++s)
                    buf[s] = 0.0f;
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t _src = vld1q_f32(ps + s);
                        float32x4_t _sum = vld1q_f32(buf + s);
                        vst1q_f32(buf + s, vaddq_f32(_sum, _src));
                    }
                    for (; s < spatial; ++s)
                        buf[s] += ps[s];
                }
                for (s = 0; s < spatialF; s += F)
                    vst1q_f32(buf + s, vmulq_f32(vld1q_f32(buf + s), _k));
                for (; s < spatial; ++s)
                    buf[s] *= k;
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t _src = vld1q_f32(ps + s);
                        float32x4_t mean = vld1q_f32(buf + s);
                        vst1q_f32(pd + s, vsubq_f32(_src, mean));
                    }
                    for (; s < spatial; ++s)
                        pd[s] = ps[s] - buf[s];
                }

                for (s = 0; s < spatialF; s += F)
                    vst1q_f32(buf + s, vdupq_n_f32(0.0f));
                for (; s < spatial; ++s)
                    buf[s] = 0.0f;
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t _dst = vld1q_f32(pd + s);
                        float32x4_t _sum = vld1q_f32(buf + s);
                        vst1q_f32(buf + s, vaddq_f32(_sum, vmulq_f32(_dst, _dst)));
                    }
                    for (; s < spatial; ++s)
                        buf[s] += Simd::Square(pd[s]);
                }
                for (s = 0; s < spatialF; s += F)
                    vst1q_f32(buf + s, ReciprocalSqrt<1>(vaddq_f32(vmulq_f32(vld1q_f32(buf + s), _k), _eps)));
                for (; s < spatial; ++s)
                    buf[s] = 1.0f / ::sqrt(buf[s] * k + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    float* pd = dst + c * spatial;
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t _dst = vld1q_f32(pd + s);
                        float32x4_t norm = vld1q_f32(buf + s);
                        vst1q_f32(pd + s, vaddq_f32(vmulq_f32(vmulq_f32(_dst, norm), _scale), _shift));
                    }
                    for (; s < spatial; ++s)
                        pd[s] = pd[s] * buf[s] * scale[c] + shift[c];
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }

        void NormalizeNhwcV2(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    float32x4_t _sum = vdupq_n_f32(0.0f);
                    for (c = 0; c < channelsF; c += F)
                        _sum = vaddq_f32(vld1q_f32(src + c), _sum);
                    float sum = ExtractSum32f(_sum);
                    for (; c < channels; ++c)
                        sum += src[c];
                    float32x4_t mean = vdupq_n_f32(sum * k);
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(dst + c, vsubq_f32(vld1q_f32(src + c), mean));
                    for (; c < channels; ++c)
                        dst[c] = src[c] - sum * k;

                    float32x4_t _sqsum = vdupq_n_f32(0.0f);
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t d = vld1q_f32(dst + c);
                        _sqsum = vaddq_f32(vmulq_f32(d, d), _sqsum);
                    }
                    float sqsum = ExtractSum32f(_sqsum);
                    for (; c < channels; ++c)
                        sqsum += Simd::Square(dst[c]);
                    float32x4_t norm = vdupq_n_f32(1.0f / ::sqrt(sqsum * k + eps));
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(dst + c, vaddq_f32(vmulq_f32(vmulq_f32(vld1q_f32(dst + c), norm), vld1q_f32(scale + c)), vld1q_f32(shift + c)));
                    for (; c < channels; ++c)
                        dst[c] = dst[c] * vgetq_lane_f32(norm, 0) * scale[c] + shift[c];

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
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _sum = vdupq_n_f32(0.0f);
                    for (s = 0; s < spatialF; s += F)
                        _sum = vaddq_f32(vld1q_f32(src + s), _sum);
                    float sum = ExtractSum32f(_sum);
                    for (; s < spatial; ++s)
                        sum += src[s];
                    float32x4_t mean = vdupq_n_f32(sum * k);
                    for (s = 0; s < spatialF; s += F)
                        vst1q_f32(dst + s, vsubq_f32(vld1q_f32(src + s), mean));
                    for (; s < spatial; ++s)
                        dst[s] = src[s] - sum * k;

                    float32x4_t _sqsum = vdupq_n_f32(0.0f);
                    for (s = 0; s < spatialF; s += F)
                    {
                        float32x4_t d = vld1q_f32(dst + s);
                        _sqsum = vaddq_f32(vmulq_f32(d, d), _sqsum);
                    }
                    float sqsum = ExtractSum32f(_sqsum);
                    for (; s < spatial; ++s)
                        sqsum += Simd::Square(dst[s]);
                    float32x4_t norm = vdupq_n_f32(1.0f / ::sqrt(sqsum * k + eps));
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    for (s = 0; s < spatialF; s += F)
                        vst1q_f32(dst + s, vaddq_f32(vmulq_f32(vmulq_f32(vld1q_f32(dst + s), norm), _scale), _shift));
                    for (; s < spatial; ++s)
                        dst[s] = dst[s] * vgetq_lane_f32(norm, 0) * scale[c] + shift[c];

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
                _buf.Resize(channels);
                buf = _buf.data;
            }
            size_t channelsF = AlignLo(channels, F);
            float32x4_t _eps = vdupq_n_f32(eps), _k = vdupq_n_f32(k), _1 = vdupq_n_f32(1.0f);
            for (size_t b = 0, c; b < batch; ++b)
            {
                for (c = 0; c < channelsF; c += F)
                    vst1q_f32(buf + c, vdupq_n_f32(0.0f));
                for (; c < channels; ++c)
                    buf[c] = 0.0f;
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _src = vld1q_f32(ps + c);
                        float32x4_t _sum = vld1q_f32(buf + c);
                        vst1q_f32(buf + c, vaddq_f32(_sum, _src));
                    }
                    for (; c < channels; ++c)
                        buf[c] += ps[c];
                }
                for (c = 0; c < channelsF; c += F)
                    vst1q_f32(buf + c, vmulq_f32(vld1q_f32(buf + c), _k));
                for (; c < channels; ++c)
                    buf[c] *= k;
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* ps = src + s * channels;
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _src = vld1q_f32(ps + c);
                        float32x4_t mean = vld1q_f32(buf + c);
                        vst1q_f32(pd + c, vsubq_f32(_src, mean));
                    }
                    for (; c < channels; ++c)
                        pd[c] = ps[c] - buf[c];
                }

                for (c = 0; c < channelsF; c += F)
                    vst1q_f32(buf + c, vdupq_n_f32(0.0f));
                for (; c < channels; ++c)
                    buf[c] = 0.0f;
                for (size_t s = 0; s < spatial; ++s)
                {
                    const float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _dst = vld1q_f32(pd + c);
                        float32x4_t _sum = vld1q_f32(buf + c);
                        vst1q_f32(buf + c, vaddq_f32(_sum, vmulq_f32(_dst, _dst)));
                    }
                    for (; c < channels; ++c)
                        buf[c] += Simd::Square(pd[c]);
                }
                for (c = 0; c < channelsF; c += F)
                    vst1q_f32(buf + c, ReciprocalSqrt<1>(vaddq_f32(vmulq_f32(vld1q_f32(buf + c), _k), _eps)));
                for (; c < channels; ++c)
                    buf[c] = 1.0f / ::sqrt(buf[c] * k + eps);
                for (size_t s = 0; s < spatial; ++s)
                {
                    float* pd = dst + s * channels;
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _dst = vld1q_f32(pd + c);
                        float32x4_t norm = vld1q_f32(buf + c);
                        vst1q_f32(pd + c, vaddq_f32(vmulq_f32(vmulq_f32(_dst, norm), vld1q_f32(scale + c)), vld1q_f32(shift + c)));
                    }
                    for (; c < channels; ++c)
                        pd[c] = pd[c] * buf[c] * scale[c] + shift[c];
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

        //-------------------------------------------------------------------------------------------------

        void NormalizeNchwV4(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t spatialF = AlignLo(spatial, F), s;
            for (size_t b = 0; b < batch; ++b)
            {
                float sum = 0;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    float32x4_t _sqsum = vdupq_n_f32(0.0f);
                    for (s = 0; s < spatialF; s += F, o += F)
                    {
                        float32x4_t _src = vld1q_f32(src + o);
                        _sqsum = vaddq_f32(vmulq_f32(_src, _src), _sqsum);
                    }
                    float sqsum = ExtractSum32f(_sqsum);
                    for (; s < spatial; ++s, ++o)
                        sqsum += Simd::Square(src[o]);
                    buf[c] = ::sqrt(sqsum);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _alpha = vdupq_n_f32(1.0f + scale[c] * buf[c] * norm);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    for (s = 0; s < spatialF; s += F)
                        vst1q_f32(dst + s, vaddq_f32(vmulq_f32(vld1q_f32(src + s), _alpha), _shift));
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * vgetq_lane_f32(_alpha, 0) + shift[c];
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwcV4(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, float* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            for (size_t b = 0; b < batch; ++b)
            {
                for (c = 0; c < channelsF; c += F)
                    vst1q_f32(buf + c, vdupq_n_f32(0.0f));
                for (; c < channels; ++c)
                    buf[c] = 0.0f;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (c = 0; c < channelsF; c += F, o += F)
                    {
                        float32x4_t _src = vld1q_f32(src + o);
                        vst1q_f32(buf + c, vaddq_f32(vmulq_f32(_src, _src), vld1q_f32(buf + c)));
                    }
                    for (; c < channels; c += 1, o += 1)
                        buf[c] += Simd::Square(src[o]);
                }
                float sum = 0;
                for (size_t c = 0; c < channels; ++c)
                {
                    buf[c] = ::sqrt(buf[c]);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 1.0f + scale[c] * buf[c] * norm;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(dst + c, vaddq_f32(vmulq_f32(vld1q_f32(src + c), vld1q_f32(buf + c)), vld1q_f32(shift + c)));
                    for (; c < channels; c += 1, o += 1)
                        dst[c] = src[c] * buf[c] + shift[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        void SynetNormalizeLayerForwardV4(const float* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst)
        {
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(channels);
                buf = _buf.data;
            }
            if (format == SimdTensorFormatNchw)
                NormalizeNchwV4(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else if (format == SimdTensorFormatNhwc)
                NormalizeNhwcV4(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
