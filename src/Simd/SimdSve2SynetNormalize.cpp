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
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE float SumSquares(const float* src, size_t size)
        {
            const size_t F = svcntw(), sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            svfloat32_t sum = svdup_n_f32(0.0f);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                svfloat32_t _src = svld1_f32(body, src + i);
                sum = svmla_f32_x(body, sum, _src, _src);
            }
            float result = svaddv_f32(body, sum);
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t _src = svld1_f32(tail, src + i);
                result += svaddv_f32(tail, svmul_f32_x(tail, _src, _src));
            }
            return result;
        }

        SIMD_INLINE svfloat32_t ReciprocalSqrt(const svfloat32_t& value, const svbool_t& mask)
        {
            return svdiv_f32_x(mask, svdup_n_f32(1.0f), svsqrt_f32_x(mask, value));
        }

        void NormalizeNchw1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            const size_t F = svcntw(), size = channels * spatial;
            for (size_t b = 0; b < batch; ++b)
            {
                float k0 = 1.0f / ::sqrt(SumSquares(src, size) + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _k = svdup_n_f32(scale[c] * k0);
                    for (size_t s = 0; s < spatial; s += F)
                    {
                        svbool_t mask = svwhilelt_b32(s, spatial);
                        svst1_f32(mask, dst + s, svmul_f32_x(mask, svld1_f32(mask, src + s), _k));
                    }
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
            const size_t F = svcntw();
            svfloat32_t _eps = svdup_n_f32(eps);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; s += F)
                {
                    svbool_t mask = svwhilelt_b32(s, spatial);
                    svst1_f32(mask, buf + s, _eps);
                }
                for (size_t c = 0; c < channels; ++c)
                {
                    const float* ps = src + c * spatial;
                    for (size_t s = 0; s < spatial; s += F)
                    {
                        svbool_t mask = svwhilelt_b32(s, spatial);
                        svfloat32_t _src = svld1_f32(mask, ps + s);
                        svst1_f32(mask, buf + s, svmla_f32_x(mask, svld1_f32(mask, buf + s), _src, _src));
                    }
                }
                for (size_t s = 0; s < spatial; s += F)
                {
                    svbool_t mask = svwhilelt_b32(s, spatial);
                    svst1_f32(mask, buf + s, ReciprocalSqrt(svld1_f32(mask, buf + s), mask));
                }
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _scale = svdup_n_f32(scale[c]);
                    for (size_t s = 0; s < spatial; s += F)
                    {
                        svbool_t mask = svwhilelt_b32(s, spatial);
                        svst1_f32(mask, dst + s, svmul_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src + s), svld1_f32(mask, buf + s)), _scale));
                    }
                    dst += spatial;
                    src += spatial;
                }
            }
        }

        void NormalizeNhwc1(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            const size_t F = svcntw(), size = channels * spatial;
            for (size_t b = 0; b < batch; ++b)
            {
                svfloat32_t _k = svdup_n_f32(1.0f / ::sqrt(SumSquares(src, size) + eps));
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += F)
                    {
                        svbool_t mask = svwhilelt_b32(c, channels);
                        svst1_f32(mask, dst + c, svmul_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src + c), svld1_f32(mask, scale + c)), _k));
                    }
                    dst += channels;
                    src += channels;
                }
            }
        }

        void NormalizeNhwc0(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, float* dst)
        {
            const size_t F = svcntw();
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    svfloat32_t _k = svdup_n_f32(1.0f / ::sqrt(SumSquares(src, channels) + eps));
                    for (size_t c = 0; c < channels; c += F)
                    {
                        svbool_t mask = svwhilelt_b32(c, channels);
                        svst1_f32(mask, dst + c, svmul_f32_x(mask, svmul_f32_x(mask, svld1_f32(mask, src + c), svld1_f32(mask, scale + c)), _k));
                    }
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
    }
#endif
}
