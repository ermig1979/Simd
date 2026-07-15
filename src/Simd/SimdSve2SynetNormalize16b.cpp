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
        SIMD_INLINE float Sum16bV2(const float* src, size_t size)
        {
            const size_t F = svcntw(), sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            svfloat32_t sum = svdup_n_f32(0.0f);
            size_t i = 0;
            for (; i < sizeF; i += F)
                sum = svadd_f32_x(body, sum, svld1_f32(body, src + i));
            float result = svaddv_f32(body, sum);
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                result += svaddv_f32(tail, svld1_f32(tail, src + i));
            }
            return result;
        }

        SIMD_INLINE float SumSquares16bV2(const float* src, size_t size)
        {
            const size_t F = svcntw(), sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            svfloat32_t sum = svdup_n_f32(0.0f);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                svfloat32_t value = svld1_f32(body, src + i);
                sum = svmla_f32_x(body, sum, value, value);
            }
            float result = svaddv_f32(body, sum);
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t value = svld1_f32(tail, src + i);
                result += svaddv_f32(tail, svmul_f32_x(tail, value, value));
            }
            return result;
        }

        void NormalizeNhwc16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, uint16_t* dst)
        {
            float k = 1.0f / float(channels);
            const size_t F = svcntw();
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(channels);
                buf = _buf.data;
            }
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    BFloat16ToFloat32(src, channels, buf);

                    svfloat32_t mean = svdup_n_f32(Sum16bV2(buf, channels) * k);
                    for (size_t c = 0; c < channels; c += F)
                    {
                        svbool_t mask = svwhilelt_b32(c, channels);
                        svst1_f32(mask, buf + c, svsub_f32_x(mask, svld1_f32(mask, buf + c), mean));
                    }

                    svfloat32_t norm = svdup_n_f32(1.0f / ::sqrt(SumSquares16bV2(buf, channels) * k + eps));
                    for (size_t c = 0; c < channels; c += F)
                    {
                        svbool_t mask = svwhilelt_b32(c, channels);
                        svfloat32_t value = svmul_f32_x(mask, svld1_f32(mask, buf + c), norm);
                        svst1_f32(mask, buf + c, svmla_f32_x(mask, svld1_f32(mask, shift + c), value, svld1_f32(mask, scale + c)));
                    }

                    Float32ToBFloat16(buf, channels, dst);

                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForward16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, uint16_t* dst)
        {
            if (format == SimdTensorFormatNhwc)
                NormalizeNhwc16bV2(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
