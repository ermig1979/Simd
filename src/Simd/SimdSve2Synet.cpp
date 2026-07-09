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
#include "Simd/SimdMemory.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t BFloat16ToFloat32(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, svld1uh_u32(mask, src), Base::Bf16::SHIFT));
        }

        SIMD_INLINE void AddBFloat16ToSum(const uint16_t* src, svfloat32_t& sum, const svbool_t& mask)
        {
            sum = svadd_f32_m(mask, sum, BFloat16ToFloat32(src, mask));
        }

        void SynetChannelSum16b(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();

            if (format == SimdTensorFormatNhwc)
            {
                size_t c = 0;
                for (; c + QF <= channels; c += QF)
                {
                    svst1_f32(body, sum + c + 0 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 1 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 2 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 3 * F, svdup_n_f32(0.0f));
                }
                for (; c + F <= channels; c += F)
                    svst1_f32(body, sum + c, svdup_n_f32(0.0f));
                if (c < channels)
                    svst1_f32(svwhilelt_b32(c, channels), sum + c, svdup_n_f32(0.0f));

                for (size_t s = 0; s < spatial; ++s)
                {
                    c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        svfloat32_t sum0 = svld1_f32(body, sum + c + 0 * F);
                        svfloat32_t sum1 = svld1_f32(body, sum + c + 1 * F);
                        svfloat32_t sum2 = svld1_f32(body, sum + c + 2 * F);
                        svfloat32_t sum3 = svld1_f32(body, sum + c + 3 * F);
                        AddBFloat16ToSum(src + c + 0 * F, sum0, body);
                        AddBFloat16ToSum(src + c + 1 * F, sum1, body);
                        AddBFloat16ToSum(src + c + 2 * F, sum2, body);
                        AddBFloat16ToSum(src + c + 3 * F, sum3, body);
                        svst1_f32(body, sum + c + 0 * F, sum0);
                        svst1_f32(body, sum + c + 1 * F, sum1);
                        svst1_f32(body, sum + c + 2 * F, sum2);
                        svst1_f32(body, sum + c + 3 * F, sum3);
                    }
                    for (; c + F <= channels; c += F)
                    {
                        svfloat32_t _sum = svld1_f32(body, sum + c);
                        AddBFloat16ToSum(src + c, _sum, body);
                        svst1_f32(body, sum + c, _sum);
                    }
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        svfloat32_t _sum = svld1_f32(tail, sum + c);
                        AddBFloat16ToSum(src + c, _sum, tail);
                        svst1_f32(tail, sum + c, _sum);
                    }
                    src += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                    svfloat32_t sum2 = svdup_n_f32(0.0f), sum3 = svdup_n_f32(0.0f);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        AddBFloat16ToSum(src + s + 0 * F, sum0, body);
                        AddBFloat16ToSum(src + s + 1 * F, sum1, body);
                        AddBFloat16ToSum(src + s + 2 * F, sum2, body);
                        AddBFloat16ToSum(src + s + 3 * F, sum3, body);
                    }
                    sum0 = svadd_f32_x(body, svadd_f32_x(body, sum0, sum1), svadd_f32_x(body, sum2, sum3));
                    for (; s + F <= spatial; s += F)
                        AddBFloat16ToSum(src + s, sum0, body);
                    if (s < spatial)
                        AddBFloat16ToSum(src + s, sum0, svwhilelt_b32(s, spatial));
                    sum[c] = svaddv_f32(body, sum0);
                    src += spatial;
                }
            }
            else
                assert(0);
        }
    }
#endif
}
