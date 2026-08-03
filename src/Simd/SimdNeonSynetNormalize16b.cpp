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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE float32x4_t LoadBFloat16(const uint16_t* src)
        {
            return BFloat16ToFloat32(vmovl_u16(vld1_u16(src)));
        }

        SIMD_INLINE void StoreBFloat16(float32x4_t value, uint16_t* dst)
        {
            vst1_u16(dst, vmovn_u32(Float32ToBFloat16(value)));
        }

        void NormalizeNhwc16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, uint16_t* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
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
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(buf + c, LoadBFloat16(src + c));
                    for (; c < channels; ++c)
                        buf[c] = Base::BFloat16ToFloat32(src[c]);

                    float32x4_t _sum = vdupq_n_f32(0.0f);
                    for (c = 0; c < channelsF; c += F)
                        _sum = vaddq_f32(vld1q_f32(buf + c), _sum);
                    float sum = ExtractSum32f(_sum);
                    for (; c < channels; ++c)
                        sum += buf[c];
                    float32x4_t mean = vdupq_n_f32(sum * k);
                    for (c = 0; c < channelsF; c += F)
                        vst1q_f32(buf + c, vsubq_f32(vld1q_f32(buf + c), mean));
                    for (; c < channels; ++c)
                        buf[c] -= sum * k;

                    float32x4_t _sqsum = vdupq_n_f32(0.0f);
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _buf = vld1q_f32(buf + c);
                        _sqsum = vaddq_f32(vmulq_f32(_buf, _buf), _sqsum);
                    }
                    float sqsum = ExtractSum32f(_sqsum);
                    for (; c < channels; ++c)
                        sqsum += Simd::Square(buf[c]);
                    float32x4_t norm = vdupq_n_f32(1.0f / ::sqrt(sqsum * k + eps));
                    for (c = 0; c < channelsF; c += F)
                    {
                        float32x4_t _buf = vmulq_f32(vld1q_f32(buf + c), norm);
                        vst1q_f32(buf + c, vaddq_f32(vmulq_f32(_buf, vld1q_f32(scale + c)), vld1q_f32(shift + c)));
                    }
                    for (; c < channels; ++c)
                        buf[c] = buf[c] * vgetq_lane_f32(norm, 0) * scale[c] + shift[c];

                    for (c = 0; c < channelsF; c += F)
                        StoreBFloat16(vld1q_f32(buf + c), dst + c);
                    for (; c < channels; ++c)
                        dst[c] = Base::Float32ToBFloat16(buf[c]);

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
