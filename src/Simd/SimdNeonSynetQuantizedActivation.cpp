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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE int32x4_t QuantizedPrelu(int32x4_t src, int32x4_t sBias, float32x4_t sNorm, float32x4_t slope, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _src = DequantizeLinear(src, sBias, sNorm);
            float32x4_t _zero = vdupq_n_f32(0.0f);
            float32x4_t pos = vmaxq_f32(_zero, _src);
            float32x4_t neg = vminq_f32(_zero, _src);
            float32x4_t _dst = vmlaq_f32(pos, slope, neg);
            return QuantizeLinear(_dst, dNorm, dZero);
        }

        SIMD_INLINE void QuantizedPrelu1(const uint8_t* src, int32x4_t sBias, float32x4_t sNorm, float32x4_t slope, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            int32x4_t _src = vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)src[0]));
            int32x4_t d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vdup_n_s16(0)));
            dst[0] = vget_lane_u8(u8, 0);
        }

        SIMD_INLINE void QuantizedPrelu4(const uint8_t* src, int32x4_t sBias, float32x4_t sNorm, float32x4_t slope, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x8_t u8src = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src));
            int32x4_t _src = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(u8src))));
            int32x4_t d0 = QuantizedPrelu(_src, sBias, sNorm, slope, dNorm, dZero);
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vdup_n_s16(0)));
            vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE void QuantizedPrelu16(const uint8_t* src, int32x4_t sBias, float32x4_t sNorm, float32x4_t slope, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x16_t s8 = vld1q_u8(src);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s8)), s16hi = vmovl_u8(vget_high_u8(s8));
            int32x4_t d0 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), sBias, sNorm, slope, dNorm, dZero);
            int32x4_t d1 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), sBias, sNorm, slope, dNorm, dZero);
            int32x4_t d2 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), sBias, sNorm, slope, dNorm, dZero);
            int32x4_t d3 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), sBias, sNorm, slope, dNorm, dZero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE void QuantizedPrelu16(const uint8_t* src, int32x4_t sBias, float32x4_t sNorm, const float* slope, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x16_t s8 = vld1q_u8(src);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s8)), s16hi = vmovl_u8(vget_high_u8(s8));
            int32x4_t d0 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), sBias, sNorm, vld1q_f32(slope + 0 * 4), dNorm, dZero);
            int32x4_t d1 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), sBias, sNorm, vld1q_f32(slope + 1 * 4), dNorm, dZero);
            int32x4_t d2 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), sBias, sNorm, vld1q_f32(slope + 2 * 4), dNorm, dZero);
            int32x4_t d3 = QuantizedPrelu(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), sBias, sNorm, vld1q_f32(slope + 3 * 4), dNorm, dZero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        void SynetQuantizedPreluLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            int32x4_t sBias = vdupq_n_s32(-srcZero), dZero = vdupq_n_s32(dstZero);
            float32x4_t sNorm = vdupq_n_f32(srcScale[0]), dNorm = vdupq_n_f32(1.0f / dstScale[0]);
            if (format == SimdTensorFormatNhwc)
            {
                size_t channels4 = AlignLo(channels, 4), channels16 = AlignLo(channels, 16);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels16; c += 16)
                        QuantizedPrelu16(src + c, sBias, sNorm, slope + c, dst + c, dNorm, dZero);
                    for (; c < channels4; c += 4)
                        QuantizedPrelu4(src + c, sBias, sNorm, vld1q_f32(slope + c), dst + c, dNorm, dZero);
                    for (; c < channels; ++c)
                        QuantizedPrelu1(src + c, sBias, sNorm, vdupq_n_f32(slope[c]), dst + c, dNorm, dZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                size_t spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _slope = vdupq_n_f32(slope[c]);
                    size_t s = 0;
                    for (; s < spatial16; s += 16)
                        QuantizedPrelu16(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    for (; s < spatial4; s += 4)
                        QuantizedPrelu4(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    for (; s < spatial; ++s)
                        QuantizedPrelu1(src + s, sBias, sNorm, _slope, dst + s, dNorm, dZero);
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
