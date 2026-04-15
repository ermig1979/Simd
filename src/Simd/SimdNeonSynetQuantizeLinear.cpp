/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE void DequantizeLinear16(const uint8_t* src, int32x4_t bias, float32x4_t norm, float* dst)
        {
            uint8x16_t u8 = vld1q_u8(src);
            uint16x8_t u16lo = vmovl_u8(vget_low_u8(u8));
            uint16x8_t u16hi = vmovl_u8(vget_high_u8(u8));
            int32x4_t i0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16lo)));
            int32x4_t i1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u16lo)));
            int32x4_t i2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16hi)));
            int32x4_t i3 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u16hi)));
            vst1q_f32(dst + 0 * F, DequantizeLinear(i0, bias, norm));
            vst1q_f32(dst + 1 * F, DequantizeLinear(i1, bias, norm));
            vst1q_f32(dst + 2 * F, DequantizeLinear(i2, bias, norm));
            vst1q_f32(dst + 3 * F, DequantizeLinear(i3, bias, norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            int32x4_t _bias = vdupq_n_s32(bias);
            float32x4_t _norm = vdupq_n_f32(norm[0]);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                DequantizeLinear16(src + i, _bias, _norm, dst + i);
            for (; i < size4; i += 4)
                DequantizeLinear4(src + i, _bias, _norm, dst + i);
            for (; i < size; i += 1)
                DequantizeLinear1(src + i, _bias, _norm, dst + i);
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizeLinear(const float* src, size_t size, const float* norm, int32_t zero, uint8_t* dst)
        {
            float32x4_t _norm = vdupq_n_f32(norm[0]);
            int32x4_t _zero = vdupq_n_s32(zero);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                QuantizeLinear16(src + i, _norm, _zero, dst + i);
            for (; i < size4; i += 4)
                QuantizeLinear4(src + i, _norm, _zero, dst + i);
            for (; i < size; i += 1)
                QuantizeLinear1(src + i, _norm, _zero, dst + i);
        }
    }
#endif
}
