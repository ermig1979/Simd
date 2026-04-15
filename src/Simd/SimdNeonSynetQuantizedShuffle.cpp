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

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cd = 0, spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            int32x4_t _bias0 = vdupq_n_s32(bias0), _bias1 = vdupq_n_s32(bias1), _zero = vdupq_n_s32(zero);
            float32x4_t _norm0 = vdupq_n_f32(norm0), _norm1 = vdupq_n_f32(norm1), _scale = vdupq_n_f32(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_1(const uint8_t* src, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst0, uint8_t* dst1)
        {
            uint8x8_t s0 = vdup_n_u8(src[0]);
            uint8x8_t s1 = vdup_n_u8(src[1]);
            uint8x8_t s01 = vzip_u8(s0, s1).val[0];
            int32x4_t i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(s01))));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(i32, bias, norm), scale, zero);
            uint8x8_t res = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vdup_n_s16(0)));
            dst0[0] = vget_lane_u8(res, 0);
            dst1[0] = vget_lane_u8(res, 1);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_4(const uint8_t* src, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst0, uint8_t* dst1)
        {
            uint8x8_t u8src = vld1_u8(src);
            uint16x8_t u16 = vmovl_u8(u8src);
            int32x4_t i0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16)));
            int32x4_t i1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u16)));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(i0, bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(i1, bias, norm), scale, zero);
            uint8x8_t res = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1)));
            uint8x8x2_t deint = vuzp_u8(res, vdup_n_u8(0));
            vst1_lane_u32((uint32_t*)dst0, vreinterpret_u32_u8(deint.val[0]), 0);
            vst1_lane_u32((uint32_t*)dst1, vreinterpret_u32_u8(deint.val[1]), 0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_8(const uint8_t* src, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst0, uint8_t* dst1)
        {
            uint8x16_t s8 = vld1q_u8(src);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s8)), s16hi = vmovl_u8(vget_high_u8(s8));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            int32x4_t d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            uint8x8_t lo = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1)));
            uint8x8_t hi = vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)));
            uint8x8x2_t deint = vuzp_u8(lo, hi);
            vst1_u8(dst0, deint.val[0]);
            vst1_u8(dst1, deint.val[1]);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc0_16(const uint8_t* src, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst0, uint8_t* dst1)
        {
            uint8x16_t s8a = vld1q_u8(src + 0);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s8a)), s16hi = vmovl_u8(vget_high_u8(s8a));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            int32x4_t d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            uint8x8_t lo_a = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1)));
            uint8x8_t hi_a = vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)));
            uint8x16_t s8b = vld1q_u8(src + 16);
            s16lo = vmovl_u8(vget_low_u8(s8b)); s16hi = vmovl_u8(vget_high_u8(s8b));
            d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            uint8x8_t lo_b = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1)));
            uint8x8_t hi_b = vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)));
            uint8x8x2_t deint_a = vuzp_u8(lo_a, hi_a);
            uint8x8x2_t deint_b = vuzp_u8(lo_b, hi_b);
            vst1q_u8(dst0, vcombine_u8(deint_a.val[0], deint_b.val[0]));
            vst1q_u8(dst1, vcombine_u8(deint_a.val[1], deint_b.val[1]));
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs, cd;
            size_t srcC0_8 = AlignLo(srcC0, 8), srcC1_8 = AlignLo(srcC1, 8);
            size_t srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16);
            size_t srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            int32x4_t _bias0 = vdupq_n_s32(bias0), _bias1 = vdupq_n_s32(bias1), _zero = vdupq_n_s32(zero);
            float32x4_t _norm0 = vdupq_n_f32(norm0), _norm1 = vdupq_n_f32(norm1), _scale = vdupq_n_f32(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                cd = 0; cs = 0;
                for (; cs < srcC0_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0_8; cs += 8, cd += 4)
                    DequantizeQuantizeLinearNhwc0_4(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC0; cs += 2, cd += 1)
                    DequantizeQuantizeLinearNhwc0_1(src0 + cs, _bias0, _norm0, _scale, _zero, dst0 + cd, dst1 + cd);
                cs = 0;
                for (; cs < srcC1_32; cs += 32, cd += 16)
                    DequantizeQuantizeLinearNhwc0_16(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_16; cs += 16, cd += 8)
                    DequantizeQuantizeLinearNhwc0_8(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1_8; cs += 8, cd += 4)
                    DequantizeQuantizeLinearNhwc0_4(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                for (; cs < srcC1; cs += 2, cd += 1)
                    DequantizeQuantizeLinearNhwc0_1(src1 + cs, _bias1, _norm1, _scale, _zero, dst0 + cd, dst1 + cd);
                src0 += srcC0;
                src1 += srcC1;
                dst0 += dstC;
                dst1 += dstC;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForwardNchw1(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs = 0, spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            int32x4_t _bias0 = vdupq_n_s32(bias0), _bias1 = vdupq_n_s32(bias1), _zero = vdupq_n_s32(zero);
            float32x4_t _norm0 = vdupq_n_f32(norm0), _norm1 = vdupq_n_f32(norm1), _scale = vdupq_n_f32(scale);
            for (size_t cd = 0; cd < srcC0; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_1(const uint8_t* src0, const uint8_t* src1, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst)
        {
            uint8x8_t s0 = vdup_n_u8(src0[0]);
            uint8x8_t s1 = vdup_n_u8(src1[0]);
            uint8x8_t s01 = vzip_u8(s0, s1).val[0];
            int32x4_t i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(s01))));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(i32, bias, norm), scale, zero);
            uint8x8_t res = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vdup_n_s16(0)));
            ((uint16_t*)dst)[0] = vget_lane_u16(vreinterpret_u16_u8(res), 0);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_4(const uint8_t* src0, const uint8_t* src1, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst)
        {
            uint8x8_t _src0 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src0));
            uint8x8_t _src1 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src1));
            uint8x8_t s01 = vzip_u8(_src0, _src1).val[0];
            uint16x8_t u16 = vmovl_u8(s01);
            int32x4_t i0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16)));
            int32x4_t i1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u16)));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(i0, bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(i1, bias, norm), scale, zero);
            uint8x8_t res = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1)));
            vst1_u8(dst, res);
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_8(const uint8_t* src0, const uint8_t* src1, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);
            uint8x8x2_t zipped = vzip_u8(_src0, _src1);
            uint8x16_t s01 = vcombine_u8(zipped.val[0], zipped.val[1]);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s01)), s16hi = vmovl_u8(vget_high_u8(s01));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            int32x4_t d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE void DequantizeQuantizeLinearNhwc1_16(const uint8_t* src0, const uint8_t* src1, int32x4_t bias, float32x4_t norm, float32x4_t scale, int32x4_t zero, uint8_t* dst)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x8x2_t zip_lo = vzip_u8(vget_low_u8(_src0), vget_low_u8(_src1));
            uint8x16_t s_lo = vcombine_u8(zip_lo.val[0], zip_lo.val[1]);
            uint16x8_t s16lo = vmovl_u8(vget_low_u8(s_lo)), s16hi = vmovl_u8(vget_high_u8(s_lo));
            int32x4_t d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            int32x4_t d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            int32x4_t d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            vst1q_u8(dst + 0, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
            uint8x8x2_t zip_hi = vzip_u8(vget_high_u8(_src0), vget_high_u8(_src1));
            uint8x16_t s_hi = vcombine_u8(zip_hi.val[0], zip_hi.val[1]);
            s16lo = vmovl_u8(vget_low_u8(s_hi)); s16hi = vmovl_u8(vget_high_u8(s_hi));
            d0 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16lo))), bias, norm), scale, zero);
            d1 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16lo))), bias, norm), scale, zero);
            d2 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16hi))), bias, norm), scale, zero);
            d3 = QuantizeLinear(DequantizeLinear(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16hi))), bias, norm), scale, zero);
            vst1q_u8(dst + 16, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0,
            const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2;
            size_t srcC0_8 = AlignLo(srcC0, 8), srcC1_8 = AlignLo(srcC1, 8);
            size_t srcC0_16 = AlignLo(srcC0, 16), srcC1_16 = AlignLo(srcC1, 16);
            size_t srcC0_32 = AlignLo(srcC0, 32), srcC1_32 = AlignLo(srcC1, 32);
            int32x4_t _bias01 = vzipq_s32(vdupq_n_s32(bias0), vdupq_n_s32(bias1)).val[0];
            float32x4_t _norm01 = vzipq_f32(vdupq_n_f32(norm0), vdupq_n_f32(norm1)).val[0];
            int32x4_t _zero = vdupq_n_s32(zero);
            float32x4_t _scale = vdupq_n_f32(scale);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cs = 0, cd;
                for (cd = 0; cd < srcC0_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0_8; cd += 8, cs += 4)
                    DequantizeQuantizeLinearNhwc1_4(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (; cd < srcC0; cd += 2, cs += 1)
                    DequantizeQuantizeLinearNhwc1_1(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst0 + cd);
                for (cd = 0; cd < srcC1_32; cd += 32, cs += 16)
                    DequantizeQuantizeLinearNhwc1_16(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_16; cd += 16, cs += 8)
                    DequantizeQuantizeLinearNhwc1_8(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1_8; cd += 8, cs += 4)
                    DequantizeQuantizeLinearNhwc1_4(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                for (; cd < srcC1; cd += 2, cs += 1)
                    DequantizeQuantizeLinearNhwc1_1(src0 + cs, src1 + cs, _bias01, _norm01, _scale, _zero, dst1 + cd);
                src0 += dstC;
                src1 += dstC;
                dst0 += srcC0;
                dst1 += srcC1;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizedShuffleLayerForward(const uint8_t* src0, int bias0, const float* norm0, size_t srcC0, const uint8_t* src1, int bias1, const float* norm1, size_t srcC1,
            size_t spatial, uint8_t* dst0, uint8_t* dst1, const float* scale, int zero, SimdTensorFormatType format, int shuffleType)
        {
            switch (shuffleType)
            {
            case 0:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            case 1:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            }
        }
    }
#endif
}
