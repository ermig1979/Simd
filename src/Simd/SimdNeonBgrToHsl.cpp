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
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE int32x4_t MulDiv32(const int32x4_t & dividend, const int32x4_t & divisor, const float32x4_t & KF)
        {
            return vcvtq_s32_f32(Div<SIMD_NEON_RCP_ITER>(vmulq_f32(KF, vcvtq_f32_s32(dividend)), vcvtq_f32_s32(divisor)));
        }

        SIMD_INLINE int16x8_t MulDiv16(const int16x8_t & dividend, const int16x8_t & divisor, const float32x4_t & KF)
        {
            int32x4_t lo = MulDiv32(UnpackI16<0>(dividend), UnpackI16<0>(divisor), KF);
            int32x4_t hi = MulDiv32(UnpackI16<1>(dividend), UnpackI16<1>(divisor), KF);
            return PackI32(lo, hi);
        }

        SIMD_INLINE void BgrToHsl16(int16x8_t blue, int16x8_t green, int16x8_t red,
            int16x8_t & hue, int16x8_t & sat, int16x8_t & lgt,
            const float32x4_t & KF_255_DIV_6, const float32x4_t & K_255F)
        {
            int16x8_t max = vmaxq_s16(red, vmaxq_s16(green, blue));
            int16x8_t min = vminq_s16(red, vminq_s16(green, blue));
            int16x8_t range = vsubq_s16(max, min);
            int16x8_t sum = vaddq_s16(max, min);

            int16x8_t redMaxMask = (int16x8_t)vceqq_s16(red, max);
            int16x8_t greenMaxMask = vandq_s16(vmvnq_s16(redMaxMask), (int16x8_t)vceqq_s16(green, max));
            int16x8_t blueMaxMask = vandq_s16(vmvnq_s16(redMaxMask), vmvnq_s16(greenMaxMask));

            int16x8_t dividend = vorrq_s16(
                vandq_s16(redMaxMask,
                    vaddq_s16(vsubq_s16(green, blue), vmulq_s16(range, (int16x8_t)K16_0006))),
                vorrq_s16(
                    vandq_s16(greenMaxMask,
                        vaddq_s16(vsubq_s16(blue, red), vmulq_s16(range, (int16x8_t)K16_0002))),
                    vandq_s16(blueMaxMask,
                        vaddq_s16(vsubq_s16(red, green), vmulq_s16(range, (int16x8_t)K16_0004)))));

            int16x8_t safeRange = vmaxq_s16(range, (int16x8_t)K16_0001);
            hue = vandq_s16(vmvnq_s16((int16x8_t)vceqq_s16(range, (int16x8_t)K16_0000)),
                vandq_s16(MulDiv16(dividend, safeRange, KF_255_DIV_6), (int16x8_t)K16_00FF));

            // Lightness: L = (max + min) / 2
            lgt = vshrq_n_s16(sum, 1);

            // Saturation: S = range * 255 / min(sum, 510 - sum), zero when range == 0
            int32x4_t range_lo = UnpackI16<0>(range);
            int32x4_t range_hi = UnpackI16<1>(range);
            int32x4_t sum_lo = UnpackI16<0>(sum);
            int32x4_t sum_hi = UnpackI16<1>(sum);

            const int32x4_t K32_510 = vdupq_n_s32(510);
            const int32x4_t K32_1 = vdupq_n_s32(1);
            const int32x4_t K32_0 = vdupq_n_s32(0);

            int32x4_t denom_lo = vminq_s32(sum_lo, vsubq_s32(K32_510, sum_lo));
            int32x4_t denom_hi = vminq_s32(sum_hi, vsubq_s32(K32_510, sum_hi));
            int32x4_t denomSafe_lo = vmaxq_s32(denom_lo, K32_1);
            int32x4_t denomSafe_hi = vmaxq_s32(denom_hi, K32_1);

            int32x4_t sat_lo = MulDiv32(range_lo, denomSafe_lo, K_255F);
            int32x4_t sat_hi = MulDiv32(range_hi, denomSafe_hi, K_255F);

            uint32x4_t zeroRangeMask_lo = vceqq_s32(range_lo, K32_0);
            uint32x4_t zeroRangeMask_hi = vceqq_s32(range_hi, K32_0);
            sat_lo = vbicq_s32(sat_lo, (int32x4_t)zeroRangeMask_lo);
            sat_hi = vbicq_s32(sat_hi, (int32x4_t)zeroRangeMask_hi);

            sat = PackI32(sat_lo, sat_hi);
        }

        template <bool align> SIMD_INLINE void BgrToHsl16x(const uint8_t * bgr, uint8_t * hsl,
            const float32x4_t & KF_255_DIV_6, const float32x4_t & K_255F)
        {
            uint8x16x3_t _bgr = Load3<align>(bgr);
            int16x8_t blue_lo = (int16x8_t)UnpackU8<0>(_bgr.val[0]);
            int16x8_t blue_hi = (int16x8_t)UnpackU8<1>(_bgr.val[0]);
            int16x8_t green_lo = (int16x8_t)UnpackU8<0>(_bgr.val[1]);
            int16x8_t green_hi = (int16x8_t)UnpackU8<1>(_bgr.val[1]);
            int16x8_t red_lo = (int16x8_t)UnpackU8<0>(_bgr.val[2]);
            int16x8_t red_hi = (int16x8_t)UnpackU8<1>(_bgr.val[2]);

            int16x8_t hue_lo, sat_lo, lgt_lo, hue_hi, sat_hi, lgt_hi;
            BgrToHsl16(blue_lo, green_lo, red_lo, hue_lo, sat_lo, lgt_lo, KF_255_DIV_6, K_255F);
            BgrToHsl16(blue_hi, green_hi, red_hi, hue_hi, sat_hi, lgt_hi, KF_255_DIV_6, K_255F);

            uint8x16x3_t _hsl;
            _hsl.val[0] = PackSaturatedI16(hue_lo, hue_hi);
            _hsl.val[1] = PackSaturatedI16(sat_lo, sat_hi);
            _hsl.val[2] = PackSaturatedI16(lgt_lo, lgt_hi);
            Store3<align>(hsl, _hsl);
        }

        template <bool align> void BgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(hsl) && Aligned(hslStride));

            size_t alignedWidth = AlignLo(width, A);
            const float32x4_t KF = vdupq_n_f32(Base::KF_255_DIV_6);
            const float32x4_t K255 = vdupq_n_f32(255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BgrToHsl16x<align>(bgr + 3 * col, hsl + 3 * col, KF, K255);
                if (width != alignedWidth)
                    BgrToHsl16x<false>(bgr + 3 * (width - A), hsl + 3 * (width - A), KF, K255);
                bgr += bgrStride;
                hsl += hslStride;
            }
        }

        void BgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(hsl) && Aligned(hslStride))
                BgrToHsl<true>(bgr, width, height, bgrStride, hsl, hslStride);
            else
                BgrToHsl<false>(bgr, width, height, bgrStride, hsl, hslStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
