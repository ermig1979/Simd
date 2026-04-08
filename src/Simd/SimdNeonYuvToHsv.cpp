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
        SIMD_INLINE int32x4_t MulDiv32Hsv(const int32x4_t & dividend, const int32x4_t & divisor, const float32x4_t & KF)
        {
            return vcvtq_s32_f32(Div<SIMD_NEON_RCP_ITER>(vmulq_f32(KF, vcvtq_f32_s32(dividend)), vcvtq_f32_s32(divisor)));
        }

        SIMD_INLINE int16x8_t MulDiv16Hsv(const int16x8_t & dividend, const int16x8_t & divisor, const float32x4_t & KF)
        {
            int32x4_t lo = MulDiv32Hsv(UnpackI16<0>(dividend), UnpackI16<0>(divisor), KF);
            int32x4_t hi = MulDiv32Hsv(UnpackI16<1>(dividend), UnpackI16<1>(divisor), KF);
            return PackI32(lo, hi);
        }

        SIMD_INLINE void YuvToHsv16(int16x8_t y, int16x8_t u, int16x8_t v,
            int16x8_t & hue, int16x8_t & sat, int16x8_t & val,
            const float32x4_t & KF_255_DIV_6, const float32x4_t & K_255F)
        {
            int16x8_t red = SaturateByU8(YuvToRed16(y, v));
            int16x8_t blue = SaturateByU8(YuvToBlue16(y, u));
            int16x8_t green = SaturateByU8(YuvToGreen16(y, u, v));

            int16x8_t max = vmaxq_s16(red, vmaxq_s16(green, blue));
            int16x8_t min = vminq_s16(red, vminq_s16(green, blue));
            int16x8_t range = vsubq_s16(max, min);

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
                vandq_s16(MulDiv16Hsv(dividend, safeRange, KF_255_DIV_6), (int16x8_t)K16_00FF));

            // Value: V = max
            val = max;

            // Saturation: S = 255 * range / max, zero when max == 0
            int16x8_t safeMax = vmaxq_s16(max, (int16x8_t)K16_0001);
            sat = vandq_s16(vmvnq_s16((int16x8_t)vceqq_s16(max, (int16x8_t)K16_0000)),
                MulDiv16Hsv(range, safeMax, K_255F));
        }

        template <bool align> SIMD_INLINE void YuvToHsv16x(const uint8_t * y, const uint8_t * u, const uint8_t * v,
            uint8_t * hsv, const float32x4_t & KF_255_DIV_6, const float32x4_t & K_255F)
        {
            uint8x16_t y8 = Load<align>(y);
            uint8x16_t u8 = Load<align>(u);
            uint8x16_t v8 = Load<align>(v);

            int16x8_t hue_lo, sat_lo, val_lo, hue_hi, sat_hi, val_hi;
            YuvToHsv16(
                AdjustY<0>(y8), AdjustUV<0>(u8), AdjustUV<0>(v8),
                hue_lo, sat_lo, val_lo, KF_255_DIV_6, K_255F);
            YuvToHsv16(
                AdjustY<1>(y8), AdjustUV<1>(u8), AdjustUV<1>(v8),
                hue_hi, sat_hi, val_hi, KF_255_DIV_6, K_255F);

            uint8x16x3_t _hsv;
            _hsv.val[0] = PackSaturatedI16(hue_lo, hue_hi);
            _hsv.val[1] = PackSaturatedI16(sat_lo, sat_hi);
            _hsv.val[2] = PackSaturatedI16(val_lo, val_hi);
            Store3<align>(hsv, _hsv);
        }

        template <bool align> void Yuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride,
            const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hsv, size_t hsvStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hsv) && Aligned(hsvStride));
            }

            const float32x4_t KF = vdupq_n_f32(Base::KF_255_DIV_6);
            const float32x4_t K255 = vdupq_n_f32(255.0f);

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    YuvToHsv16x<align>(y + col, u + col, v + col, hsv + 3 * col, KF, K255);
                if (width != alignedWidth)
                {
                    size_t col = width - A;
                    YuvToHsv16x<false>(y + col, u + col, v + col, hsv + 3 * col, KF, K255);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hsv += hsvStride;
            }
        }

        void Yuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride,
            const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hsv, size_t hsvStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) &&
                Aligned(v) && Aligned(vStride) && Aligned(hsv) && Aligned(hsvStride))
                Yuv444pToHsv<true>(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
            else
                Yuv444pToHsv<false>(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
