/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE int32x4_t MulDiv(const int32x4_t & dividend, const int32x4_t & divisor, const float32x4_t & KF_255_DIV_6)
        {
            return vcvtq_s32_f32(Div<SIMD_NEON_RCP_ITER>(vmulq_f32(KF_255_DIV_6, vcvtq_f32_s32(dividend)), vcvtq_f32_s32(divisor)));
        }

        SIMD_INLINE int16x8_t MulDiv(const int16x8_t & dividend, const int16x8_t & divisor, const float32x4_t & KF_255_DIV_6)
        {
            int32x4_t lo = MulDiv(UnpackI16<0>(dividend), UnpackI16<0>(divisor), KF_255_DIV_6);
            int32x4_t hi = MulDiv(UnpackI16<1>(dividend), UnpackI16<1>(divisor), KF_255_DIV_6);
            return PackI32(lo, hi);
        }

        SIMD_INLINE int16x8_t YuvToHue(const int16x8_t & y, const int16x8_t & u, const int16x8_t & v, const float32x4_t & KF_255_DIV_6)
        {
            int16x8_t red = SaturateByU8(YuvToRed(y, v));
            int16x8_t blue = SaturateByU8(YuvToBlue(y, u));
            int16x8_t green = SaturateByU8(YuvToGreen(y, u, v));
            int16x8_t max = vmaxq_s16(blue, vmaxq_s16(green, red));
            int16x8_t min = vminq_s16(blue, vminq_s16(green, red));
            int16x8_t range = vsubq_s16(max, min);

            int16x8_t redMaxMask = (int16x8_t)vceqq_s16(red, max);
            int16x8_t greenMaxMask = vandq_s16(vmvnq_s16(redMaxMask), (int16x8_t)vceqq_s16(green, max));
            int16x8_t blueMaxMask = vandq_s16(vmvnq_s16(redMaxMask), vmvnq_s16(greenMaxMask));

            int16x8_t redMaxCase = vandq_s16(redMaxMask, vaddq_s16(vsubq_s16(green, blue), vmulq_s16(range, (int16x8_t)K16_0006)));
            int16x8_t greenMaxCase = vandq_s16(greenMaxMask, vaddq_s16(vsubq_s16(blue, red), vmulq_s16(range, (int16x8_t)K16_0002)));
            int16x8_t blueMaxCase = vandq_s16(blueMaxMask, vaddq_s16(vsubq_s16(red, green), vmulq_s16(range, (int16x8_t)K16_0004)));

            int16x8_t dividend = vorrq_s16(vorrq_s16(redMaxCase, greenMaxCase), blueMaxCase);

            return vandq_s16(vmvnq_s16((int16x8_t)vceqq_s16(range, (int16x8_t)K16_0000)), vandq_s16(MulDiv(dividend, range, KF_255_DIV_6), (int16x8_t)K16_00FF));
        }

        SIMD_INLINE uint8x16_t YuvToHue(const uint8x16_t & y, const uint8x16_t & u, const uint8x16_t & v, const float32x4_t & KF_255_DIV_6)
        {
            uint16x8_t lo = (uint16x8_t)YuvToHue(AdjustY<0>(y), AdjustUV<0>(u), AdjustUV<0>(v), KF_255_DIV_6);
            uint16x8_t hi = (uint16x8_t)YuvToHue(AdjustY<1>(y), AdjustUV<1>(u), AdjustUV<1>(v), KF_255_DIV_6);
            return PackU16(lo, hi);
        }

        template <bool align> SIMD_INLINE void Yuv420pToHue(const uint8_t * y, const uint8x16x2_t & u, const uint8x16x2_t & v, uint8_t * hue, const float32x4_t & KF_255_DIV_6)
        {
            Store<align>(hue + 0, YuvToHue(Load<align>(y + 0), u.val[0], v.val[0], KF_255_DIV_6));
            Store<align>(hue + A, YuvToHue(Load<align>(y + A), u.val[1], v.val[1], KF_255_DIV_6));
        }

        template <bool align> void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride));
            }

            const float32x4_t KF_255_DIV_6 = vdupq_n_f32(Base::KF_255_DIV_6);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, col = 0; col < bodyWidth; col += DA, colUV += A)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv420pToHue<align>(y + col, _u, _v, hue + col, KF_255_DIV_6);
                    Yuv420pToHue<align>(y + yStride + col, _u, _v, hue + hueStride + col, KF_255_DIV_6);
                }
                if (tail)
                {
                    size_t col = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + col / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + col / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv420pToHue<false>(y + col, _u, _v, hue + col, KF_255_DIV_6);
                    Yuv420pToHue<false>(y + yStride + col, _u, _v, hue + hueStride + col, KF_255_DIV_6);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                hue += 2 * hueStride;
            }
        }

        void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv420pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv420pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }

        template <bool align> void Yuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride));
            }

            const float32x4_t KF_255_DIV_6 = vdupq_n_f32(Base::KF_255_DIV_6);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                    Store<align>(hue + col, YuvToHue(Load<align>(y + col), Load<align>(u + col), Load<align>(v + col), KF_255_DIV_6));
                if (tail)
                {
                    size_t col = width - A;
                    Store<false>(hue + col, YuvToHue(Load<false>(y + col), Load<false>(u + col), Load<false>(v + col), KF_255_DIV_6));
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hue += hueStride;
            }
        }

        void Yuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv444pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv444pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
