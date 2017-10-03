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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        SIMD_INLINE v128_u32 MulDiv(const v128_u32 & dividend, const v128_u32 & divisor, const v128_f32 & KF_255_DIV_6)
        {
            return vec_ctu(vec_div(vec_mul(KF_255_DIV_6, vec_ctf(dividend, 0)), vec_ctf(divisor, 0)), 0);
        }

        SIMD_INLINE v128_u16 MulDiv(const v128_u16 & dividend, const v128_u16 & divisor, const v128_f32 & KF_255_DIV_6)
        {
            const v128_u32 quotientLo = MulDiv(UnpackLoU16(dividend), UnpackLoU16(divisor), KF_255_DIV_6);
            const v128_u32 quotientHi = MulDiv(UnpackHiU16(dividend), UnpackHiU16(divisor), KF_255_DIV_6);
            return vec_pack(quotientLo, quotientHi);
        }

        SIMD_INLINE v128_u16 AdjustedYuvToHue(const v128_s16 & y, const v128_s16 & u, const v128_s16 & v, const v128_f32 & KF_255_DIV_6)
        {
            const v128_u16 blue = AdjustedYuvToBlue(y, u);
            const v128_u16 green = AdjustedYuvToGreen(y, u, v);
            const v128_u16 red = AdjustedYuvToRed(y, v);
            const v128_u16 max = Max(red, green, blue);
            const v128_u16 range = vec_subs(max, Min(red, green, blue));

            const v128_u16 dividend = vec_sel(vec_sel(
                vec_mladd(range, K16_0004, vec_sub(red, green)),
                vec_mladd(range, K16_0002, vec_sub(blue, red)), vec_cmpeq(green, max)),
                vec_mladd(range, K16_0006, vec_sub(green, blue)), vec_cmpeq(red, max));

            return vec_sel(vec_and(MulDiv(dividend, range, KF_255_DIV_6), K16_00FF), K16_0000, vec_cmpeq(range, K16_0000));
        }

        SIMD_INLINE v128_u8 YuvToHue(const v128_u8 & y, const v128_u8 & u, const v128_u8 & v, const v128_f32 & KF_255_DIV_6)
        {
            return vec_pack(
                AdjustedYuvToHue(AdjustY(UnpackLoU8(y)), AdjustUV(UnpackLoU8(u)), AdjustUV(UnpackLoU8(v)), KF_255_DIV_6),
                AdjustedYuvToHue(AdjustY(UnpackHiU8(y)), AdjustUV(UnpackHiU8(u)), AdjustUV(UnpackHiU8(v)), KF_255_DIV_6));
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv420pToHue(const uint8_t * y, const v128_u8 & u, const v128_u8 & v, const v128_f32 & KF_255_DIV_6, Storer<align> & hue)
        {
            Store<align, first>(hue, YuvToHue(Load<align>(y + 0), (v128_u8)UnpackLoU8(u, u), (v128_u8)UnpackLoU8(v, v), KF_255_DIV_6));
            Store<align, false>(hue, YuvToHue(Load<align>(y + A), (v128_u8)UnpackHiU8(u, u), (v128_u8)UnpackHiU8(v, v), KF_255_DIV_6));
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

            const v128_f32 KF_255_DIV_6 = SetF32(Base::KF_255_DIV_6);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                Storer<align> _hue0(hue), _hue1(hue + hueStride);
                v128_u8 _u = Load<align>(u);
                v128_u8 _v = Load<align>(v);
                Yuv420pToHue<align, true>(y, _u, _v, KF_255_DIV_6, _hue0);
                Yuv420pToHue<align, true>(y + yStride, _u, _v, KF_255_DIV_6, _hue1);
                for (size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                {
                    v128_u8 _u = Load<align>(u + colUV);
                    v128_u8 _v = Load<align>(v + colUV);
                    Yuv420pToHue<align, false>(y + colY, _u, _v, KF_255_DIV_6, _hue0);
                    Yuv420pToHue<align, false>(y + colY + yStride, _u, _v, KF_255_DIV_6, _hue1);
                }
                Flush(_hue0, _hue1);

                if (tail)
                {
                    size_t offset = width - DA;
                    Storer<false> _hue0(hue + offset), _hue1(hue + offset + hueStride);
                    v128_u8 _u = Load<false>(u + offset / 2);
                    v128_u8 _v = Load<false>(v + offset / 2);
                    Yuv420pToHue<false, true>(y + offset, _u, _v, KF_255_DIV_6, _hue0);
                    Yuv420pToHue<false, true>(y + offset + yStride, _u, _v, KF_255_DIV_6, _hue1);
                    Flush(_hue0, _hue1);
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

        template <bool align, bool first>
        SIMD_INLINE void Yuv444pToHue(const uint8_t * y, const uint8_t * u, const uint8_t * v, const v128_f32 & KF_255_DIV_6, Storer<align> & hue)
        {
            Store<align, first>(hue, YuvToHue(Load<align>(y), Load<align>(u), Load<align>(v), KF_255_DIV_6));
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

            const v128_f32 KF_255_DIV_6 = SetF32(Base::KF_255_DIV_6);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _hue(hue);
                Yuv444pToHue<align, true>(y, u, v, KF_255_DIV_6, _hue);
                for (size_t col = A; col < bodyWidth; col += A)
                    Yuv444pToHue<align, false>(y + col, u + col, v + col, KF_255_DIV_6, _hue);
                Flush(_hue);

                if (tail)
                {
                    size_t col = width - A;
                    Storer<false> _hue(hue + col);
                    Yuv444pToHue<false, true>(y + col, u + col, v + col, KF_255_DIV_6, _hue);
                    Flush(_hue);
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
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv444pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv444pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}
