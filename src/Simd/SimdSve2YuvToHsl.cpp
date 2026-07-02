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
#include "Simd/SimdConversion.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdPack.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t MulDiv32Hsl(const svint32_t& dividend, const svint32_t& divisor, const svfloat32_t& scale)
        {
            const svbool_t mask = svptrue_b32();
            return svcvt_s32_f32_x(mask, svdiv_f32_x(mask,
                svmul_f32_x(mask, scale, svcvt_f32_s32_x(mask, dividend)),
                svcvt_f32_s32_x(mask, divisor)));
        }

        SIMD_INLINE svint16_t MulDiv16Hsl(const svint16_t& dividend, const svint16_t& divisor, const svfloat32_t& scale)
        {
            svint32_t lo = MulDiv32Hsl(svmovlb_s32(dividend), svmovlb_s32(divisor), scale);
            svint32_t hi = MulDiv32Hsl(svmovlt_s32(dividend), svmovlt_s32(divisor), scale);
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svint16_t AdjustYLoHsl(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustYHiHsl(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVLoHsl(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVHiHsl(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t YuvToBlue16Hsl(const svint16_t& y, const svint16_t& u)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, u, Base::U_TO_BLUE_WEIGHT);
            hi = svmlalt_n_s32(hi, u, Base::U_TO_BLUE_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToGreen16Hsl(const svint16_t& y, const svint16_t& u, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(svmlalb_n_s32(lo, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            hi = svmlalt_n_s32(svmlalt_n_s32(hi, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToRed16Hsl(const svint16_t& y, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, v, Base::V_TO_RED_WEIGHT);
            hi = svmlalt_n_s32(hi, v, Base::V_TO_RED_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t SaturateByU8Hsl(const svint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svmin_n_s16_x(mask, svmax_n_s16_x(mask, value, 0), 255);
        }

        SIMD_INLINE svint16_t Saturation16Hsl(const svint16_t& range, const svint16_t& sum, const svfloat32_t& scale)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t rangeLo = svmovlb_s32(range);
            svint32_t rangeHi = svmovlt_s32(range);
            svint32_t sumLo = svmovlb_s32(sum);
            svint32_t sumHi = svmovlt_s32(sum);
            svint32_t denomLo = svmin_s32_x(mask, sumLo, svsub_s32_x(mask, svdup_n_s32(510), sumLo));
            svint32_t denomHi = svmin_s32_x(mask, sumHi, svsub_s32_x(mask, svdup_n_s32(510), sumHi));
            svint32_t satLo = MulDiv32Hsl(rangeLo, svmax_n_s32_x(mask, denomLo, 1), scale);
            svint32_t satHi = MulDiv32Hsl(rangeHi, svmax_n_s32_x(mask, denomHi, 1), scale);
            satLo = svsel_s32(svcmpeq_n_s32(mask, rangeLo, 0), svdup_n_s32(0), satLo);
            satHi = svsel_s32(svcmpeq_n_s32(mask, rangeHi, 0), svdup_n_s32(0), satHi);
            return svqxtnt_s32(svqxtnb_s32(satLo), satHi);
        }

        SIMD_INLINE void YuvToHsl16(const svint16_t& y, const svint16_t& u, const svint16_t& v,
            svint16_t& hue, svint16_t& sat, svint16_t& lgt, const svfloat32_t& hueScale, const svfloat32_t& satScale)
        {
            const svbool_t mask = svptrue_b16();
            svint16_t red = SaturateByU8Hsl(YuvToRed16Hsl(y, v));
            svint16_t green = SaturateByU8Hsl(YuvToGreen16Hsl(y, u, v));
            svint16_t blue = SaturateByU8Hsl(YuvToBlue16Hsl(y, u));
            svint16_t max = svmax_s16_x(mask, red, svmax_s16_x(mask, green, blue));
            svint16_t min = svmin_s16_x(mask, red, svmin_s16_x(mask, green, blue));
            svint16_t range = svsub_s16_x(mask, max, min);
            svint16_t sum = svadd_s16_x(mask, max, min);

            svbool_t redMax = svcmpeq_s16(mask, red, max);
            svbool_t greenMax = svand_b_z(mask, svcmpne_s16(mask, red, max), svcmpeq_s16(mask, green, max));
            svint16_t redCase = svmla_n_s16_x(mask, svsub_s16_x(mask, green, blue), range, 6);
            svint16_t greenCase = svmla_n_s16_x(mask, svsub_s16_x(mask, blue, red), range, 2);
            svint16_t blueCase = svmla_n_s16_x(mask, svsub_s16_x(mask, red, green), range, 4);
            svint16_t dividend = svsel_s16(redMax, redCase, svsel_s16(greenMax, greenCase, blueCase));
            svint16_t safeRange = svmax_n_s16_x(mask, range, 1);
            hue = svsel_s16(svcmpeq_n_s16(mask, range, 0), svdup_n_s16(0),
                svand_n_s16_x(mask, MulDiv16Hsl(dividend, safeRange, hueScale), 0x00FF));

            sat = Saturation16Hsl(range, sum, satScale);
            lgt = svasr_n_s16_x(mask, sum, 1);
        }

        SIMD_INLINE void Yuv444pToHsl(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* hsl,
            const svfloat32_t& hueScale, const svfloat32_t& satScale)
        {
            const svbool_t body = svptrue_b8();
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svint16_t hueLo, satLo, lgtLo, hueHi, satHi, lgtHi;
            YuvToHsl16(AdjustYLoHsl(_y), AdjustUVLoHsl(_u), AdjustUVLoHsl(_v), hueLo, satLo, lgtLo, hueScale, satScale);
            YuvToHsl16(AdjustYHiHsl(_y), AdjustUVHiHsl(_u), AdjustUVHiHsl(_v), hueHi, satHi, lgtHi, hueScale, satScale);
            svst3_u8(body, hsl, svcreate3_u8(
                PackSatIntI16ToU8(hueLo, hueHi),
                PackSatIntI16ToU8(satLo, satHi),
                PackSatIntI16ToU8(lgtLo, lgtHi)));
        }

        void Yuv444pToHsl(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* hsl, size_t hslStride)
        {
            const size_t A = svcntb(), A3 = A * 3;
            const size_t widthA = AlignLo(width, A);
            const svfloat32_t hueScale = svdup_n_f32(Base::KF_255_DIV_6);
            const svfloat32_t satScale = svdup_n_f32(255.0f);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colHsl = 0;
                for (; col < widthA; col += A, colHsl += A3)
                    Yuv444pToHsl(y + col, u + col, v + col, hsl + colHsl, hueScale, satScale);
                for (; col < width; ++col, colHsl += 3)
                    Base::YuvToHsl(y[col], u[col], v[col], hsl + colHsl);
                y += yStride;
                u += uStride;
                v += vStride;
                hsl += hslStride;
            }
        }
    }
#endif
}
