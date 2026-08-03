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
        SIMD_INLINE svint32_t MulDiv32YuvHsv(const svint32_t& dividend, const svint32_t& divisor, const svfloat32_t& scale)
        {
            const svbool_t mask = svptrue_b32();
            return svcvt_s32_f32_x(mask, svdiv_f32_x(mask,
                svmul_f32_x(mask, scale, svcvt_f32_s32_x(mask, dividend)),
                svcvt_f32_s32_x(mask, divisor)));
        }

        SIMD_INLINE svint16_t MulDiv16YuvHsv(const svint16_t& dividend, const svint16_t& divisor, const svfloat32_t& scale)
        {
            svint32_t lo = MulDiv32YuvHsv(svmovlb_s32(dividend), svmovlb_s32(divisor), scale);
            svint32_t hi = MulDiv32YuvHsv(svmovlt_s32(dividend), svmovlt_s32(divisor), scale);
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svint16_t AdjustYLoYuvHsv(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustYHiYuvHsv(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVLoYuvHsv(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVHiYuvHsv(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t YuvToBlue16Hsv(const svint16_t& y, const svint16_t& u)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, u, Base::U_TO_BLUE_WEIGHT);
            hi = svmlalt_n_s32(hi, u, Base::U_TO_BLUE_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToGreen16Hsv(const svint16_t& y, const svint16_t& u, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(svmlalb_n_s32(lo, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            hi = svmlalt_n_s32(svmlalt_n_s32(hi, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToRed16Hsv(const svint16_t& y, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, v, Base::V_TO_RED_WEIGHT);
            hi = svmlalt_n_s32(hi, v, Base::V_TO_RED_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t SaturateByU8Hsv(const svint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svmin_n_s16_x(mask, svmax_n_s16_x(mask, value, 0), 255);
        }

        SIMD_INLINE void YuvToHsv16(const svint16_t& y, const svint16_t& u, const svint16_t& v,
            svint16_t& hue, svint16_t& sat, svint16_t& val, const svfloat32_t& hueScale, const svfloat32_t& satScale)
        {
            const svbool_t mask = svptrue_b16();
            svint16_t red = SaturateByU8Hsv(YuvToRed16Hsv(y, v));
            svint16_t green = SaturateByU8Hsv(YuvToGreen16Hsv(y, u, v));
            svint16_t blue = SaturateByU8Hsv(YuvToBlue16Hsv(y, u));
            svint16_t max = svmax_s16_x(mask, red, svmax_s16_x(mask, green, blue));
            svint16_t min = svmin_s16_x(mask, red, svmin_s16_x(mask, green, blue));
            svint16_t range = svsub_s16_x(mask, max, min);

            svbool_t redMax = svcmpeq_s16(mask, red, max);
            svbool_t greenMax = svand_b_z(mask, svcmpne_s16(mask, red, max), svcmpeq_s16(mask, green, max));
            svint16_t redCase = svmla_n_s16_x(mask, svsub_s16_x(mask, green, blue), range, 6);
            svint16_t greenCase = svmla_n_s16_x(mask, svsub_s16_x(mask, blue, red), range, 2);
            svint16_t blueCase = svmla_n_s16_x(mask, svsub_s16_x(mask, red, green), range, 4);
            svint16_t dividend = svsel_s16(redMax, redCase, svsel_s16(greenMax, greenCase, blueCase));
            svint16_t safeRange = svmax_n_s16_x(mask, range, 1);
            hue = svsel_s16(svcmpeq_n_s16(mask, range, 0), svdup_n_s16(0),
                svand_n_s16_x(mask, MulDiv16YuvHsv(dividend, safeRange, hueScale), 0x00FF));

            val = max;
            svint16_t safeMax = svmax_n_s16_x(mask, max, 1);
            sat = svsel_s16(svcmpeq_n_s16(mask, max, 0), svdup_n_s16(0), MulDiv16YuvHsv(range, safeMax, satScale));
        }

        SIMD_INLINE void Yuv444pToHsv(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* hsv,
            const svfloat32_t& hueScale, const svfloat32_t& satScale)
        {
            const svbool_t body = svptrue_b8();
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svint16_t hueLo, satLo, valLo, hueHi, satHi, valHi;
            YuvToHsv16(AdjustYLoYuvHsv(_y), AdjustUVLoYuvHsv(_u), AdjustUVLoYuvHsv(_v), hueLo, satLo, valLo, hueScale, satScale);
            YuvToHsv16(AdjustYHiYuvHsv(_y), AdjustUVHiYuvHsv(_u), AdjustUVHiYuvHsv(_v), hueHi, satHi, valHi, hueScale, satScale);
            svst3_u8(body, hsv, svcreate3_u8(
                PackSatIntI16ToU8(hueLo, hueHi),
                PackSatIntI16ToU8(satLo, satHi),
                PackSatIntI16ToU8(valLo, valHi)));
        }

        void Yuv444pToHsv(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* hsv, size_t hsvStride)
        {
            const size_t A = svcntb(), A3 = A * 3;
            const size_t widthA = AlignLo(width, A);
            const svfloat32_t hueScale = svdup_n_f32(Base::KF_255_DIV_6);
            const svfloat32_t satScale = svdup_n_f32(255.0f);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colHsv = 0;
                for (; col < widthA; col += A, colHsv += A3)
                    Yuv444pToHsv(y + col, u + col, v + col, hsv + colHsv, hueScale, satScale);
                for (; col < width; ++col, colHsv += 3)
                    Base::YuvToHsv(y[col], u[col], v[col], hsv + colHsv);
                y += yStride;
                u += uStride;
                v += vStride;
                hsv += hsvStride;
            }
        }
    }
#endif
}
