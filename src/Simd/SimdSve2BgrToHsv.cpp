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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t MulDiv32Hsv(const svint32_t& dividend, const svint32_t& divisor, const svfloat32_t& scale)
        {
            const svbool_t mask = svptrue_b32();
            return svcvt_s32_f32_x(mask, svdiv_f32_x(mask,
                svmul_f32_x(mask, scale, svcvt_f32_s32_x(mask, dividend)),
                svcvt_f32_s32_x(mask, divisor)));
        }

        SIMD_INLINE svint16_t MulDiv16Hsv(const svint16_t& dividend, const svint16_t& divisor, const svfloat32_t& scale)
        {
            svint32_t lo = MulDiv32Hsv(svmovlb_s32(dividend), svmovlb_s32(divisor), scale);
            svint32_t hi = MulDiv32Hsv(svmovlt_s32(dividend), svmovlt_s32(divisor), scale);
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svuint8_t PackI16ToU8(const svint16_t& lo, const svint16_t& hi)
        {
            return svqxtnt_u16(svqxtnb_u16(svreinterpret_u16_s16(lo)), svreinterpret_u16_s16(hi));
        }

        SIMD_INLINE void BgrToHsv16(svint16_t blue, svint16_t green, svint16_t red,
            svint16_t& hue, svint16_t& sat, svint16_t& val,
            const svfloat32_t& KF_255_DIV_6, const svfloat32_t& K_255F)
        {
            const svbool_t mask16 = svptrue_b16();

            svint16_t max = svmax_s16_x(mask16, red, svmax_s16_x(mask16, green, blue));
            svint16_t min = svmin_s16_x(mask16, red, svmin_s16_x(mask16, green, blue));
            svint16_t range = svsub_s16_x(mask16, max, min);

            svbool_t redMaxMask = svcmpeq_s16(mask16, red, max);
            svbool_t greenMaxMask = svcmpeq_s16(mask16, green, max);

            svint16_t dividendRed = svmla_n_s16_x(mask16, svsub_s16_x(mask16, green, blue), range, 6);
            svint16_t dividendGreen = svmla_n_s16_x(mask16, svsub_s16_x(mask16, blue, red), range, 2);
            svint16_t dividendBlue = svmla_n_s16_x(mask16, svsub_s16_x(mask16, red, green), range, 4);
            svint16_t dividend = svsel_s16(redMaxMask, dividendRed, svsel_s16(greenMaxMask, dividendGreen, dividendBlue));

            svint16_t safeRange = svmax_n_s16_x(mask16, range, 1);
            hue = svsel_s16(svcmpeq_n_s16(mask16, range, 0), svdup_n_s16(0),
                svand_n_s16_x(mask16, MulDiv16Hsv(dividend, safeRange, KF_255_DIV_6), 0x00FF));

            val = max;

            svint16_t safeMax = svmax_n_s16_x(mask16, max, 1);
            sat = svsel_s16(svcmpeq_n_s16(mask16, max, 0), svdup_n_s16(0),
                MulDiv16Hsv(range, safeMax, K_255F));
        }

        SIMD_INLINE void BgrToHsv(const uint8_t* bgr, uint8_t* hsv,
            const svfloat32_t& KF_255_DIV_6, const svfloat32_t& K_255F, const svbool_t& mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svint16_t blueLo = svreinterpret_s16_u16(svmovlb_u16(svget3(_bgr, 0)));
            svint16_t blueHi = svreinterpret_s16_u16(svmovlt_u16(svget3(_bgr, 0)));
            svint16_t greenLo = svreinterpret_s16_u16(svmovlb_u16(svget3(_bgr, 1)));
            svint16_t greenHi = svreinterpret_s16_u16(svmovlt_u16(svget3(_bgr, 1)));
            svint16_t redLo = svreinterpret_s16_u16(svmovlb_u16(svget3(_bgr, 2)));
            svint16_t redHi = svreinterpret_s16_u16(svmovlt_u16(svget3(_bgr, 2)));

            svint16_t hueLo, satLo, valLo, hueHi, satHi, valHi;
            BgrToHsv16(blueLo, greenLo, redLo, hueLo, satLo, valLo, KF_255_DIV_6, K_255F);
            BgrToHsv16(blueHi, greenHi, redHi, hueHi, satHi, valHi, KF_255_DIV_6, K_255F);

            svst3_u8(mask, hsv, svcreate3_u8(PackI16ToU8(hueLo, hueHi), PackI16ToU8(satLo, satHi), PackI16ToU8(valLo, valHi)));
        }

        void BgrToHsv(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsv, size_t hsvStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svfloat32_t KF = svdup_n_f32(Base::KF_255_DIV_6);
            const svfloat32_t K255 = svdup_n_f32(255.0f);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToHsv(bgr + offset, hsv + offset, KF, K255, body);
                if (widthA < width)
                    BgrToHsv(bgr + offset, hsv + offset, KF, K255, tail);
                bgr += bgrStride;
                hsv += hsvStride;
            }
        }
    }
#endif
}
