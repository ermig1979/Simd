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
#include "Simd/SimdPack.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t MulDiv32(const svint32_t& dividend, const svint32_t& divisor, const svfloat32_t& scale)
        {
            const svbool_t mask = svptrue_b32();
            return svcvt_s32_f32_x(mask, svdiv_f32_x(mask,
                svmul_f32_x(mask, scale, svcvt_f32_s32_x(mask, dividend)),
                svcvt_f32_s32_x(mask, divisor)));
        }

        SIMD_INLINE svint16_t MulDiv16(const svint16_t& dividend, const svint16_t& divisor, const svfloat32_t& scale)
        {
            svint32_t lo = MulDiv32(svmovlb_s32(dividend), svmovlb_s32(divisor), scale);
            svint32_t hi = MulDiv32(svmovlt_s32(dividend), svmovlt_s32(divisor), scale);
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svint16_t AdjustYLo(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustYHi(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(y)), Base::Y_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVLo(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t AdjustUVHi(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(uv)), Base::UV_ADJUST);
        }

        SIMD_INLINE svint16_t YuvToBlue16(const svint16_t& y, const svint16_t& u)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, u, Base::U_TO_BLUE_WEIGHT);
            hi = svmlalt_n_s32(hi, u, Base::U_TO_BLUE_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToGreen16(const svint16_t& y, const svint16_t& u, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(svmlalb_n_s32(lo, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            hi = svmlalt_n_s32(svmlalt_n_s32(hi, u, Base::U_TO_GREEN_WEIGHT), v, Base::V_TO_GREEN_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t YuvToRed16(const svint16_t& y, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(Base::YUV_TO_BGR_ROUND_TERM), y, Base::Y_TO_RGB_WEIGHT);
            lo = svmlalb_n_s32(lo, v, Base::V_TO_RED_WEIGHT);
            hi = svmlalt_n_s32(hi, v, Base::V_TO_RED_WEIGHT);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, Base::YUV_TO_BGR_AVERAGING_SHIFT)), svasr_n_s32_x(mask, hi, Base::YUV_TO_BGR_AVERAGING_SHIFT));
        }

        SIMD_INLINE svint16_t SaturateByU8(const svint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svmin_n_s16_x(mask, svmax_n_s16_x(mask, value, 0), 255);
        }

        SIMD_INLINE svuint8_t YuvToHue(const svuint8_t& y, const svuint8_t& u, const svuint8_t& v, const svfloat32_t& scale)
        {
            const svbool_t mask16 = svptrue_b16();
            svint16_t yLo = AdjustYLo(y), yHi = AdjustYHi(y);
            svint16_t uLo = AdjustUVLo(u), uHi = AdjustUVHi(u);
            svint16_t vLo = AdjustUVLo(v), vHi = AdjustUVHi(v);
            svint16_t redLo = SaturateByU8(YuvToRed16(yLo, vLo));
            svint16_t redHi = SaturateByU8(YuvToRed16(yHi, vHi));
            svint16_t greenLo = SaturateByU8(YuvToGreen16(yLo, uLo, vLo));
            svint16_t greenHi = SaturateByU8(YuvToGreen16(yHi, uHi, vHi));
            svint16_t blueLo = SaturateByU8(YuvToBlue16(yLo, uLo));
            svint16_t blueHi = SaturateByU8(YuvToBlue16(yHi, uHi));

            svint16_t maxLo = svmax_s16_x(mask16, redLo, svmax_s16_x(mask16, greenLo, blueLo));
            svint16_t maxHi = svmax_s16_x(mask16, redHi, svmax_s16_x(mask16, greenHi, blueHi));
            svint16_t minLo = svmin_s16_x(mask16, redLo, svmin_s16_x(mask16, greenLo, blueLo));
            svint16_t minHi = svmin_s16_x(mask16, redHi, svmin_s16_x(mask16, greenHi, blueHi));
            svint16_t rangeLo = svsub_s16_x(mask16, maxLo, minLo);
            svint16_t rangeHi = svsub_s16_x(mask16, maxHi, minHi);

            svbool_t redMaxLo = svcmpeq_s16(mask16, redLo, maxLo);
            svbool_t redMaxHi = svcmpeq_s16(mask16, redHi, maxHi);
            svbool_t greenMaxLo = svand_b_z(mask16, svcmpne_s16(mask16, redLo, maxLo), svcmpeq_s16(mask16, greenLo, maxLo));
            svbool_t greenMaxHi = svand_b_z(mask16, svcmpne_s16(mask16, redHi, maxHi), svcmpeq_s16(mask16, greenHi, maxHi));

            svint16_t redCaseLo = svmla_n_s16_x(mask16, svsub_s16_x(mask16, greenLo, blueLo), rangeLo, 6);
            svint16_t redCaseHi = svmla_n_s16_x(mask16, svsub_s16_x(mask16, greenHi, blueHi), rangeHi, 6);
            svint16_t greenCaseLo = svmla_n_s16_x(mask16, svsub_s16_x(mask16, blueLo, redLo), rangeLo, 2);
            svint16_t greenCaseHi = svmla_n_s16_x(mask16, svsub_s16_x(mask16, blueHi, redHi), rangeHi, 2);
            svint16_t blueCaseLo = svmla_n_s16_x(mask16, svsub_s16_x(mask16, redLo, greenLo), rangeLo, 4);
            svint16_t blueCaseHi = svmla_n_s16_x(mask16, svsub_s16_x(mask16, redHi, greenHi), rangeHi, 4);
            svint16_t dividendLo = svsel_s16(redMaxLo, redCaseLo, svsel_s16(greenMaxLo, greenCaseLo, blueCaseLo));
            svint16_t dividendHi = svsel_s16(redMaxHi, redCaseHi, svsel_s16(greenMaxHi, greenCaseHi, blueCaseHi));

            svint16_t rangeSafeLo = svmax_n_s16_x(mask16, rangeLo, 1);
            svint16_t rangeSafeHi = svmax_n_s16_x(mask16, rangeHi, 1);
            svint16_t hueLo = svsel_s16(svcmpeq_n_s16(mask16, rangeLo, 0), svdup_n_s16(0),
                svand_n_s16_x(mask16, MulDiv16(dividendLo, rangeSafeLo, scale), 0x00FF));
            svint16_t hueHi = svsel_s16(svcmpeq_n_s16(mask16, rangeHi, 0), svdup_n_s16(0),
                svand_n_s16_x(mask16, MulDiv16(dividendHi, rangeSafeHi, scale), 0x00FF));
            return PackSatIntI16ToU8(hueLo, hueHi);
        }

        SIMD_INLINE int YuvToHue(int y, int u, int v)
        {
            int red = Base::YuvToRed(y, v);
            int green = Base::YuvToGreen(y, u, v);
            int blue = Base::YuvToBlue(y, u);

            int max = Max(red, Max(green, blue));
            int min = Min(red, Min(green, blue));
            int range = max - min;
            if (range)
            {
                int dividend;
                if (red == max)
                    dividend = green - blue + 6 * range;
                else if (green == max)
                    dividend = blue - red + 2 * range;
                else
                    dividend = red - green + 4 * range;
                return int(Base::KF_255_DIV_6 * float(dividend) / float(range));
            }
            return 0;
        }

        void Yuv420pToHue(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* hue, size_t hueStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            const size_t A = svcntb(), DA = 2 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            const svfloat32_t scale = svdup_n_f32(Base::KF_255_DIV_6);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUV = 0, col = 0;
                for (; col < widthDA; col += DA, colUV += A)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    svuint8_t u0 = svzip1_u8(_u, _u), v0 = svzip1_u8(_v, _v);
                    svuint8_t u1 = svzip2_u8(_u, _u), v1 = svzip2_u8(_v, _v);
                    svst1_u8(body, hue + col + 0 * A, YuvToHue(svld1_u8(body, y + col + 0 * A), u0, v0, scale));
                    svst1_u8(body, hue + col + 1 * A, YuvToHue(svld1_u8(body, y + col + 1 * A), u1, v1, scale));
                    svst1_u8(body, hue + hueStride + col + 0 * A, YuvToHue(svld1_u8(body, y + yStride + col + 0 * A), u0, v0, scale));
                    svst1_u8(body, hue + hueStride + col + 1 * A, YuvToHue(svld1_u8(body, y + yStride + col + 1 * A), u1, v1, scale));
                }
                for (; col < width; col += 2, colUV += 1)
                {
                    int _u = u[colUV], _v = v[colUV];
                    hue[col] = YuvToHue(y[col], _u, _v);
                    hue[col + 1] = YuvToHue(y[col + 1], _u, _v);
                    hue[hueStride + col] = YuvToHue(y[yStride + col], _u, _v);
                    hue[hueStride + col + 1] = YuvToHue(y[yStride + col + 1], _u, _v);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                hue += 2 * hueStride;
            }
        }
    }
#endif
}
