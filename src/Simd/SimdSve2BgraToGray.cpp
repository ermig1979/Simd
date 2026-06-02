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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        SIMD_INLINE svuint16_t BgrToGray(const svuint16_t& b, const svuint16_t& g, const svuint16_t& r, 
            const svuint16_t& wb, const svuint16_t& wg, const svuint16_t& wr, const svuint32_t& rt)
        {
            svuint32_t bGray = svmlalb_u32(svmlalb_u32(svmlalb_u32(rt, b, wb), g, wg), r, wr);
            svuint32_t tGray = svmlalt_u32(svmlalt_u32(svmlalt_u32(rt, b, wb), g, wg), r, wr);
            return svshrnt_n_u32(svshrnb_n_u32(bGray, Base::BGR_TO_GRAY_AVERAGING_SHIFT), tGray, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE void BgraToGray(const uint8_t* bgra, const svuint16_t& wb, const svuint16_t& wg, const svuint16_t& wr, const svuint32_t& rt, uint8_t* gray, const svbool_t& mask)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svuint16_t bGray = BgrToGray(svmovlb_u16(svget4(_bgra, 0)), svmovlb_u16(svget4(_bgra, 1)), svmovlb_u16(svget4(_bgra, 2)), wb, wg, wr, rt);
            svuint16_t tGray = BgrToGray(svmovlt_u16(svget4(_bgra, 0)), svmovlt_u16(svget4(_bgra, 1)), svmovlt_u16(svget4(_bgra, 2)), wb, wg, wr, rt);
            svst1_u8(mask, gray, svqxtnt_u16(svqxtnb_u16(bGray), tGray));
        }

        void BgraToGray(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* gray, size_t grayStride)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint16_t wb = svdup_n_u16(Base::BLUE_TO_GRAY_WEIGHT);
            svuint16_t wg = svdup_n_u16(Base::GREEN_TO_GRAY_WEIGHT);
            svuint16_t wr = svdup_n_u16(Base::RED_TO_GRAY_WEIGHT);
            svuint32_t rt = svdup_n_u32(Base::BGR_TO_GRAY_ROUND_TERM);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    BgraToGray(bgra + offset, wb, wg, wr, rt, gray + col, body);
                if (widthA < width)
                    BgraToGray(bgra + offset, wb, wg, wr, rt, gray + col, tail);
                bgra += bgraStride;
                gray += grayStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t RgbaToGray(svuint8_t rgba, const svuint16_t& wrb, const svuint16_t& wg0, const svuint32_t& rt)
        {
            svuint16_t rb = svmovlb_u16(rgba);
            svuint16_t ga = svmovlt_u16(rgba);
            svuint32_t gray = svmlalt_u32(svmlalb_u32(svmlalb_u32(rt, ga, wg0), rb, wrb), rb, wrb);
            return svlsr_n_u32_x(svptrue_b8(), gray, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE void RgbaToGray(const uint8_t* rgba, const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3, 
            const svuint16_t& wrb, const svuint16_t& wg0, const svuint32_t& rt, uint8_t* gray, const svbool_t& mask)
        {
            svuint32_t gray0 = RgbaToGray(svld1_vnum_u8(mask0, rgba, 0), wrb, wg0, rt);
            svuint32_t gray1 = RgbaToGray(svld1_vnum_u8(mask1, rgba, 1), wrb, wg0, rt);
            svuint32_t gray2 = RgbaToGray(svld1_vnum_u8(mask2, rgba, 2), wrb, wg0, rt);
            svuint32_t gray3 = RgbaToGray(svld1_vnum_u8(mask3, rgba, 3), wrb, wg0, rt);
            svuint16_t gray01 = svuzp1_u16(svreinterpret_u16_u32(gray0), svreinterpret_u16_u32(gray1));
            svuint16_t gray23 = svuzp1_u16(svreinterpret_u16_u32(gray2), svreinterpret_u16_u32(gray3));
            svst1_u8(mask, gray, svuzp1_u8(svreinterpret_u8_u16(gray01), svreinterpret_u8_u16(gray23)));
        }

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail4 = (width - widthA) * 4;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svbool_t tail0 = svwhilelt_b8(0 * A, tail4);
            const svbool_t tail1 = svwhilelt_b8(1 * A, tail4);
            const svbool_t tail2 = svwhilelt_b8(2 * A, tail4);
            const svbool_t tail3 = svwhilelt_b8(3 * A, tail4);
            svuint16_t wrb = Set16u(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);
            svuint16_t wg0 = Set16u(Base::GREEN_TO_GRAY_WEIGHT, 0);
            svuint32_t rt = svdup_n_u32(Base::BGR_TO_GRAY_ROUND_TERM);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    RgbaToGray(rgba + offset, body, body, body, body, wrb, wg0, rt, gray + col, body);
                if (widthA < width)
                    RgbaToGray(rgba + offset, tail0, tail1, tail2, tail3, wrb, wg0, rt, gray + col, tail);
                rgba += rgbaStride;
                gray += grayStride;
            }
        }
    }
#endif
}
