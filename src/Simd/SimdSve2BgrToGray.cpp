/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
            return svshrnt_n_u32(svshrnb_n_u32(bGray, 14), tGray, 14);
        }

        SIMD_INLINE void BgrToGray(const uint8_t* bgr, const svuint16_t& wb, const svuint16_t& wg, const svuint16_t& wr, const svuint32_t& rt, uint8_t* gray, const svbool_t& mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svuint16_t bGray = BgrToGray(svmovlb_u16(svget3(_bgr, 0)), svmovlb_u16(svget3(_bgr, 1)), svmovlb_u16(svget3(_bgr, 2)), wb, wg, wr, rt);
            svuint16_t tGray = BgrToGray(svmovlt_u16(svget3(_bgr, 0)), svmovlt_u16(svget3(_bgr, 1)), svmovlt_u16(svget3(_bgr, 2)), wb, wg, wr, rt);
            svst1_u8(mask, gray, svqxtnt_u16(svqxtnb_u16(bGray), tGray));
        }

        void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* gray, size_t grayStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A), size = width * 3;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint16_t wb = svdup_n_u16(Base::BLUE_TO_GRAY_WEIGHT);
            svuint16_t wg = svdup_n_u16(Base::GREEN_TO_GRAY_WEIGHT);
            svuint16_t wr = svdup_n_u16(Base::RED_TO_GRAY_WEIGHT);
            svuint32_t rt = svdup_n_u32(Base::BGR_TO_GRAY_ROUND_TERM);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToGray(bgr + offset, wb, wg, wr, rt, gray + col, body);
                if (widthA < width)
                    BgrToGray(bgr + offset, wb, wg, wr, rt, gray + col, tail);
                bgr += bgrStride;
                gray += grayStride;
            }
        }
    }
#endif
}
