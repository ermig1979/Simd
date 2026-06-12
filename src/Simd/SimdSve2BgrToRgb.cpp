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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void BgrToRgbTail(const uint8_t* bgr, uint8_t* rgb, const svbool_t& mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svst3_u8(mask, rgb, svcreate3_u8(svget3(_bgr, 2), svget3(_bgr, 1), svget3(_bgr, 0)));
        }

        SIMD_INLINE size_t BgrToRgbSrc(size_t offset)
        {
            return offset + 2 - 2 * (offset % 3);
        }

        SIMD_INLINE bool InitBgrToRgbIndex(uint8_t index[4][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                size_t src0 = BgrToRgbSrc(i);
                size_t src1 = BgrToRgbSrc(A + i);
                size_t src2 = BgrToRgbSrc(2 * A + i);
                index[0][i] = (uint8_t)src0;
                index[1][i] = src1 < 2 * A ? (uint8_t)src1 : 0xFF;
                index[2][i] = src1 >= A ? (uint8_t)(src1 - A) : 0xFF;
                index[3][i] = (uint8_t)(src2 - A);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BGR_TO_RGB_INDEX[4][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BGR_TO_RGB_INDEX_INITED = InitBgrToRgbIndex(BGR_TO_RGB_INDEX);

        SIMD_INLINE void BgrToRgb(const uint8_t* bgr, uint8_t* rgb, size_t A, const svuint8_t& index0,
            const svuint8_t& index10, const svuint8_t& index11, const svuint8_t& index2, const svbool_t& mask)
        {
            svuint8_t bgr0 = svld1_u8(mask, bgr + 0 * A);
            svuint8_t bgr1 = svld1_u8(mask, bgr + 1 * A);
            svuint8_t bgr2 = svld1_u8(mask, bgr + 2 * A);
            svuint8x2_t bgr01 = svcreate2_u8(bgr0, bgr1);
            svuint8x2_t bgr12 = svcreate2_u8(bgr1, bgr2);

            svst1_u8(mask, rgb + 0 * A, svtbl2_u8(bgr01, index0));
            svst1_u8(mask, rgb + 1 * A, svorr_u8_x(mask, svtbl2_u8(bgr01, index10), svtbl2_u8(bgr12, index11)));
            svst1_u8(mask, rgb + 2 * A, svtbl2_u8(bgr12, index2));
        }

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t index0 = svld1_u8(body, BGR_TO_RGB_INDEX[0]);
            const svuint8_t index10 = svld1_u8(body, BGR_TO_RGB_INDEX[1]);
            const svuint8_t index11 = svld1_u8(body, BGR_TO_RGB_INDEX[2]);
            const svuint8_t index2 = svld1_u8(body, BGR_TO_RGB_INDEX[3]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToRgb(bgr + offset, rgb + offset, A, index0, index10, index11, index2, body);
                if (widthA < width)
                    BgrToRgbTail(bgr + offset, rgb + offset, tail);
                bgr += bgrStride;
                rgb += rgbStride;
            }
        }
    }
#endif
}
