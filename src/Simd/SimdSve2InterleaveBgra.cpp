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
        SIMD_INLINE void InterleaveBgra(const uint8_t* b, const uint8_t* g, const uint8_t* r, const uint8_t* a, uint8_t* bgra,
            const svbool_t& load, const svbool_t& store0, const svbool_t& store1, const svbool_t& store2, const svbool_t& store3)
        {
            svuint8_t _b = svld1_u8(load, b);
            svuint8_t _g = svld1_u8(load, g);
            svuint8_t _r = svld1_u8(load, r);
            svuint8_t _a = svld1_u8(load, a);
            svuint8_t bg0 = svzip1_u8(_b, _g);
            svuint8_t bg1 = svzip2_u8(_b, _g);
            svuint8_t ra0 = svzip1_u8(_r, _a);
            svuint8_t ra1 = svzip2_u8(_r, _a);
            size_t A = svcntb();
            svst1_u8(store0, bgra + 0 * A, svreinterpret_u8_u16(svzip1_u16(svreinterpret_u16_u8(bg0), svreinterpret_u16_u8(ra0))));
            svst1_u8(store1, bgra + 1 * A, svreinterpret_u8_u16(svzip2_u16(svreinterpret_u16_u8(bg0), svreinterpret_u16_u8(ra0))));
            svst1_u8(store2, bgra + 2 * A, svreinterpret_u8_u16(svzip1_u16(svreinterpret_u16_u8(bg1), svreinterpret_u16_u8(ra1))));
            svst1_u8(store3, bgra + 3 * A, svreinterpret_u8_u16(svzip2_u16(svreinterpret_u16_u8(bg1), svreinterpret_u16_u8(ra1))));
        }

        void InterleaveBgra(const uint8_t* b, size_t bStride, const uint8_t* g, size_t gStride, const uint8_t* r, size_t rStride, const uint8_t* a, size_t aStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            size_t A = svcntb(), A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 4;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            const svbool_t tail2 = svwhilelt_b8(size_t(2) * A, tailSize);
            const svbool_t tail3 = svwhilelt_b8(size_t(3) * A, tailSize);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    InterleaveBgra(b + col, g + col, r + col, a + col, bgra + offset, body, body, body, body, body);
                if (widthA < width)
                    InterleaveBgra(b + col, g + col, r + col, a + col, bgra + offset, tail, tail0, tail1, tail2, tail3);
                b += bStride;
                g += gStride;
                r += rStride;
                a += aStride;
                bgra += bgraStride;
            }
        }
    }
#endif
}
