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
        SIMD_INLINE void BgrToBgra(const uint8_t* bgr, uint8_t* bgra, const svuint8_t& alpha, const svbool_t& mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svst4_u8(mask, bgra, svcreate4_u8(svget3(_bgr, 0), svget3(_bgr, 1), svget3(_bgr, 2), alpha));
        }

        void BgrToBgra(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3, A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, bgrOffset = 0, bgraOffset = 0;
                for (; col < widthA; col += A, bgrOffset += A3, bgraOffset += A4)
                    BgrToBgra(bgr + bgrOffset, bgra + bgraOffset, _alpha, body);
                if (widthA < width)
                    BgrToBgra(bgr + bgrOffset, bgra + bgraOffset, _alpha, tail);
                bgr += bgrStride;
                bgra += bgraStride;
            }
        }

        SIMD_INLINE void RgbToBgra(const uint8_t* rgb, uint8_t* bgra, const svuint8_t& alpha, const svbool_t& mask)
        {
            svuint8x3_t _rgb = svld3_u8(mask, rgb);
            svst4_u8(mask, bgra, svcreate4_u8(svget3(_rgb, 2), svget3(_rgb, 1), svget3(_rgb, 0), alpha));
        }

        void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3, A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, rgbOffset = 0, bgraOffset = 0;
                for (; col < widthA; col += A, rgbOffset += A3, bgraOffset += A4)
                    RgbToBgra(rgb + rgbOffset, bgra + bgraOffset, _alpha, body);
                if (widthA < width)
                    RgbToBgra(rgb + rgbOffset, bgra + bgraOffset, _alpha, tail);
                rgb += rgbStride;
                bgra += bgraStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void Bgr48pToBgra32(const uint8_t* blue, const uint8_t* green, const uint8_t* red, const svbool_t& mask, 
            const svuint8_t& alpha, uint8_t* bgra, const svbool_t& mask0, const svbool_t& mask1)
        {
            svuint8_t _blue = svld1_u8(mask, blue);
            svuint8_t _green = svld1_u8(mask, green);
            svuint8_t _red = svld1_u8(mask, red);
            svuint8_t br = svqxtnt_u16(_blue, svmovlb_u16(_red));
            svuint8_t ga = svqxtnt_u16(_green, svreinterpret_u16_u8(alpha));
            svst1_vnum_u8(mask0, bgra, 0, svzip1_u8(br, ga));
            svst1_vnum_u8(mask1, bgra, 1, svzip2_u8(br, ga));
        }

        void Bgr48pToBgra32(const uint8_t* blue, size_t blueStride, size_t width, size_t height,
            const uint8_t* green, size_t greenStride, const uint8_t* red, size_t redStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            size_t A = svlen(svuint8_t()), A2 = A * 2;
            size_t size = width * 2, sizeA = AlignLo(size, A), tail2 = (size - sizeA) * 2;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(sizeA, size);
            const svbool_t tail0 = svwhilelt_b8(0 * A, tail2);
            const svbool_t tail1 = svwhilelt_b8(1 * A, tail2);
            const svuint8_t _alpha = svreinterpret_u8_u16(svdup_n_u16(alpha));
            for (size_t row = 0; row < height; ++row)
            {
                size_t cs = 0, dc = 0;
                for (; cs < sizeA; cs += A, dc += A2)
                    Bgr48pToBgra32(blue + cs, green + cs, red + cs, body, _alpha, bgra + dc, body, body);
                if (sizeA < size)
                    Bgr48pToBgra32(blue + cs, green + cs, red + cs, tail, _alpha, bgra + dc, tail0, tail1);
                blue += blueStride;
                green += greenStride;
                red += redStride;
                bgra += bgraStride;
            }
        }
    }
#endif
}
