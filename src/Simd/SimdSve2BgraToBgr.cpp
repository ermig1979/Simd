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
        SIMD_INLINE void BgraToBgrTail(const uint8_t* bgra, uint8_t* bgr, const svbool_t& mask)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svst3_u8(mask, bgr, svcreate3_u8(svget4(_bgra, 0), svget4(_bgra, 1), svget4(_bgra, 2)));
        }

        SIMD_INLINE void InitBgraToBgrIndex(size_t A, uint8_t index[3][64])
        {
            for (size_t k = 0; k < 3; ++k)
            {
                size_t dst = k * A;
                size_t src = k * A;
                for (size_t i = 0; i < A; ++i)
                {
                    size_t offset = dst + i;
                    index[k][i] = (uint8_t)(4 * (offset / 3) + offset % 3 - src);
                }
            }
        }

        SIMD_INLINE void BgraToBgr(const uint8_t* bgra, uint8_t* bgr, size_t A,
            const svuint8_t& index0, const svuint8_t& index1, const svuint8_t& index2, const svbool_t& mask)
        {
            svuint8_t bgra0 = svld1_u8(mask, bgra + 0 * A);
            svuint8_t bgra1 = svld1_u8(mask, bgra + 1 * A);
            svuint8_t bgra2 = svld1_u8(mask, bgra + 2 * A);
            svuint8_t bgra3 = svld1_u8(mask, bgra + 3 * A);

            svst1_u8(mask, bgr + 0 * A, svtbl2_u8(svcreate2_u8(bgra0, bgra1), index0));
            svst1_u8(mask, bgr + 1 * A, svtbl2_u8(svcreate2_u8(bgra1, bgra2), index1));
            svst1_u8(mask, bgr + 2 * A, svtbl2_u8(svcreate2_u8(bgra2, bgra3), index2));
        }

        void BgraToBgr(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bgr, size_t bgrStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3, A4 = A * 4;
            assert(A <= 64);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            SIMD_ALIGNED(64) uint8_t _index[3][64];
            InitBgraToBgrIndex(A, _index);
            const svuint8_t index0 = svld1_u8(body, _index[0]);
            const svuint8_t index1 = svld1_u8(body, _index[1]);
            const svuint8_t index2 = svld1_u8(body, _index[2]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, bgraOffset = 0, bgrOffset = 0;
                for (; col < widthA; col += A, bgraOffset += A4, bgrOffset += A3)
                    BgraToBgr(bgra + bgraOffset, bgr + bgrOffset, A, index0, index1, index2, body);
                if (widthA < width)
                    BgraToBgrTail(bgra + bgraOffset, bgr + bgrOffset, tail);
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BgraToRgb(const uint8_t* bgra, uint8_t* rgb, const svbool_t& mask)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svst3_u8(mask, rgb, svcreate3_u8(svget4(_bgra, 2), svget4(_bgra, 1), svget4(_bgra, 0)));
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3, A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, bgraOffset = 0, rgbOffset = 0;
                for (; col < widthA; col += A, bgraOffset += A4, rgbOffset += A3)
                    BgraToRgb(bgra + bgraOffset, rgb + rgbOffset, body);
                if (widthA < width)
                    BgraToRgb(bgra + bgraOffset, rgb + rgbOffset, tail);
                bgra += bgraStride;
                rgb += rgbStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_ALIGNED(64) const uint8_t BGRA_TO_RGBA_INDEX[64] = {
            0x02, 0x01, 0x00, 0x03, 0x06, 0x05, 0x04, 0x07, 0x0A, 0x09, 0x08, 0x0B, 0x0E, 0x0D, 0x0C, 0x0F, 
            0x12, 0x11, 0x10, 0x13, 0x16, 0x15, 0x14, 0x17, 0x1A, 0x19, 0x18, 0x1B, 0x1E, 0x1D, 0x1C, 0x1F,
            0x22, 0x21, 0x20, 0x23, 0x26, 0x25, 0x24, 0x27, 0x2A, 0x29, 0x28, 0x2B, 0x2E, 0x2D, 0x2C, 0x2F,
            0x32, 0x31, 0x30, 0x33, 0x36, 0x35, 0x34, 0x37, 0x3A, 0x39, 0x38, 0x3B, 0x3E, 0x3D, 0x3C, 0x3F
        };

        SIMD_INLINE void BgraToRgba(const uint8_t* bgra, uint8_t* rgba, const svuint8_t& index, const svbool_t& mask)
        {
            svuint8_t _bgra = svld1_u8(mask, bgra);
            svuint8_t _rgba = svtbl_u8(_bgra, index);
            svst1_u8(mask, rgba, _rgba);
        }

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            size_t A = svlen(svuint8_t());
            assert(A <= 64);
            size_t size = width*4, sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(sizeA, size);
            const svuint8_t index = svld1_u8(body, BGRA_TO_RGBA_INDEX);
            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA; i += A)
                    BgraToRgba(bgra + i, rgba + i, index, body);
                if (i < size)
                    BgraToRgba(bgra + i, rgba + i, index, tail);
                bgra += bgraStride;
                rgba += rgbaStride;
            }
        }
    }
#endif
}
