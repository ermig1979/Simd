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
        template <int c0, int c1> SIMD_INLINE void BgraToBayer(const uint8_t* bgra, uint8_t* bayer, const svbool_t& mask, const svbool_t& even)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svst1_u8(mask, bayer, svsel_u8(even, svget4(_bgra, c0), svget4(_bgra, c1)));
        }

        template <int c00, int c01, int c10, int c11> void BgraToBayer(const uint8_t* bgra, size_t width, size_t height,
            size_t bgraStride, uint8_t* bayer, size_t bayerStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svbool_t even = svcmpeq_n_u8(body, svand_n_u8_x(body, svindex_u8(0, 1), 1), 0);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    BgraToBayer<c00, c01>(bgra + offset, bayer + col, body, even);
                if (widthA < width)
                    BgraToBayer<c00, c01>(bgra + offset, bayer + col, tail, even);
                bgra += bgraStride;
                bayer += bayerStride;

                col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    BgraToBayer<c10, c11>(bgra + offset, bayer + col, body, even);
                if (widthA < width)
                    BgraToBayer<c10, c11>(bgra + offset, bayer + col, tail, even);
                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        void BgraToBayer(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bayer,
            size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<1, 2, 0, 1>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<1, 0, 2, 1>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<2, 1, 1, 0>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<0, 1, 1, 2>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
