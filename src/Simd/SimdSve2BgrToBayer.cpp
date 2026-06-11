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
        template <int c0, int c1> SIMD_INLINE void BgrToBayer(const uint8_t* bgr, uint8_t* bayer, const svbool_t& mask, const svbool_t& even)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svst1_u8(mask, bayer, svsel_u8(even, svget3(_bgr, c0), svget3(_bgr, c1)));
        }

        template <int c00, int c01, int c10, int c11> void BgrToBayer(const uint8_t* bgr, size_t width, size_t height,
            size_t bgrStride, uint8_t* bayer, size_t bayerStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svbool_t even = svcmpeq_n_u8(body, svand_n_u8_x(body, svindex_u8(0, 1), 1), 0);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToBayer<c00, c01>(bgr + offset, bayer + col, body, even);
                if (widthA < width)
                    BgrToBayer<c00, c01>(bgr + offset, bayer + col, tail, even);
                bgr += bgrStride;
                bayer += bayerStride;

                col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToBayer<c10, c11>(bgr + offset, bayer + col, body, even);
                if (widthA < width)
                    BgrToBayer<c10, c11>(bgr + offset, bayer + col, tail, even);
                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer,
            size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<1, 2, 0, 1>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<1, 0, 2, 1>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<2, 1, 1, 0>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<0, 1, 1, 2>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
