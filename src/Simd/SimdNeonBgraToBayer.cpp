/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        template <int c0, int c1, bool align>
        SIMD_INLINE void BgraToBayer(const uint8_t * bgra, uint8_t * bayer)
        {
            uint8x16x4_t _bgra = Load4<align>(bgra);
            Store<align>(bayer, vbslq_u8((uint8x16_t)K16_00FF, _bgra.val[c0], _bgra.val[c1]));
        }

        template <int c00, int c01, int c10, int c11, bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += QA)
                    BgraToBayer<c00, c01, align>(bgra + offset, bayer + col);
                if (alignedWidth != width)
                    BgraToBayer<c00, c01, false>(bgra + 4 * (width - A), bayer + width - A);
                bgra += bgraStride;
                bayer += bayerStride;

                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += QA)
                    BgraToBayer<c10, c11, align>(bgra + offset, bayer + col);
                if (alignedWidth != width)
                    BgraToBayer<c10, c11, false>(bgra + 4 * (width - A), bayer + width - A);
                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<1, 2, 0, 1, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<1, 0, 2, 1, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<2, 1, 1, 0, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<0, 1, 1, 2, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride))
                BgraToBayer<true>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
            else
                BgraToBayer<false>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif// SIMD_NEON_ENABLE
}
