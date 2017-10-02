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
        SIMD_INLINE void BgrToBayer(const uint8_t * bgr, uint8_t * bayer)
        {
            uint8x16x3_t _bgr = Load3<align>(bgr);
            Store<align>(bayer, vbslq_u8((uint8x16_t)K16_00FF, _bgr.val[c0], _bgr.val[c1]));
        }

        template <int c00, int c01, int c10, int c11, bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);
            const size_t A3 = A * 3;

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += A3)
                    BgrToBayer<c00, c01, align>(bgr + offset, bayer + col);
                if (alignedWidth != width)
                    BgrToBayer<c00, c01, false>(bgr + 3 * (width - A), bayer + width - A);
                bgr += bgrStride;
                bayer += bayerStride;

                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += A3)
                    BgrToBayer<c10, c11, align>(bgr + offset, bayer + col);
                if (alignedWidth != width)
                    BgrToBayer<c10, c11, false>(bgr + 3 * (width - A), bayer + width - A);
                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<1, 2, 0, 1, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<1, 0, 2, 1, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<2, 1, 1, 0, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<0, 1, 1, 2, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride))
                BgrToBayer<true>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
            else
                BgrToBayer<false>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif// SIMD_NEON_ENABLE
}
