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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        const size_t A3 = A * 3;
        const size_t A4 = A * 4;

        template <bool align> SIMD_INLINE void BgraToBgr(const uint8_t * bgra, uint8_t * bgr)
        {
            uint8x16x4_t _bgra = Load4<align>(bgra);
            Store3<align>(bgr, *(uint8x16x3_t*)&_bgra);
        }

        template <bool align> void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);
            if (width == alignedWidth)
                alignedWidth -= A;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0, colBgr = 0; col < alignedWidth; col += A, colBgra += A4, colBgr += A3)
                    BgraToBgr<align>(bgra + colBgra, bgr + colBgr);
                if (width != alignedWidth)
                    BgraToBgr<false>(bgra + 4 * (width - A), bgr + 3 * (width - A));
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
                BgraToBgr<true>(bgra, width, height, bgraStride, bgr, bgrStride);
            else
                BgraToBgr<false>(bgra, width, height, bgraStride, bgr, bgrStride);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void BgraToRgb(const uint8_t* bgra, uint8_t* rgb)
        {
            uint8x16x4_t _bgra = Load4<align>(bgra);
            uint8x16x3_t _rgb;
            _rgb.val[0] = _bgra.val[2];
            _rgb.val[1] = _bgra.val[1];
            _rgb.val[2] = _bgra.val[0];
            Store3<align>(rgb, _rgb);
        }

        template <bool align> void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);
            if (width == alignedWidth)
                alignedWidth -= A;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0, colRgb = 0; col < alignedWidth; col += A, colBgra += A4, colRgb += A3)
                    BgraToRgb<align>(bgra + colBgra, rgb + colRgb);
                if (width != alignedWidth)
                    BgraToRgb<false>(bgra + 4 * (width - A), rgb + 3 * (width - A));
                bgra += bgraStride;
                rgb += rgbStride;
            }
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride))
                BgraToRgb<true>(bgra, width, height, bgraStride, rgb, rgbStride);
            else
                BgraToRgb<false>(bgra, width, height, bgraStride, rgb, rgbStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
