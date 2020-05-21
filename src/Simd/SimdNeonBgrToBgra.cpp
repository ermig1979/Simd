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

        union Bgra
        {
            uint8x16x4_t bgra;
            uint8x16x3_t bgr;
        };

        template <bool align> SIMD_INLINE void BgrToBgra(const uint8_t * bgr, uint8_t * bgra, Bgra & _bgra)
        {
            _bgra.bgr = Load3<align>(bgr);
            Store4<align>(bgra, _bgra.bgra);
        }

        template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            Bgra _bgra;
            _bgra.bgra.val[3] = vdupq_n_u8(alpha);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0, colBgr = 0; col < alignedWidth; col += A, colBgra += A4, colBgr += A3)
                    BgrToBgra<align>(bgr + colBgr, bgra + colBgra, _bgra);
                if (width != alignedWidth)
                    BgrToBgra<false>(bgr + 3 * (width - A), bgra + 4 * (width - A), _bgra);
                bgr += bgrStride;
                bgra += bgraStride;
            }
        }

        void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToBgra<true>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
            else
                BgrToBgra<false>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void Bgr48pToBgra32(uint8_t * bgra,
            const uint8_t * blue, const uint8_t * green, const uint8_t * red, size_t offset, const uint8x16_t & alpha)
        {
            uint8x16x2_t _blue = Load2<align>(blue + offset);
            uint8x16x2_t _green = Load2<align>(green + offset);
            uint8x16x2_t _red = Load2<align>(red + offset);

            uint8x16x4_t _bgra;
            _bgra.val[0] = _blue.val[0];
            _bgra.val[1] = _green.val[0];
            _bgra.val[2] = _red.val[0];
            _bgra.val[3] = alpha;

            Store4<align>(bgra, _bgra);
        }

        template <bool align> void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(blue) && Aligned(blueStride));
                assert(Aligned(green) && Aligned(greenStride));
                assert(Aligned(red) && Aligned(redStride));
                assert(Aligned(bgra) && Aligned(bgraStride));
            }

            uint8x16_t _alpha;
            _alpha = vdupq_n_u8(alpha);

            size_t alignedWidth = AlignLo(width, A) * 2;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t srcOffset = 0, dstOffset = 0; srcOffset < alignedWidth; srcOffset += DA, dstOffset += QA)
                    Bgr48pToBgra32<align>(bgra + dstOffset, blue, green, red, srcOffset, _alpha);
                if (width != alignedWidth)
                    Bgr48pToBgra32<false>(bgra + (width - A) * 4, blue, green, red, (width - A) * 2, _alpha);
                blue += blueStride;
                green += greenStride;
                red += redStride;
                bgra += bgraStride;
            }
        }

        void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(blue) && Aligned(blueStride) && Aligned(green) && Aligned(greenStride) &&
                Aligned(red) && Aligned(redStride) && Aligned(bgra) && Aligned(bgraStride))
                Bgr48pToBgra32<true>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
            else
                Bgr48pToBgra32<false>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void RgbToBgra(const uint8_t* rgb, uint8_t* bgra, uint8x16_t alpha)
        {
            uint8x16x3_t _rgb = Load3<align>(rgb);
            uint8x16x4_t _bgra;
            _bgra.val[0] = _rgb.val[2];
            _bgra.val[1] = _rgb.val[1];
            _bgra.val[2] = _rgb.val[0];
            _bgra.val[3] = alpha;
            Store4<align>(bgra, _bgra);
        }

        template <bool align> void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t _alpha = vdupq_n_u8(alpha);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0, colRgb = 0; col < alignedWidth; col += A, colBgra += A4, colRgb += A3)
                    RgbToBgra<align>(rgb + colRgb, bgra + colBgra, _alpha);
                if (width != alignedWidth)
                    RgbToBgra<false>(rgb + 3 * (width - A), bgra + 4 * (width - A), _alpha);
                rgb += rgbStride;
                bgra += bgraStride;
            }
        }

        void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride))
                RgbToBgra<true>(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
            else
                RgbToBgra<false>(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_NEON_ENABLE
}
