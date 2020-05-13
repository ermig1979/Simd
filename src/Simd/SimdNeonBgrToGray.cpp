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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint8x8_t BgrToGray(uint8x8x3_t bgr)
        {
            return vmovn_u16(BgrToGray(vmovl_u8(bgr.val[0]), vmovl_u8(bgr.val[1]), vmovl_u8(bgr.val[2])));
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                {
                    uint8x8x3_t _bgr = LoadHalf3<align>(bgr + 3 * col);
                    Store<align>(gray + col, BgrToGray(_bgr));
                }
                if (alignedWidth != width)
                {
                    uint8x8x3_t _bgr = LoadHalf3<false>(bgr + 3 * (width - HA));
                    Store<false>(gray + width - HA, BgrToGray(_bgr));
                }
                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if (Aligned(bgr) && Aligned(gray) && Aligned(bgrStride) && Aligned(grayStride))
                BgrToGray<true>(bgr, width, height, bgrStride, gray, grayStride);
            else
                BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE uint8x8_t RgbToGray(uint8x8x3_t rgb)
        {
            return vmovn_u16(BgrToGray(vmovl_u8(rgb.val[2]), vmovl_u8(rgb.val[1]), vmovl_u8(rgb.val[0])));
        }

        template <bool align> void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(rgb) && Aligned(rgbStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                {
                    uint8x8x3_t _rgb = LoadHalf3<align>(rgb + 3 * col);
                    Store<align>(gray + col, RgbToGray(_rgb));
                }
                if (alignedWidth != width)
                {
                    uint8x8x3_t _rgb = LoadHalf3<false>(rgb + 3 * (width - HA));
                    Store<false>(gray + width - HA, RgbToGray(_rgb));
                }
                rgb += rgbStride;
                gray += grayStride;
            }
        }

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            if (Aligned(rgb) && Aligned(gray) && Aligned(rgbStride) && Aligned(grayStride))
                RgbToGray<true>(rgb, width, height, rgbStride, gray, grayStride);
            else
                RgbToGray<false>(rgb, width, height, rgbStride, gray, grayStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
