/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
        SIMD_INLINE uint8x8_t BgraToGray(uint8x8x4_t bgra)
        {
            return vmovn_u16(BgrToGray(vmovl_u8(bgra.val[0]), vmovl_u8(bgra.val[1]), vmovl_u8(bgra.val[2])));
        }

        template <bool align> void BgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                {
                    uint8x8x4_t _bgra = LoadHalf4<align>(bgra + 4 * col);
                    Store<align>(gray + col, BgraToGray(_bgra));
                }
                if (alignedWidth != width)
                {
                    uint8x8x4_t _bgra = LoadHalf4<false>(bgra + 4 * (width - HA));
                    Store<false>(gray + width - HA, BgraToGray(_bgra));
                }
                bgra += bgraStride;
                gray += grayStride;
            }
        }

        void BgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                BgraToGray<true>(bgra, width, height, bgraStride, gray, grayStride);
            else
                BgraToGray<false>(bgra, width, height, bgraStride, gray, grayStride);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE uint8x8_t RgbaToGray(uint8x8x4_t rgba)
        {
            return vmovn_u16(BgrToGray(vmovl_u8(rgba.val[2]), vmovl_u8(rgba.val[1]), vmovl_u8(rgba.val[0])));
        }

        template <bool align> void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(rgba) && Aligned(rgbaStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                {
                    uint8x8x4_t _rgba = LoadHalf4<align>(rgba + 4 * col);
                    Store<align>(gray + col, RgbaToGray(_rgba));
                }
                if (alignedWidth != width)
                {
                    uint8x8x4_t _rgba = LoadHalf4<false>(rgba + 4 * (width - HA));
                    Store<false>(gray + width - HA, RgbaToGray(_rgba));
                }
                rgba += rgbaStride;
                gray += grayStride;
            }
        }

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            if (Aligned(rgba) && Aligned(gray) && Aligned(rgbaStride) && Aligned(grayStride))
                RgbaToGray<true>(rgba, width, height, rgbaStride, gray, grayStride);
            else
                RgbaToGray<false>(rgba, width, height, rgbaStride, gray, grayStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
