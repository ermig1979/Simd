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
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void Bgr48pToBgra32(uint8_t * bgra,
            const uint8_t * blue, const uint8_t * green, const uint8_t * red, size_t offset, __m128i alpha)
        {
            __m128i _blue = _mm_and_si128(Load<align>((__m128i*)(blue + offset)), K16_00FF);
            __m128i _green = _mm_and_si128(Load<align>((__m128i*)(green + offset)), K16_00FF);
            __m128i _red = _mm_and_si128(Load<align>((__m128i*)(red + offset)), K16_00FF);

            __m128i bg = _mm_or_si128(_blue, _mm_slli_si128(_green, 1));
            __m128i ra = _mm_or_si128(_red, alpha);

            Store<align>((__m128i*)bgra + 0, _mm_unpacklo_epi16(bg, ra));
            Store<align>((__m128i*)bgra + 1, _mm_unpackhi_epi16(bg, ra));
        }

        template <bool align> void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= HA);
            if (align)
            {
                assert(Aligned(blue) && Aligned(blueStride));
                assert(Aligned(green) && Aligned(greenStride));
                assert(Aligned(red) && Aligned(redStride));
                assert(Aligned(bgra) && Aligned(bgraStride));
            }

            __m128i _alpha = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, srcOffset = 0, dstOffset = 0; col < alignedWidth; col += HA, srcOffset += A, dstOffset += DA)
                    Bgr48pToBgra32<align>(bgra + dstOffset, blue, green, red, srcOffset, _alpha);
                if (width != alignedWidth)
                    Bgr48pToBgra32<false>(bgra + (width - HA) * 4, blue, green, red, (width - HA) * 2, _alpha);
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
    }
#endif//SIMD_SSE2_ENABLE
}
