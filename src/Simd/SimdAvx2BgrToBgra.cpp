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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void BgrToBgra(const uint8_t * bgr, uint8_t * bgra, __m256i alpha)
        {
            Store<align>((__m256i*)bgra + 0, BgrToBgra<false>(Load<align>((__m256i*)(bgr + 0)), alpha));
            Store<align>((__m256i*)bgra + 1, BgrToBgra<false>(Load<false>((__m256i*)(bgr + 24)), alpha));
            Store<align>((__m256i*)bgra + 2, BgrToBgra<false>(Load<false>((__m256i*)(bgr + 48)), alpha));
            Store<align>((__m256i*)bgra + 3, BgrToBgra<true >(Load<align>((__m256i*)(bgr + 64)), alpha));
        }

        template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            __m256i _alpha = _mm256_slli_si256(_mm256_set1_epi32(alpha), 3);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BgrToBgra<align>(bgr + 3 * col, bgra + 4 * col, _alpha);
                if (width != alignedWidth)
                    BgrToBgra<false>(bgr + 3 * (width - A), bgra + 4 * (width - A), _alpha);
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
            const uint8_t * blue, const uint8_t * green, const uint8_t * red, size_t offset, __m256i alpha)
        {
            __m256i _blue = _mm256_and_si256(LoadPermuted<align>((__m256i*)(blue + offset)), K16_00FF);
            __m256i _green = _mm256_and_si256(LoadPermuted<align>((__m256i*)(green + offset)), K16_00FF);
            __m256i _red = _mm256_and_si256(LoadPermuted<align>((__m256i*)(red + offset)), K16_00FF);

            __m256i bg = _mm256_or_si256(_blue, _mm256_slli_si256(_green, 1));
            __m256i ra = _mm256_or_si256(_red, alpha);

            Store<align>((__m256i*)bgra + 0, _mm256_unpacklo_epi16(bg, ra));
            Store<align>((__m256i*)bgra + 1, _mm256_unpackhi_epi16(bg, ra));
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

            __m256i _alpha = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
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

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void RgbToBgra(const uint8_t* rgb, uint8_t* bgra, __m256i alpha)
        {
            Store<align>((__m256i*)bgra + 0, RgbToBgra<false>(Load<align>((__m256i*)(rgb + 0)), alpha));
            Store<align>((__m256i*)bgra + 1, RgbToBgra<false>(Load<false>((__m256i*)(rgb + 24)), alpha));
            Store<align>((__m256i*)bgra + 2, RgbToBgra<false>(Load<false>((__m256i*)(rgb + 48)), alpha));
            Store<align>((__m256i*)bgra + 3, RgbToBgra<true >(Load<align>((__m256i*)(rgb + 64)), alpha));
        }

        template <bool align> void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);

            __m256i _alpha = _mm256_slli_si256(_mm256_set1_epi32(alpha), 3);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    RgbToBgra<align>(rgb + 3 * col, bgra + 4 * col, _alpha);
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
#endif//SIMD_AVX2_ENABLE
}
