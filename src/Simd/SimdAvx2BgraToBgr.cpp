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
#ifdef SIMD_AVX2_ENABLE  
    namespace Avx2
    {
        const __m256i K8_SUFFLE_BGRA_TO_BGR = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        const __m256i K32_PERMUTE_BGRA_TO_BGR = SIMD_MM256_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, -1, -1);

        template <bool align> SIMD_INLINE __m256i BgraToBgr(const uint8_t* bgra)
        {
            __m256i _bgra = Load<align>((__m256i*)bgra);
            return _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_bgra, K8_SUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR);
        }

        template <bool align> void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t widthF = AlignLo(width, F);
            if (width == widthF)
                widthF -= F;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthF; col += F)
                    Store<false>((__m256i*)(bgr + 3 * col), BgraToBgr<align>(bgra + 4 * col));
                if (width != widthF)
                    Store24<false>(bgr + 3 * (width - F), BgraToBgr<false>(bgra + 4 * (width - F)));
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

        const __m256i K8_SUFFLE_BGRA_TO_RGB = SIMD_MM256_SETR_EPI8(
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1);

        template <bool align> SIMD_INLINE __m256i BgraToRgb(const uint8_t* bgra)
        {
            __m256i _bgra = Load<align>((__m256i*)bgra);
            return _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_bgra, K8_SUFFLE_BGRA_TO_RGB), K32_PERMUTE_BGRA_TO_BGR);
        }

        template <bool align> void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t widthF = AlignLo(width, F);
            if (width == widthF)
                widthF -= F;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthF; col += F)
                    Store<false>((__m256i*)(rgb + 3 * col), BgraToRgb<align>(bgra + 4 * col));
                if (width != widthF)
                    Store24<false>(rgb + 3 * (width - F), BgraToRgb<false>(bgra + 4 * (width - F)));
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
#endif// SIMD_AVX2_ENABLE
}
