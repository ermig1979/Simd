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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K16_BLUE_RED = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m256i K16_GREEN_ROUND = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m256i BgraToGray32(__m256i bgra)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(bgra, 1), K16_00FF);
            const __m256i b0r0 = _mm256_and_si256(bgra, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_ROUND), _mm256_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm256_srli_epi32(weightedSum, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgraToGray(__m256i bgra[4])
        {
            const __m256i lo = PackI32ToI16(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m256i hi = PackI32ToI16(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return PackI16ToU8(lo, hi);
        }

        template <bool align> SIMD_INLINE __m256i BgrToGray(const uint8_t * bgr)
        {
            __m256i bgra[4];
            bgra[0] = BgrToBgra<false>(Load<align>((__m256i*)(bgr + 0)), K32_01000000);
            bgra[1] = BgrToBgra<false>(Load<false>((__m256i*)(bgr + 24)), K32_01000000);
            bgra[2] = BgrToBgra<false>(Load<false>((__m256i*)(bgr + 48)), K32_01000000);
            bgra[3] = BgrToBgra<true>(Load<align>((__m256i*)(bgr + 64)), K32_01000000);
            return BgraToGray(bgra);
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Store<align>((__m256i*)(gray + col), BgrToGray<align>(bgr + 3 * col));
                if (width != alignedWidth)
                    Store<false>((__m256i*)(gray + width - A), BgrToGray<false>(bgr + 3 * (width - A)));
                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if (Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToGray<true>(bgr, width, height, bgrStride, gray, grayStride);
            else
                BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }


        //---------------------------------------------------------------------

        const __m256i K16_RED_BLUE = SIMD_MM256_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m256i RgbaToGray32(__m256i rgba)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(rgba, 1), K16_00FF);
            const __m256i r0b0 = _mm256_and_si256(rgba, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_ROUND), _mm256_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm256_srli_epi32(weightedSum, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i RgbaToGray(__m256i rgba[4])
        {
            const __m256i lo = PackI32ToI16(RgbaToGray32(rgba[0]), RgbaToGray32(rgba[1]));
            const __m256i hi = PackI32ToI16(RgbaToGray32(rgba[2]), RgbaToGray32(rgba[3]));
            return PackI16ToU8(lo, hi);
        }

        template <bool align> SIMD_INLINE __m256i RgbToGray(const uint8_t* rgb)
        {
            __m256i rgba[4];
            rgba[0] = BgrToBgra<false>(Load<align>((__m256i*)(rgb + 0)), K32_01000000);
            rgba[1] = BgrToBgra<false>(Load<false>((__m256i*)(rgb + 24)), K32_01000000);
            rgba[2] = BgrToBgra<false>(Load<false>((__m256i*)(rgb + 48)), K32_01000000);
            rgba[3] = BgrToBgra<true>(Load<align>((__m256i*)(rgb + 64)), K32_01000000);
            return RgbaToGray(rgba);
        }

        template <bool align> void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Store<align>((__m256i*)(gray + col), RgbToGray<align>(rgb + 3 * col));
                if (width != alignedWidth)
                    Store<false>((__m256i*)(gray + width - A), RgbToGray<false>(rgb + 3 * (width - A)));
                rgb += rgbStride;
                gray += grayStride;
            }
        }

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            if (Aligned(gray) && Aligned(grayStride) && Aligned(rgb) && Aligned(rgbStride))
                RgbToGray<true>(rgb, width, height, rgbStride, gray, grayStride);
            else
                RgbToGray<false>(rgb, width, height, rgbStride, gray, grayStride);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
