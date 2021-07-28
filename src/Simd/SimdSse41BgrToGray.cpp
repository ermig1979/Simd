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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE  
    namespace Sse41
    {
        const __m128i K16_BLUE_RED = SIMD_MM_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m128i K16_GREEN_ROUND = SIMD_MM_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m128i BgraToGray32(__m128i bgra)
        {
            const __m128i g0a0 = _mm_and_si128(_mm_srli_si128(bgra, 1), K16_00FF);
            const __m128i b0r0 = _mm_and_si128(bgra, K16_00FF);
            const __m128i weightedSum = _mm_add_epi32(_mm_madd_epi16(g0a0, K16_GREEN_ROUND), _mm_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm_srli_epi32(weightedSum, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgraToGray(__m128i bgra[4])
        {
            const __m128i lo = _mm_packs_epi32(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m128i hi = _mm_packs_epi32(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return _mm_packus_epi16(lo, hi);
        }

        template <bool align> SIMD_INLINE __m128i BgrToGray(const uint8_t * bgr, __m128i shuffle)
        {
            __m128i bgra[4];
            bgra[0] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<align>((__m128i*)(bgr + 0)), shuffle));
            bgra[1] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<false>((__m128i*)(bgr + 12)), shuffle));
            bgra[2] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<false>((__m128i*)(bgr + 24)), shuffle));
            bgra[3] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(_mm_srli_si128(Load<align>((__m128i*)(bgr + 32)), 4), shuffle));
            return BgraToGray(bgra);
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            __m128i _shuffle = _mm_setr_epi8(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Store<align>((__m128i*)(gray + col), BgrToGray<align>(bgr + 3 * col, _shuffle));
                if (width != alignedWidth)
                    Store<false>((__m128i*)(gray + width - A), BgrToGray<false>(bgr + 3 * (width - A), _shuffle));
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

        const __m128i K16_RED_BLUE = SIMD_MM_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m128i RgbaToGray32(__m128i rgba)
        {
            const __m128i g0a0 = _mm_and_si128(_mm_srli_si128(rgba, 1), K16_00FF);
            const __m128i r0b0 = _mm_and_si128(rgba, K16_00FF);
            const __m128i weightedSum = _mm_add_epi32(_mm_madd_epi16(g0a0, K16_GREEN_ROUND), _mm_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm_srli_epi32(weightedSum, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i RgbaToGray(__m128i rgba[4])
        {
            const __m128i lo = _mm_packs_epi32(RgbaToGray32(rgba[0]), RgbaToGray32(rgba[1]));
            const __m128i hi = _mm_packs_epi32(RgbaToGray32(rgba[2]), RgbaToGray32(rgba[3]));
            return _mm_packus_epi16(lo, hi);
        }

        template <bool align> SIMD_INLINE __m128i RgbToGray(const uint8_t* rgb, __m128i shuffle)
        {
            __m128i rgba[4];
            rgba[0] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<align>((__m128i*)(rgb + 0)), shuffle));
            rgba[1] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<false>((__m128i*)(rgb + 12)), shuffle));
            rgba[2] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(Load<false>((__m128i*)(rgb + 24)), shuffle));
            rgba[3] = _mm_or_si128(K32_01000000, _mm_shuffle_epi8(_mm_srli_si128(Load<align>((__m128i*)(rgb + 32)), 4), shuffle));
            return RgbaToGray(rgba);
        }

        template <bool align> void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);

            __m128i _shuffle = _mm_setr_epi8(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Store<align>((__m128i*)(gray + col), RgbToGray<align>(rgb + 3 * col, _shuffle));
                if (width != alignedWidth)
                    Store<false>((__m128i*)(gray + width - A), RgbToGray<false>(rgb + 3 * (width - A), _shuffle));
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
#endif
}
