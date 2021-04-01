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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K16_BLUE_RED = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m256i K16_GREEN_0000 = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m256i K32_ROUND_TERM = SIMD_MM256_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m256i BgraToGray32(__m256i bgra)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(bgra, 1), K16_00FF);
            const __m256i b0r0 = _mm256_and_si256(bgra, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_0000), _mm256_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm256_srli_epi32(_mm256_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgraToGray(__m256i bgra[4])
        {
            const __m256i lo = PackI32ToI16(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m256i hi = PackI32ToI16(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return PackI16ToU8(lo, hi);
        }

        template <bool align> SIMD_INLINE void Load(const uint8_t* p, __m256i a[4])
        {
            a[0] = Load<align>((__m256i*)p + 0);
            a[1] = Load<align>((__m256i*)p + 1);
            a[2] = Load<align>((__m256i*)p + 2);
            a[3] = Load<align>((__m256i*)p + 3);
        }

        template <bool align> void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            __m256i a[4];
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    Load<align>(bgra + 4 * col, a);
                    Store<align>((__m256i*)(gray + col), BgraToGray(a));
                }
                if (alignedWidth != width)
                {
                    Load<false>(bgra + 4 * (width - A), a);
                    Store<false>((__m256i*)(gray + width - A), BgraToGray(a));
                }
                bgra += bgraStride;
                gray += grayStride;
            }
        }

        void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                BgraToGray<true>(bgra, width, height, bgraStride, gray, grayStride);
            else
                BgraToGray<false>(bgra, width, height, bgraStride, gray, grayStride);
        }

        //---------------------------------------------------------------------

        const __m256i K16_RED_BLUE = SIMD_MM256_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m256i RgbaToGray32(__m256i rgba)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(rgba, 1), K16_00FF);
            const __m256i r0b0 = _mm256_and_si256(rgba, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_0000), _mm256_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm256_srli_epi32(_mm256_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i RgbaToGray(__m256i rgba[4])
        {
            const __m256i lo = PackI32ToI16(RgbaToGray32(rgba[0]), RgbaToGray32(rgba[1]));
            const __m256i hi = PackI32ToI16(RgbaToGray32(rgba[2]), RgbaToGray32(rgba[3]));
            return PackI16ToU8(lo, hi);
        }

        template <bool align> void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(rgba) && Aligned(rgbaStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            __m256i a[4];
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    Load<align>(rgba + 4 * col, a);
                    Store<align>((__m256i*)(gray + col), RgbaToGray(a));
                }
                if (alignedWidth != width)
                {
                    Load<false>(rgba + 4 * (width - A), a);
                    Store<false>((__m256i*)(gray + width - A), RgbaToGray(a));
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
#endif// SIMD_AVX2_ENABLE
}
