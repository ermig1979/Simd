/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i K16_BLUE_RED = SIMD_MM_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m128i K16_GREEN_0000 = SIMD_MM_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m128i K32_ROUND_TERM = SIMD_MM_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m128i BgraToGray32(__m128i bgra)
        {
            const __m128i g0a0 = _mm_and_si128(_mm_srli_si128(bgra, 1), K16_00FF);
            const __m128i b0r0 = _mm_and_si128(bgra, K16_00FF);
            const __m128i weightedSum = _mm_add_epi32(_mm_madd_epi16(g0a0, K16_GREEN_0000), _mm_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm_srli_epi32(_mm_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgraToGray(__m128i bgra[4])
        {
            const __m128i lo = _mm_packs_epi32(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m128i hi = _mm_packs_epi32(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return _mm_packus_epi16(lo, hi);
        }

        void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i a[4];
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const uint8_t * p = bgra + 4 * col;
                    a[0] = _mm_loadu_si128((__m128i*)p + 0);
                    a[1] = _mm_loadu_si128((__m128i*)p + 1);
                    a[2] = _mm_loadu_si128((__m128i*)p + 2);
                    a[3] = _mm_loadu_si128((__m128i*)p + 3);
                    _mm_storeu_si128((__m128i*)(gray + col), BgraToGray(a));
                }
                if (alignedWidth != width)
                {
                    const uint8_t * p = bgra + 4 * (width - A);
                    a[0] = _mm_loadu_si128((__m128i*)p + 0);
                    a[1] = _mm_loadu_si128((__m128i*)p + 1);
                    a[2] = _mm_loadu_si128((__m128i*)p + 2);
                    a[3] = _mm_loadu_si128((__m128i*)p + 3);
                    _mm_storeu_si128((__m128i*)(gray + width - A), BgraToGray(a));
                }
                bgra += bgraStride;
                gray += grayStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        const __m128i K16_RED_BLUE = SIMD_MM_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m128i RgbaToGray32(__m128i rgba)
        {
            const __m128i g0a0 = _mm_and_si128(_mm_srli_si128(rgba, 1), K16_00FF);
            const __m128i r0b0 = _mm_and_si128(rgba, K16_00FF);
            const __m128i weightedSum = _mm_add_epi32(_mm_madd_epi16(g0a0, K16_GREEN_0000), _mm_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm_srli_epi32(_mm_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i RgbaToGray(__m128i rgba[4])
        {
            const __m128i lo = _mm_packs_epi32(RgbaToGray32(rgba[0]), RgbaToGray32(rgba[1]));
            const __m128i hi = _mm_packs_epi32(RgbaToGray32(rgba[2]), RgbaToGray32(rgba[3]));
            return _mm_packus_epi16(lo, hi);
        }

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            if(width < A)
            {
                Base::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
                return;
            }

            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i a[4];
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const uint8_t* p = rgba + 4 * col;
                    a[0] = _mm_loadu_si128((__m128i*)p + 0);
                    a[1] = _mm_loadu_si128((__m128i*)p + 1);
                    a[2] = _mm_loadu_si128((__m128i*)p + 2);
                    a[3] = _mm_loadu_si128((__m128i*)p + 3);
                    _mm_storeu_si128((__m128i*)(gray + col), RgbaToGray(a));
                }
                if (alignedWidth != width)
                {
                    const uint8_t* p = rgba + 4 * (width - A);
                    a[0] = _mm_loadu_si128((__m128i*)p + 0);
                    a[1] = _mm_loadu_si128((__m128i*)p + 1);
                    a[2] = _mm_loadu_si128((__m128i*)p + 2);
                    a[3] = _mm_loadu_si128((__m128i*)p + 3);
                    _mm_storeu_si128((__m128i*)(gray + width - A), RgbaToGray(a));
                }
                rgba += rgbaStride;
                gray += grayStride;
            }
        }
    }
#endif
}
