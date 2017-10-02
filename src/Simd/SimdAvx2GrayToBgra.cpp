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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void GrayToBgra(uint8_t * bgra, __m256i gray, __m256i alpha)
        {
            __m256i bgLo = _mm256_unpacklo_epi8(gray, gray);
            __m256i bgHi = _mm256_unpackhi_epi8(gray, gray);
            __m256i raLo = _mm256_unpacklo_epi8(gray, alpha);
            __m256i raHi = _mm256_unpackhi_epi8(gray, alpha);

            Store<align>((__m256i*)bgra + 0, _mm256_unpacklo_epi16(bgLo, raLo));
            Store<align>((__m256i*)bgra + 1, _mm256_unpackhi_epi16(bgLo, raLo));
            Store<align>((__m256i*)bgra + 2, _mm256_unpacklo_epi16(bgHi, raHi));
            Store<align>((__m256i*)bgra + 3, _mm256_unpackhi_epi16(bgHi, raHi));
        }

        template <bool align> void GrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            __m256i _alpha = _mm256_set1_epi8(alpha);
            __m256i permuteOffsets = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _gray = _mm256_permutevar8x32_epi32(Load<align>((__m256i*)(gray + col)), permuteOffsets);
                    GrayToBgra<align>(bgra + 4 * col, _gray, _alpha);
                }
                if (alignedWidth != width)
                {
                    __m256i _gray = _mm256_permutevar8x32_epi32(Load<false>((__m256i*)(gray + width - A)), permuteOffsets);
                    GrayToBgra<false>(bgra + 4 * (width - A), _gray, _alpha);
                }
                gray += grayStride;
                bgra += bgraStride;
            }
        }

        void GrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                GrayToBgra<true>(gray, width, height, grayStride, bgra, bgraStride, alpha);
            else
                GrayToBgra<false>(gray, width, height, grayStride, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
