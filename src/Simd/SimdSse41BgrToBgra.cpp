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
        template <bool align> SIMD_INLINE void BgrToBgra(const uint8_t * bgr, uint8_t * bgra, __m128i alpha, __m128i shuffle)
        {
            Store<align>((__m128i*)bgra + 0, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<align>((__m128i*)(bgr + 0)), shuffle)));
            Store<align>((__m128i*)bgra + 1, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<false>((__m128i*)(bgr + 12)), shuffle)));
            Store<align>((__m128i*)bgra + 2, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<false>((__m128i*)(bgr + 24)), shuffle)));
            Store<align>((__m128i*)bgra + 3, _mm_or_si128(alpha, _mm_shuffle_epi8(_mm_srli_si128(Load<align>((__m128i*)(bgr + 32)), 4), shuffle)));
        }

        template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            __m128i _alpha = _mm_slli_si128(_mm_set1_epi32(alpha), 3);
            __m128i _shuffle = _mm_setr_epi8(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BgrToBgra<align>(bgr + 3 * col, bgra + 4 * col, _alpha, _shuffle);
                if (width != alignedWidth)
                    BgrToBgra<false>(bgr + 3 * (width - A), bgra + 4 * (width - A), _alpha, _shuffle);
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

        template <bool align> SIMD_INLINE void RgbToBgra(const uint8_t* rgb, uint8_t* bgra, __m128i alpha, __m128i shuffle)
        {
            Store<align>((__m128i*)bgra + 0, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<align>((__m128i*)(rgb + 0)), shuffle)));
            Store<align>((__m128i*)bgra + 1, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<false>((__m128i*)(rgb + 12)), shuffle)));
            Store<align>((__m128i*)bgra + 2, _mm_or_si128(alpha, _mm_shuffle_epi8(Load<false>((__m128i*)(rgb + 24)), shuffle)));
            Store<align>((__m128i*)bgra + 3, _mm_or_si128(alpha, _mm_shuffle_epi8(_mm_srli_si128(Load<align>((__m128i*)(rgb + 32)), 4), shuffle)));
        }

        template <bool align> void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);

            __m128i _alpha = _mm_slli_si128(_mm_set1_epi32(alpha), 3);
            __m128i _shuffle = _mm_setr_epi8(0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    RgbToBgra<align>(rgb + 3 * col, bgra + 4 * col, _alpha, _shuffle);
                if (width != alignedWidth)
                    RgbToBgra<false>(rgb + 3 * (width - A), bgra + 4 * (width - A), _alpha, _shuffle);
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
#endif
}
