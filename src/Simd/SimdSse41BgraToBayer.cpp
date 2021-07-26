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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE  
    namespace Sse41
    {
        __m128i K8_SHUFFLE_GR = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1, 0xE, -1, -1, -1);
        __m128i K8_SHUFFLE_BG = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xD, -1, -1, -1);
        __m128i K8_SHUFFLE_GB = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x9, -1, -1, -1, 0xC, -1, -1, -1);
        __m128i K8_SHUFFLE_RG = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x5, -1, -1, -1, 0xA, -1, -1, -1, 0xD, -1, -1, -1);

        template <int format, int row, bool align>
        SIMD_INLINE void BgraToBayer(const uint8_t * bgra, uint8_t * bayer, const __m128i shuffle[4][2])
        {
            const __m128i bayer0 = _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 0), shuffle[format][row]);
            const __m128i bayer1 = _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 1), shuffle[format][row]);
            const __m128i bayer2 = _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 2), shuffle[format][row]);
            const __m128i bayer3 = _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 3), shuffle[format][row]);
            Store<align>((__m128i*)bayer, _mm_packus_epi16(_mm_packs_epi32(bayer0, bayer1), _mm_packs_epi32(bayer2, bayer3)));
        }

        template <int format, bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const __m128i shuffle[4][2] =
            {
                {K8_SHUFFLE_GR, K8_SHUFFLE_BG},
                {K8_SHUFFLE_GB, K8_SHUFFLE_RG},
                {K8_SHUFFLE_RG, K8_SHUFFLE_GB},
                {K8_SHUFFLE_BG, K8_SHUFFLE_GR}
            };

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += QA)
                    BgraToBayer<format, 0, align>(bgra + offset, bayer + col, shuffle);
                if (alignedWidth != width)
                    BgraToBayer<format, 0, false>(bgra + 4 * (width - A), bayer + width - A, shuffle);
                bgra += bgraStride;
                bayer += bayerStride;

                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += QA)
                    BgraToBayer<format, 1, align>(bgra + offset, bayer + col, shuffle);
                if (alignedWidth != width)
                    BgraToBayer<format, 1, false>(bgra + 4 * (width - A), bayer + width - A, shuffle);
                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<0, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<1, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<2, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<3, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride))
                BgraToBayer<true>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
            else
                BgraToBayer<false>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif
}
