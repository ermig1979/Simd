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
        __m128i K8_SHUFFLE_GR_0 = SIMD_MM_SETR_EPI8(0x1, 0x5, 0x7, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_GR_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, 0xF, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_GR_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x3, 0x5, 0x9, 0xB, 0xF);

        __m128i K8_SHUFFLE_BG_0 = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x6, 0xA, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_BG_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x0, 0x2, 0x6, 0x8, 0xC, 0xE, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_BG_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE);

        __m128i K8_SHUFFLE_GB_0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x7, 0x9, 0xD, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_GB_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, 0x3, 0x5, 0x9, 0xB, 0xF, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_GB_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x5, 0x7, 0xB, 0xD);

        __m128i K8_SHUFFLE_RG_0 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_RG_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x0, 0x4, 0x6, 0xA, 0xC, -1, -1, -1, -1, -1, -1);
        __m128i K8_SHUFFLE_RG_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x2, 0x6, 0x8, 0xC, 0xE);

        template <int format, int row, bool align>
        SIMD_INLINE void BgrToBayer(const uint8_t * bgr, uint8_t * bayer, const __m128i shuffle[4][2][3])
        {
            const __m128i bayer0 = _mm_shuffle_epi8(Load<align>((__m128i*)bgr + 0), shuffle[format][row][0]);
            const __m128i bayer1 = _mm_shuffle_epi8(Load<align>((__m128i*)bgr + 1), shuffle[format][row][1]);
            const __m128i bayer2 = _mm_shuffle_epi8(Load<align>((__m128i*)bgr + 2), shuffle[format][row][2]);
            Store<align>((__m128i*)bayer, _mm_or_si128(_mm_or_si128(bayer0, bayer1), bayer2));
        }

        template <int format, bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const __m128i shuffle[4][2][3] =
            {
                {{K8_SHUFFLE_GR_0, K8_SHUFFLE_GR_1, K8_SHUFFLE_GR_2}, {K8_SHUFFLE_BG_0, K8_SHUFFLE_BG_1, K8_SHUFFLE_BG_2}},
                {{K8_SHUFFLE_GB_0, K8_SHUFFLE_GB_1, K8_SHUFFLE_GB_2}, {K8_SHUFFLE_RG_0, K8_SHUFFLE_RG_1, K8_SHUFFLE_RG_2}},
                {{K8_SHUFFLE_RG_0, K8_SHUFFLE_RG_1, K8_SHUFFLE_RG_2}, {K8_SHUFFLE_GB_0, K8_SHUFFLE_GB_1, K8_SHUFFLE_GB_2}},
                {{K8_SHUFFLE_BG_0, K8_SHUFFLE_BG_1, K8_SHUFFLE_BG_2}, {K8_SHUFFLE_GR_0, K8_SHUFFLE_GR_1, K8_SHUFFLE_GR_2}}
            };

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += 3 * A)
                    BgrToBayer<format, 0, align>(bgr + offset, bayer + col, shuffle);
                if (alignedWidth != width)
                    BgrToBayer<format, 0, false>(bgr + 3 * (width - A), bayer + width - A, shuffle);
                bgr += bgrStride;
                bayer += bayerStride;

                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += 3 * A)
                    BgrToBayer<format, 1, align>(bgr + offset, bayer + col, shuffle);
                if (alignedWidth != width)
                    BgrToBayer<format, 1, false>(bgr + 3 * (width - A), bayer + width - A, shuffle);
                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<0, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<1, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<2, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<3, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride))
                BgrToBayer<true>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
            else
                BgrToBayer<false>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif
}
