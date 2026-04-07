/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        // Shuffle masks selecting Bayer bytes from a 16-byte BGR chunk.
        // Three masks per pattern cover the three possible BGR-chunk offsets (mod 3).
        // Offset 0: chunk starts at B component (B0 G0 R0 B1 G1 R1 ...)
        // Offset 1: chunk starts at G component (G5 R5 B6 G6 R6 ...)
        // Offset 2: chunk starts at R component (R10 B11 G11 R11 ...)

        const __m128i K8_SHUFFLE_GR_0 = SIMD_MM_SETR_EPI8(0x1, 0x5, 0x7, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GR_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, 0xF, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GR_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x3, 0x5, 0x9, 0xB, 0xF);

        const __m128i K8_SHUFFLE_BG_0 = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x6, 0xA, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_BG_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x0, 0x2, 0x6, 0x8, 0xC, 0xE, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_BG_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE);

        const __m128i K8_SHUFFLE_GB_0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x7, 0x9, 0xD, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GB_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, 0x3, 0x5, 0x9, 0xB, 0xF, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GB_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x5, 0x7, 0xB, 0xD);

        const __m128i K8_SHUFFLE_RG_0 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_RG_1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, 0x0, 0x4, 0x6, 0xA, 0xC, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_RG_2 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x2, 0x6, 0x8, 0xC, 0xE);

        // Process one row of 32 Bayer pixels from 96 bytes of BGR input.
        //
        // The algorithm rearranges the three 32-byte source registers so that both
        // 128-bit lanes within each rearranged register have the same BGR offset
        // (mod 3), then applies the same shuffle mask to both lanes simultaneously.
        //
        // The three source registers cover bytes [0..31], [32..63], [64..95]:
        //   bgr0 lane0 (bytes  0..15): offset 0  (starts with B)
        //   bgr0 lane1 (bytes 16..31): offset 1  (starts with G)
        //   bgr1 lane0 (bytes 32..47): offset 2  (starts with R)
        //   bgr1 lane1 (bytes 48..63): offset 0  (starts with B)
        //   bgr2 lane0 (bytes 64..79): offset 1  (starts with G)
        //   bgr2 lane1 (bytes 80..95): offset 2  (starts with R)
        //
        // We group by offset:
        //   vec0 = { bgr0 lane0, bgr1 lane1 }  (both offset 0)
        //   vec1 = { bgr0 lane1, bgr2 lane0 }  (both offset 1)
        //   vec2 = { bgr1 lane0, bgr2 lane1 }  (both offset 2)
        //
        // Each vec_k produces non-zero values at specific output positions in both
        // the lower and upper output lanes, and the three results are OR'd together.
        template <int format, int row, bool align>
        SIMD_INLINE void BgrToBayer(const uint8_t* bgr, uint8_t* bayer, const __m256i shuffle[4][2][3])
        {
            const __m256i bgr0 = Load<align>((__m256i*)bgr + 0);
            const __m256i bgr1 = Load<align>((__m256i*)bgr + 1);
            const __m256i bgr2 = Load<align>((__m256i*)bgr + 2);

            const __m256i vec0 = _mm256_permute2x128_si256(bgr0, bgr1, 0x30);
            const __m256i vec1 = _mm256_permute2x128_si256(bgr0, bgr2, 0x21);
            const __m256i vec2 = _mm256_permute2x128_si256(bgr1, bgr2, 0x30);

            const __m256i bayer0 = _mm256_shuffle_epi8(vec0, shuffle[format][row][0]);
            const __m256i bayer1 = _mm256_shuffle_epi8(vec1, shuffle[format][row][1]);
            const __m256i bayer2 = _mm256_shuffle_epi8(vec2, shuffle[format][row][2]);

            Store<align>((__m256i*)bayer, _mm256_or_si256(_mm256_or_si256(bayer0, bayer1), bayer2));
        }

        template <int format, bool align>
        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const __m256i shuffle[4][2][3] =
            {
                {
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_2) },
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_2) }
                },
                {
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_2) },
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_2) }
                },
                {
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_RG_2) },
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_GB_2) }
                },
                {
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_BG_2) },
                    { _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_0), _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_1), _mm256_broadcastsi128_si256(K8_SHUFFLE_GR_2) }
                }
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
        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
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

        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride))
                BgrToBayer<true>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
            else
                BgrToBayer<false>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
