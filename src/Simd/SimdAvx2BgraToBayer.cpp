/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
        // AVX2 shuffle masks - same pattern as SSE4.1 but broadcast to both 128-bit lanes.
        // _mm256_shuffle_epi8 operates independently on each 128-bit lane, so the same
        // 128-bit lane mask is replicated in both halves of the 256-bit constant.
        // Each mask extracts one channel byte per BGRA pixel into the lowest byte of a dword.
        // BGRA pixel layout: B=offset 0, G=offset 1, R=offset 2, A=offset 3.
        // Odd row (row 0): pair is (row0_channel, row0_other_channel)
        // Even row (row 1): pair is (row1_channel, row1_other_channel)
        // Format 0 = GRBG: row0 = GR, row1 = BG
        // Format 1 = GBRG: row0 = GB, row1 = RG
        // Format 2 = RGGB: row0 = RG, row1 = GB
        // Format 3 = BGGR: row0 = BG, row1 = GR
        const __m256i K8_SHUFFLE_GR = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1, 0xE, -1, -1, -1,
            0x1, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1, 0xE, -1, -1, -1);
        const __m256i K8_SHUFFLE_BG = SIMD_MM256_SETR_EPI8(
            0x0, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xD, -1, -1, -1,
            0x0, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xD, -1, -1, -1);
        const __m256i K8_SHUFFLE_GB = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x9, -1, -1, -1, 0xC, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x9, -1, -1, -1, 0xC, -1, -1, -1);
        const __m256i K8_SHUFFLE_RG = SIMD_MM256_SETR_EPI8(
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0xA, -1, -1, -1, 0xD, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0xA, -1, -1, -1, 0xD, -1, -1, -1);

        // After packing bayer0..bayer3 with packs_epi32 + packus_epi16, the 32-bit
        // elements are in the order: [px0-3, px8-11, px16-19, px24-27, px4-7, px12-15, px20-23, px28-31].
        // This permutation reorders them to sequential pixel order: [0,4,1,5,2,6,3,7].
        const __m256i K32_PERMUTE_BGRA_TO_BAYER = SIMD_MM256_SETR_EPI32(0, 4, 1, 5, 2, 6, 3, 7);

        template <int format, int row, bool align>
        SIMD_INLINE void BgraToBayer(const uint8_t * bgra, uint8_t * bayer, const __m256i shuffle[4][2])
        {
            const __m256i bayer0 = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 0), shuffle[format][row]);
            const __m256i bayer1 = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 1), shuffle[format][row]);
            const __m256i bayer2 = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 2), shuffle[format][row]);
            const __m256i bayer3 = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 3), shuffle[format][row]);
            Store<align>((__m256i*)bayer, _mm256_permutevar8x32_epi32(
                _mm256_packus_epi16(
                    _mm256_packs_epi32(bayer0, bayer1),
                    _mm256_packs_epi32(bayer2, bayer3)),
                K32_PERMUTE_BGRA_TO_BAYER));
        }

        template <int format, bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const __m256i shuffle[4][2] =
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
#endif// SIMD_AVX2_ENABLE
}
