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
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE  
    namespace Avx512bw
    {
        const __m128i K8_SHUFFLE_GR = SIMD_MM_SETR_EPI8(0x1, 0x6, 0x9, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_BG = SIMD_MM_SETR_EPI8(0x0, 0x5, 0x8, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GB = SIMD_MM_SETR_EPI8(0x1, 0x4, 0x9, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_RG = SIMD_MM_SETR_EPI8(0x2, 0x5, 0xA, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGRA_TO_BAYER_0 = SIMD_MM512_SETR_EPI32(0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K32_PERMUTE_BGRA_TO_BAYER_1 = SIMD_MM512_SETR_EPI32(-1, -1, -1, -1, -1, -1, -1, -1, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C);

        template <int format, int row, bool mask> SIMD_INLINE void BgraToBayer(const uint8_t * bgra, uint8_t * bayer, const __m512i shuffle[4][2], __mmask64 ms[5])
        {
            const __m512i bgra0 = mask ? _mm512_maskz_loadu_epi8(ms[0], bgra + 0 * A) : _mm512_loadu_si512(bgra + 0 * A);
            const __m512i bgra1 = mask ? _mm512_maskz_loadu_epi8(ms[1], bgra + 1 * A) : _mm512_loadu_si512(bgra + 1 * A);
            const __m512i bgra2 = mask ? _mm512_maskz_loadu_epi8(ms[2], bgra + 2 * A) : _mm512_loadu_si512(bgra + 2 * A);
            const __m512i bgra3 = mask ? _mm512_maskz_loadu_epi8(ms[3], bgra + 3 * A) : _mm512_loadu_si512(bgra + 3 * A);
            const __m512i bayer0 = _mm512_shuffle_epi8(bgra0, shuffle[format][row]);
            const __m512i bayer1 = _mm512_shuffle_epi8(bgra1, shuffle[format][row]);
            const __m512i bayer2 = _mm512_shuffle_epi8(bgra2, shuffle[format][row]);
            const __m512i bayer3 = _mm512_shuffle_epi8(bgra3, shuffle[format][row]);
            __m512i bayer01xx = _mm512_permutex2var_epi32(bayer0, K32_PERMUTE_BGRA_TO_BAYER_0, bayer1);
            __m512i bayerxx23 = _mm512_permutex2var_epi32(bayer2, K32_PERMUTE_BGRA_TO_BAYER_1, bayer3);
            const __m512i bayerOut = _mm512_or_si512(bayer01xx, bayerxx23);
            if (mask)
                _mm512_mask_storeu_epi8(bayer, ms[4], bayerOut);
            else
                _mm512_storeu_si512(bayer, bayerOut);
        }

        template <int format> void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            const __m512i shuffle[4][2] =
            {
                { _mm512_broadcast_i32x4(K8_SHUFFLE_GR), _mm512_broadcast_i32x4(K8_SHUFFLE_BG)},
                { _mm512_broadcast_i32x4(K8_SHUFFLE_GB), _mm512_broadcast_i32x4(K8_SHUFFLE_RG)},
                { _mm512_broadcast_i32x4(K8_SHUFFLE_RG), _mm512_broadcast_i32x4(K8_SHUFFLE_GB)},
                { _mm512_broadcast_i32x4(K8_SHUFFLE_BG), _mm512_broadcast_i32x4(K8_SHUFFLE_GR)}
            };

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A*c);
            tailMasks[4] = TailMask64(width - alignedWidth);

            for (size_t row = 0, col = 0; row < height; row += 2)
            {
                for (col = 0; col < alignedWidth; col += A)
                    BgraToBayer<format, 0, false>(bgra + 4 * col, bayer + col, shuffle, tailMasks);
                if (col < width)
                    BgraToBayer<format, 0, true>(bgra + 4 * col, bayer + col, shuffle, tailMasks);
                bgra += bgraStride;
                bayer += bayerStride;

                for (col = 0; col < alignedWidth; col += A)
                    BgraToBayer<format, 1, false>(bgra + 4 * col, bayer + col, shuffle, tailMasks);
                if (col < width)
                    BgraToBayer<format, 1, true>(bgra + 4 * col, bayer + col, shuffle, tailMasks);
                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<0>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<1>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<2>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<3>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
