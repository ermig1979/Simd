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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE  
    namespace Avx512bw
    {
        const __m128i K8_SHUFFLE_GR = SIMD_MM_SETR_EPI8(0x1, 0x5, 0x7, 0xB, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_BG = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x6, 0xA, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_GB = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x7, 0x9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_RG = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x8, 0xA, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGRA_TO_BAYER_0 = SIMD_MM512_SETR_EPI32(0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K32_PERMUTE_BGRA_TO_BAYER_1 = SIMD_MM512_SETR_EPI32(-1, -1, -1, -1, -1, -1, -1, -1, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C);

        template <int format, int row, bool align, bool mask> SIMD_INLINE void BgrToBayer(const uint8_t * bgr, uint8_t * bayer, const __m512i shuffle[4][2], __mmask64 ms[5])
        {
            const __m512i bgr0 = Load<align, mask>(bgr + 0 * A, ms[0]);
            const __m512i bgr1 = Load<align, mask>(bgr + 1 * A, ms[1]);
            const __m512i bgr2 = Load<align, mask>(bgr + 2 * A, ms[2]);

            const __m512i bgra0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            const __m512i bgra1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            const __m512i bgra2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            const __m512i bgra3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);

            const __m512i bayer0 = _mm512_shuffle_epi8(bgra0, shuffle[format][row]);
            const __m512i bayer1 = _mm512_shuffle_epi8(bgra1, shuffle[format][row]);
            const __m512i bayer2 = _mm512_shuffle_epi8(bgra2, shuffle[format][row]);
            const __m512i bayer3 = _mm512_shuffle_epi8(bgra3, shuffle[format][row]);

            __m512i bayer01xx = _mm512_permutex2var_epi32(bayer0, K32_PERMUTE_BGRA_TO_BAYER_0, bayer1);
            __m512i bayerxx23 = _mm512_permutex2var_epi32(bayer2, K32_PERMUTE_BGRA_TO_BAYER_1, bayer3);
            Store<align, mask>(bayer, _mm512_or_si512(bayer01xx, bayerxx23), ms[3]);
        }

        template <int format, bool align> void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride));

            const __m512i shuffle[4][2] =
            {
                { _mm512_broadcast_i32x4(K8_SHUFFLE_GR), _mm512_broadcast_i32x4(K8_SHUFFLE_BG) },
                { _mm512_broadcast_i32x4(K8_SHUFFLE_GB), _mm512_broadcast_i32x4(K8_SHUFFLE_RG) },
                { _mm512_broadcast_i32x4(K8_SHUFFLE_RG), _mm512_broadcast_i32x4(K8_SHUFFLE_GB) },
                { _mm512_broadcast_i32x4(K8_SHUFFLE_BG), _mm512_broadcast_i32x4(K8_SHUFFLE_GR) }
            };

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            tailMasks[3] = TailMask64(width - alignedWidth);

            for (size_t row = 0, col = 0; row < height; row += 2)
            {
                for (col = 0; col < alignedWidth; col += A)
                    BgrToBayer<format, 0, align, false>(bgr + 3 * col, bayer + col, shuffle, tailMasks);
                if (col < width)
                    BgrToBayer<format, 0, align, true>(bgr + 3 * col, bayer + col, shuffle, tailMasks);
                bgr += bgrStride;
                bayer += bayerStride;

                for (col = 0; col < alignedWidth; col += A)
                    BgrToBayer<format, 1, align, false>(bgr + 3 * col, bayer + col, shuffle, tailMasks);
                if (col < width)
                    BgrToBayer<format, 1, align, true>(bgr + 3 * col, bayer + col, shuffle, tailMasks);
                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        template<bool align> void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
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
#endif// SIMD_AVX512BW_ENABLE
}
