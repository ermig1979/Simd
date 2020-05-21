/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512BW_ENABLE  
    namespace Avx512bw
    {
        const __m512i K8_SHUFFLE_BGR_TO_BGRA = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        template <bool align, bool mask> SIMD_INLINE void BgrToBgra(const uint8_t * bgr, uint8_t * bgra, const __m512i & alpha, const __mmask64 * ms)
        {
            __m512i bgr0 = Load<align, mask>(bgr + 0 * A, ms[0]);
            __m512i bgr1 = Load<align, mask>(bgr + 1 * A, ms[1]);
            __m512i bgr2 = Load<align, mask>(bgr + 2 * A, ms[2]);

            const __m512i bgra0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            const __m512i bgra1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            const __m512i bgra2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            const __m512i bgra3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);

            Store<align, mask>(bgra + 0 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra0, K8_SHUFFLE_BGR_TO_BGRA)), ms[3]);
            Store<align, mask>(bgra + 1 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra1, K8_SHUFFLE_BGR_TO_BGRA)), ms[4]);
            Store<align, mask>(bgra + 2 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra2, K8_SHUFFLE_BGR_TO_BGRA)), ms[5]);
            Store<align, mask>(bgra + 3 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra3, K8_SHUFFLE_BGR_TO_BGRA)), ms[6]);
        }

        template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[7];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            for (size_t c = 0; c < 4; ++c)
                tailMasks[3 + c] = TailMask64((width - alignedWidth) * 4 - A*c);
            __m512i _alpha = _mm512_set1_epi32(alpha * 0x1000000);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgrToBgra<align, false>(bgr + 3 * col, bgra + 4 * col, _alpha, tailMasks);
                if (col < width)
                    BgrToBgra<align, true>(bgr + 3 * col, bgra + 4 * col, _alpha, tailMasks);
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

        template <bool align, bool mask> SIMD_INLINE void Bgr48pToBgra32(
            const uint8_t * blue, const uint8_t * green, const uint8_t * red, uint8_t * bgra, __m512i alpha, const __mmask64 * ms)
        {
            __m512i _blue = Load<align, true>(blue, ms[0]);
            __m512i _green = Load<align, true>(green, ms[0]);
            __m512i _red = Load<align, true>(red, ms[0]);

            __m512i bg = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_or_si512(_blue, _mm512_slli_epi16(_green, 8)));
            __m512i ra = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_or_si512(_red, alpha));

            Store<align, mask>(bgra + 0, _mm512_unpacklo_epi16(bg, ra), ms[1]);
            Store<align, mask>(bgra + A, _mm512_unpackhi_epi16(bg, ra), ms[2]);
        }

        template <bool align> void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (align)
            {
                assert(Aligned(blue) && Aligned(blueStride));
                assert(Aligned(green) && Aligned(greenStride));
                assert(Aligned(red) && Aligned(redStride));
                assert(Aligned(bgra) && Aligned(bgraStride));
            }

            width *= 2;
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 bodyMask = 0x5555555555555555;
            __mmask64 tailMasks[3];
            tailMasks[0] = TailMask64(width - alignedWidth)&bodyMask;
            for (size_t c = 0; c < 2; ++c)
                tailMasks[1 + c] = TailMask64((width - alignedWidth) * 2 - A*c);
            __m512i _alpha = _mm512_set1_epi16(alpha * 0x100);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Bgr48pToBgra32<align, false>(blue + col, green + col, red + col, bgra + col * 2, _alpha, &bodyMask);
                if (col < width)
                    Bgr48pToBgra32<align, true>(blue + col, green + col, red + col, bgra + col * 2, _alpha, tailMasks);
                blue += blueStride;
                green += greenStride;
                red += redStride;
                bgra += bgraStride;
            }
        }

        void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(blue) && Aligned(blueStride) && Aligned(green) && Aligned(greenStride) &&
                Aligned(red) && Aligned(redStride) && Aligned(bgra) && Aligned(bgraStride))
                Bgr48pToBgra32<true>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
            else
                Bgr48pToBgra32<false>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
        }

        //---------------------------------------------------------------------

        const __m512i K8_SHUFFLE_RGB_TO_BGRA = SIMD_MM512_SETR_EPI8(
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
            0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1);

        template <bool align, bool mask> SIMD_INLINE void RgbToBgra(const uint8_t* rgb, uint8_t* bgra, const __m512i& alpha, const __mmask64* ms)
        {
            __m512i rgb0 = Load<align, mask>(rgb + 0 * A, ms[0]);
            __m512i rgb1 = Load<align, mask>(rgb + 1 * A, ms[1]);
            __m512i rgb2 = Load<align, mask>(rgb + 2 * A, ms[2]);

            const __m512i bgra0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, rgb0);
            const __m512i bgra1 = _mm512_permutex2var_epi32(rgb0, K32_PERMUTE_BGR_TO_BGRA_1, rgb1);
            const __m512i bgra2 = _mm512_permutex2var_epi32(rgb1, K32_PERMUTE_BGR_TO_BGRA_2, rgb2);
            const __m512i bgra3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, rgb2);

            Store<align, mask>(bgra + 0 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra0, K8_SHUFFLE_RGB_TO_BGRA)), ms[3]);
            Store<align, mask>(bgra + 1 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra1, K8_SHUFFLE_RGB_TO_BGRA)), ms[4]);
            Store<align, mask>(bgra + 2 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra2, K8_SHUFFLE_RGB_TO_BGRA)), ms[5]);
            Store<align, mask>(bgra + 3 * A, _mm512_or_si512(alpha, _mm512_shuffle_epi8(bgra3, K8_SHUFFLE_RGB_TO_BGRA)), ms[6]);
        }

        template <bool align> void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[7];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A * c);
            for (size_t c = 0; c < 4; ++c)
                tailMasks[3 + c] = TailMask64((width - alignedWidth) * 4 - A * c);
            __m512i _alpha = _mm512_set1_epi32(alpha * 0x1000000);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    RgbToBgra<align, false>(rgb + 3 * col, bgra + 4 * col, _alpha, tailMasks);
                if (col < width)
                    RgbToBgra<align, true>(rgb + 3 * col, bgra + 4 * col, _alpha, tailMasks);
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
#endif// SIMD_AVX512BW_ENABLE
}
