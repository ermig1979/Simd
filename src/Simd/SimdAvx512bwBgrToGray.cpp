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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K16_BLUE_RED = SIMD_MM512_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m512i K16_GREEN_0000 = SIMD_MM512_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m512i K32_ROUND_TERM = SIMD_MM512_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m512i PermutedBgrToGray32(__m512i permutedBgr)
        {
            const __m512i b0r0 = _mm512_shuffle_epi8(permutedBgr, K8_SUFFLE_BGR_TO_B0R0);
            const __m512i g000 = _mm512_shuffle_epi8(permutedBgr, K8_SUFFLE_BGR_TO_G000);
            const __m512i weightedSum = _mm512_add_epi32(_mm512_madd_epi16(g000, K16_GREEN_0000), _mm512_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm512_srli_epi32(_mm512_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template <bool align, bool mask> SIMD_INLINE void BgrToGray(const uint8_t * bgr, uint8_t * gray, const __mmask64 ms[4])
        {
            const __m512i bgr0 = Load<align, mask>(bgr + 0 * A, ms[0]);
            const __m512i bgr1 = Load<align, mask>(bgr + 1 * A, ms[1]);
            const __m512i bgr2 = Load<align, mask>(bgr + 2 * A, ms[2]);

            const __m512i permutedBgr0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            const __m512i permutedBgr1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            const __m512i permutedBgr2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            const __m512i permutedBgr3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);

            __m512i gray0 = PermutedBgrToGray32(permutedBgr0);
            __m512i gray1 = PermutedBgrToGray32(permutedBgr1);
            __m512i gray2 = PermutedBgrToGray32(permutedBgr2);
            __m512i gray3 = PermutedBgrToGray32(permutedBgr3);

            __m512i gray01 = _mm512_packs_epi32(gray0, gray1);
            __m512i gray23 = _mm512_packs_epi32(gray2, gray3);
            __m512i gray0123 = _mm512_packus_epi16(gray01, gray23);
            Store<align, mask>(gray, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, gray0123), ms[3]);
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            tailMasks[3] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgrToGray<align, false>(bgr + col * 3, gray + col, tailMasks);
                if (col < width)
                    BgrToGray<align, true>(bgr + col * 3, gray + col, tailMasks);
                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if (Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToGray<true>(bgr, width, height, bgrStride, gray, grayStride);
            else
                BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }

        //---------------------------------------------------------------------

        const __m512i K16_RED_BLUE = SIMD_MM512_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m512i PermutedRgbToGray32(__m512i permutedRgb)
        {
            const __m512i r0b0 = _mm512_shuffle_epi8(permutedRgb, K8_SUFFLE_BGR_TO_B0R0);
            const __m512i g000 = _mm512_shuffle_epi8(permutedRgb, K8_SUFFLE_BGR_TO_G000);
            const __m512i weightedSum = _mm512_add_epi32(_mm512_madd_epi16(g000, K16_GREEN_0000), _mm512_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm512_srli_epi32(_mm512_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template <bool align, bool mask> SIMD_INLINE void RgbToGray(const uint8_t* rgb, uint8_t* gray, const __mmask64 ms[4])
        {
            const __m512i rgb0 = Load<align, mask>(rgb + 0 * A, ms[0]);
            const __m512i rgb1 = Load<align, mask>(rgb + 1 * A, ms[1]);
            const __m512i rgb2 = Load<align, mask>(rgb + 2 * A, ms[2]);

            const __m512i permutedRgb0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, rgb0);
            const __m512i permutedRgb1 = _mm512_permutex2var_epi32(rgb0, K32_PERMUTE_BGR_TO_BGRA_1, rgb1);
            const __m512i permutedRgb2 = _mm512_permutex2var_epi32(rgb1, K32_PERMUTE_BGR_TO_BGRA_2, rgb2);
            const __m512i permutedRgb3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, rgb2);

            __m512i gray0 = PermutedRgbToGray32(permutedRgb0);
            __m512i gray1 = PermutedRgbToGray32(permutedRgb1);
            __m512i gray2 = PermutedRgbToGray32(permutedRgb2);
            __m512i gray3 = PermutedRgbToGray32(permutedRgb3);

            __m512i gray01 = _mm512_packs_epi32(gray0, gray1);
            __m512i gray23 = _mm512_packs_epi32(gray2, gray3);
            __m512i gray0123 = _mm512_packus_epi16(gray01, gray23);
            Store<align, mask>(gray, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, gray0123), ms[3]);
        }

        template <bool align> void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            if (align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A * c);
            tailMasks[3] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    RgbToGray<align, false>(rgb + col * 3, gray + col, tailMasks);
                if (col < width)
                    RgbToGray<align, true>(rgb + col * 3, gray + col, tailMasks);
                rgb += rgbStride;
                gray += grayStride;
            }
        }

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
        {
            if (Aligned(gray) && Aligned(grayStride) && Aligned(rgb) && Aligned(rgbStride))
                RgbToGray<true>(rgb, width, height, rgbStride, gray, grayStride);
            else
                RgbToGray<false>(rgb, width, height, rgbStride, gray, grayStride);
        }
    }
#endif//SIMD_AVX512BW_ENABLE
}
