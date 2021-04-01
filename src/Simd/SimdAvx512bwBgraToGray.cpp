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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K16_BLUE_RED = SIMD_MM512_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m512i K16_GREEN_0000 = SIMD_MM512_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m512i K32_ROUND_TERM = SIMD_MM512_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m512i BgraToGray32(__m512i bgra)
        {
            const __m512i g0a0 = _mm512_shuffle_epi8(bgra, K8_SUFFLE_BGRA_TO_G0A0);
            const __m512i b0r0 = _mm512_and_si512(bgra, K16_00FF);
            const __m512i weightedSum = _mm512_add_epi32(_mm512_madd_epi16(g0a0, K16_GREEN_0000), _mm512_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm512_srli_epi32(_mm512_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template <bool align, bool mask> SIMD_INLINE void BgraToGray(const uint8_t * bgra, uint8_t * gray, __mmask64 ms[5])
        {
            __m512i gray0 = BgraToGray32(Load<align, mask>(bgra + 0 * A, ms[0]));
            __m512i gray1 = BgraToGray32(Load<align, mask>(bgra + 1 * A, ms[1]));
            __m512i gray2 = BgraToGray32(Load<align, mask>(bgra + 2 * A, ms[2]));
            __m512i gray3 = BgraToGray32(Load<align, mask>(bgra + 3 * A, ms[3]));
            __m512i gray01 = _mm512_packs_epi32(gray0, gray1);
            __m512i gray23 = _mm512_packs_epi32(gray2, gray3);
            __m512i gray0123 = _mm512_packus_epi16(gray01, gray23);
            Store<align, mask>(gray, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, gray0123), ms[4]);
        }

        template <bool align> void BgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride)
        {
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A * c);
            tailMasks[4] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgraToGray<align, false>(bgra + col * 4, gray + col, tailMasks);
                if (col < width)
                    BgraToGray<align, true>(bgra + col * 4, gray + col, tailMasks);
                bgra += bgraStride;
                gray += grayStride;
            }
        }

        void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                BgraToGray<true>(bgra, width, height, bgraStride, gray, grayStride);
            else
                BgraToGray<false>(bgra, width, height, bgraStride, gray, grayStride);
        }

        //---------------------------------------------------------------------

        const __m512i K16_RED_BLUE = SIMD_MM512_SET2_EPI16(Base::RED_TO_GRAY_WEIGHT, Base::BLUE_TO_GRAY_WEIGHT);

        SIMD_INLINE __m512i RgbaToGray32(__m512i rgba)
        {
            const __m512i g0a0 = _mm512_shuffle_epi8(rgba, K8_SUFFLE_BGRA_TO_G0A0);
            const __m512i r0b0 = _mm512_and_si512(rgba, K16_00FF);
            const __m512i weightedSum = _mm512_add_epi32(_mm512_madd_epi16(g0a0, K16_GREEN_0000), _mm512_madd_epi16(r0b0, K16_RED_BLUE));
            return _mm512_srli_epi32(_mm512_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template <bool align, bool mask> SIMD_INLINE void RgbaToGray(const uint8_t* rgba, uint8_t* gray, __mmask64 ms[5])
        {
            __m512i gray0 = RgbaToGray32(Load<align, mask>(rgba + 0 * A, ms[0]));
            __m512i gray1 = RgbaToGray32(Load<align, mask>(rgba + 1 * A, ms[1]));
            __m512i gray2 = RgbaToGray32(Load<align, mask>(rgba + 2 * A, ms[2]));
            __m512i gray3 = RgbaToGray32(Load<align, mask>(rgba + 3 * A, ms[3]));
            __m512i gray01 = _mm512_packs_epi32(gray0, gray1);
            __m512i gray23 = _mm512_packs_epi32(gray2, gray3);
            __m512i gray0123 = _mm512_packus_epi16(gray01, gray23);
            Store<align, mask>(gray, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, gray0123), ms[4]);
        }

        template <bool align> void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            if (align)
                assert(Aligned(rgba) && Aligned(rgbaStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A * c);
            tailMasks[4] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    RgbaToGray<align, false>(rgba + col * 4, gray + col, tailMasks);
                if (col < width)
                    RgbaToGray<align, true>(rgba + col * 4, gray + col, tailMasks);
                rgba += rgbaStride;
                gray += grayStride;
            }
        }

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            if (Aligned(rgba) && Aligned(gray) && Aligned(rgbaStride) && Aligned(grayStride))
                RgbaToGray<true>(rgba, width, height, rgbaStride, gray, grayStride);
            else
                RgbaToGray<false>(rgba, width, height, rgbaStride, gray, grayStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
