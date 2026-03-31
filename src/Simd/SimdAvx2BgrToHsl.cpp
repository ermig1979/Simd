/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdInterleave.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE __m256i MulDiv32(__m256i dividend, __m256i divisor, const __m256& scale)
        {
            return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(dividend)), _mm256_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m256i MulDiv16(__m256i dividend, __m256i divisor, const __m256& scale)
        {
            const __m256i lo = MulDiv32(_mm256_unpacklo_epi16(dividend, K_ZERO), _mm256_unpacklo_epi16(divisor, K_ZERO), scale);
            const __m256i hi = MulDiv32(_mm256_unpackhi_epi16(dividend, K_ZERO), _mm256_unpackhi_epi16(divisor, K_ZERO), scale);
            return _mm256_packs_epi32(lo, hi);
        }

        SIMD_INLINE void BgrToHsl16(__m256i blue, __m256i green, __m256i red,
            __m256i& hue, __m256i& sat, __m256i& lgt,
            const __m256& KF_255_DIV_6, const __m256& K_255F)
        {
            __m256i max = MaxI16(red, green, blue);
            __m256i min = MinI16(red, green, blue);
            __m256i range = _mm256_sub_epi16(max, min);
            __m256i sum = _mm256_add_epi16(max, min);

            // Hue: determine which channel is the maximum
            __m256i redMaxMask = _mm256_cmpeq_epi16(red, max);
            __m256i greenMaxMask = _mm256_andnot_si256(redMaxMask, _mm256_cmpeq_epi16(green, max));
            __m256i blueMaxMask = _mm256_andnot_si256(_mm256_or_si256(redMaxMask, greenMaxMask), K_INV_ZERO);

            __m256i hueDividend = _mm256_or_si256(
                _mm256_and_si256(redMaxMask,
                    _mm256_add_epi16(_mm256_sub_epi16(green, blue), _mm256_mullo_epi16(range, K16_0006))),
                _mm256_or_si256(
                    _mm256_and_si256(greenMaxMask,
                        _mm256_add_epi16(_mm256_sub_epi16(blue, red), _mm256_mullo_epi16(range, K16_0002))),
                    _mm256_and_si256(blueMaxMask,
                        _mm256_add_epi16(_mm256_sub_epi16(red, green), _mm256_mullo_epi16(range, K16_0004)))));

            __m256i safeRange = _mm256_max_epi16(range, K16_0001);
            hue = _mm256_andnot_si256(_mm256_cmpeq_epi16(range, K_ZERO),
                _mm256_and_si256(MulDiv16(hueDividend, safeRange, KF_255_DIV_6), K16_00FF));

            // Lightness: L = (max + min) / 2
            lgt = _mm256_srli_epi16(sum, 1);

            // Saturation: S = range * 255 / min(sum, 510 - sum), zero when range == 0
            __m256i range_lo = _mm256_unpacklo_epi16(range, K_ZERO);
            __m256i range_hi = _mm256_unpackhi_epi16(range, K_ZERO);
            __m256i sum_lo = _mm256_unpacklo_epi16(sum, K_ZERO);
            __m256i sum_hi = _mm256_unpackhi_epi16(sum, K_ZERO);

            const __m256i K32_510 = _mm256_set1_epi32(510);
            const __m256i K32_1 = _mm256_set1_epi32(1);

            __m256i denom_lo = _mm256_min_epi32(sum_lo, _mm256_sub_epi32(K32_510, sum_lo));
            __m256i denom_hi = _mm256_min_epi32(sum_hi, _mm256_sub_epi32(K32_510, sum_hi));
            __m256i denomSafe_lo = _mm256_max_epi32(denom_lo, K32_1);
            __m256i denomSafe_hi = _mm256_max_epi32(denom_hi, K32_1);

            __m256i sat_lo = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_div_ps(
                _mm256_mul_ps(K_255F, _mm256_cvtepi32_ps(range_lo)),
                _mm256_cvtepi32_ps(denomSafe_lo))));
            __m256i sat_hi = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_div_ps(
                _mm256_mul_ps(K_255F, _mm256_cvtepi32_ps(range_hi)),
                _mm256_cvtepi32_ps(denomSafe_hi))));

            __m256i zeroRangeMask_lo = _mm256_cmpeq_epi32(range_lo, K_ZERO);
            __m256i zeroRangeMask_hi = _mm256_cmpeq_epi32(range_hi, K_ZERO);
            sat_lo = _mm256_andnot_si256(zeroRangeMask_lo, sat_lo);
            sat_hi = _mm256_andnot_si256(zeroRangeMask_hi, sat_hi);

            sat = _mm256_packs_epi32(sat_lo, sat_hi);
        }

        template <bool align> SIMD_INLINE void BgrToHsl32(const uint8_t* bgr, uint8_t* hsl,
            const __m256& KF_255_DIV_6, const __m256& K_255F)
        {
            __m256i bgr_data[3];
            bgr_data[0] = Load<align>((__m256i*)bgr + 0);
            bgr_data[1] = Load<align>((__m256i*)bgr + 1);
            bgr_data[2] = Load<align>((__m256i*)bgr + 2);

            __m256i blue8 = BgrToBlue(bgr_data);
            __m256i green8 = BgrToGreen(bgr_data);
            __m256i red8 = BgrToRed(bgr_data);

            __m256i blue_lo = _mm256_unpacklo_epi8(blue8, K_ZERO);
            __m256i blue_hi = _mm256_unpackhi_epi8(blue8, K_ZERO);
            __m256i green_lo = _mm256_unpacklo_epi8(green8, K_ZERO);
            __m256i green_hi = _mm256_unpackhi_epi8(green8, K_ZERO);
            __m256i red_lo = _mm256_unpacklo_epi8(red8, K_ZERO);
            __m256i red_hi = _mm256_unpackhi_epi8(red8, K_ZERO);

            __m256i hue_lo, sat_lo, lgt_lo, hue_hi, sat_hi, lgt_hi;
            BgrToHsl16(blue_lo, green_lo, red_lo, hue_lo, sat_lo, lgt_lo, KF_255_DIV_6, K_255F);
            BgrToHsl16(blue_hi, green_hi, red_hi, hue_hi, sat_hi, lgt_hi, KF_255_DIV_6, K_255F);

            __m256i hue8 = _mm256_packus_epi16(hue_lo, hue_hi);
            __m256i sat8 = _mm256_packus_epi16(sat_lo, sat_hi);
            __m256i lgt8 = _mm256_packus_epi16(lgt_lo, lgt_hi);

            Store<align>((__m256i*)hsl + 0, InterleaveBgr<0>(hue8, sat8, lgt8));
            Store<align>((__m256i*)hsl + 1, InterleaveBgr<1>(hue8, sat8, lgt8));
            Store<align>((__m256i*)hsl + 2, InterleaveBgr<2>(hue8, sat8, lgt8));
        }

        template <bool align> void BgrToHsl(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsl, size_t hslStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(hsl) && Aligned(hslStride));

            size_t alignedWidth = AlignLo(width, A);
            const __m256 KF = _mm256_set1_ps(Base::KF_255_DIV_6);
            const __m256 K255 = _mm256_set1_ps(255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BgrToHsl32<align>(bgr + 3 * col, hsl + 3 * col, KF, K255);
                if (width != alignedWidth)
                    BgrToHsl32<false>(bgr + 3 * (width - A), hsl + 3 * (width - A), KF, K255);
                bgr += bgrStride;
                hsl += hslStride;
            }
        }

        void BgrToHsl(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsl, size_t hslStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(hsl) && Aligned(hslStride))
                BgrToHsl<true>(bgr, width, height, bgrStride, hsl, hslStride);
            else
                BgrToHsl<false>(bgr, width, height, bgrStride, hsl, hslStride);
        }
    }
#endif
}
