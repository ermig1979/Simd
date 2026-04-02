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
        SIMD_INLINE __m256i MulDiv32Hsl(__m256i dividend, __m256i divisor, const __m256& scale)
        {
            return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(dividend)), _mm256_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m256i MulDiv16Hsl(__m256i dividend, __m256i divisor, const __m256& scale)
        {
            const __m256i lo = MulDiv32Hsl(_mm256_unpacklo_epi16(dividend, K_ZERO), _mm256_unpacklo_epi16(divisor, K_ZERO), scale);
            const __m256i hi = MulDiv32Hsl(_mm256_unpackhi_epi16(dividend, K_ZERO), _mm256_unpackhi_epi16(divisor, K_ZERO), scale);
            return _mm256_packs_epi32(lo, hi);
        }

        SIMD_INLINE void AdjustedYuvToHsl16(__m256i y, __m256i u, __m256i v,
            __m256i& hue, __m256i& sat, __m256i& lgt,
            const __m256& KF_255_DIV_6, const __m256& K_255F)
        {
            const __m256i red = AdjustedYuvToRed16(y, v);
            const __m256i green = AdjustedYuvToGreen16(y, u, v);
            const __m256i blue = AdjustedYuvToBlue16(y, u);

            const __m256i max = MaxI16(red, green, blue);
            const __m256i min = MinI16(red, green, blue);
            const __m256i range = _mm256_sub_epi16(max, min);
            const __m256i sum = _mm256_add_epi16(max, min);

            // Hue: determine which channel is the maximum
            const __m256i redMaxMask = _mm256_cmpeq_epi16(red, max);
            const __m256i greenMaxMask = _mm256_andnot_si256(redMaxMask, _mm256_cmpeq_epi16(green, max));
            const __m256i blueMaxMask = _mm256_andnot_si256(_mm256_or_si256(redMaxMask, greenMaxMask), K_INV_ZERO);

            const __m256i dividend = _mm256_or_si256(
                _mm256_and_si256(redMaxMask,
                    _mm256_add_epi16(_mm256_sub_epi16(green, blue), _mm256_mullo_epi16(range, K16_0006))),
                _mm256_or_si256(
                    _mm256_and_si256(greenMaxMask,
                        _mm256_add_epi16(_mm256_sub_epi16(blue, red), _mm256_mullo_epi16(range, K16_0002))),
                    _mm256_and_si256(blueMaxMask,
                        _mm256_add_epi16(_mm256_sub_epi16(red, green), _mm256_mullo_epi16(range, K16_0004)))));

            const __m256i safeRange = _mm256_max_epi16(range, K16_0001);
            hue = _mm256_andnot_si256(_mm256_cmpeq_epi16(range, K_ZERO),
                _mm256_and_si256(MulDiv16Hsl(dividend, safeRange, KF_255_DIV_6), K16_00FF));

            // Lightness: L = (max + min) / 2
            lgt = _mm256_srli_epi16(sum, 1);

            // Saturation: S = range * 255 / min(sum, 510 - sum), zero when range == 0
            const __m256i range_lo = _mm256_unpacklo_epi16(range, K_ZERO);
            const __m256i range_hi = _mm256_unpackhi_epi16(range, K_ZERO);
            const __m256i sum_lo = _mm256_unpacklo_epi16(sum, K_ZERO);
            const __m256i sum_hi = _mm256_unpackhi_epi16(sum, K_ZERO);

            const __m256i K32_510 = _mm256_set1_epi32(510);
            const __m256i K32_1 = _mm256_set1_epi32(1);

            const __m256i denom_lo = _mm256_min_epi32(sum_lo, _mm256_sub_epi32(K32_510, sum_lo));
            const __m256i denom_hi = _mm256_min_epi32(sum_hi, _mm256_sub_epi32(K32_510, sum_hi));
            const __m256i denomSafe_lo = _mm256_max_epi32(denom_lo, K32_1);
            const __m256i denomSafe_hi = _mm256_max_epi32(denom_hi, K32_1);

            __m256i sat_lo = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_div_ps(
                _mm256_mul_ps(K_255F, _mm256_cvtepi32_ps(range_lo)),
                _mm256_cvtepi32_ps(denomSafe_lo))));
            __m256i sat_hi = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_div_ps(
                _mm256_mul_ps(K_255F, _mm256_cvtepi32_ps(range_hi)),
                _mm256_cvtepi32_ps(denomSafe_hi))));

            sat_lo = _mm256_andnot_si256(_mm256_cmpeq_epi32(range_lo, K_ZERO), sat_lo);
            sat_hi = _mm256_andnot_si256(_mm256_cmpeq_epi32(range_hi, K_ZERO), sat_hi);

            sat = _mm256_packs_epi32(sat_lo, sat_hi);
        }

        template <bool align> SIMD_INLINE void YuvToHsl32x(const uint8_t* y, const uint8_t* u, const uint8_t* v,
            uint8_t* hsl, const __m256& KF_255_DIV_6, const __m256& K_255F)
        {
            const __m256i y8 = Load<align>((__m256i*)y);
            const __m256i u8 = Load<align>((__m256i*)u);
            const __m256i v8 = Load<align>((__m256i*)v);

            __m256i hue_lo, sat_lo, lgt_lo, hue_hi, sat_hi, lgt_hi;
            AdjustedYuvToHsl16(
                AdjustY16(_mm256_unpacklo_epi8(y8, K_ZERO)),
                AdjustUV16(_mm256_unpacklo_epi8(u8, K_ZERO)),
                AdjustUV16(_mm256_unpacklo_epi8(v8, K_ZERO)),
                hue_lo, sat_lo, lgt_lo, KF_255_DIV_6, K_255F);
            AdjustedYuvToHsl16(
                AdjustY16(_mm256_unpackhi_epi8(y8, K_ZERO)),
                AdjustUV16(_mm256_unpackhi_epi8(u8, K_ZERO)),
                AdjustUV16(_mm256_unpackhi_epi8(v8, K_ZERO)),
                hue_hi, sat_hi, lgt_hi, KF_255_DIV_6, K_255F);

            const __m256i hue8 = _mm256_packus_epi16(hue_lo, hue_hi);
            const __m256i sat8 = _mm256_packus_epi16(sat_lo, sat_hi);
            const __m256i lgt8 = _mm256_packus_epi16(lgt_lo, lgt_hi);

            Store<align>((__m256i*)hsl + 0, InterleaveBgr<0>(hue8, sat8, lgt8));
            Store<align>((__m256i*)hsl + 1, InterleaveBgr<1>(hue8, sat8, lgt8));
            Store<align>((__m256i*)hsl + 2, InterleaveBgr<2>(hue8, sat8, lgt8));
        }

        template <bool align> void Yuv444pToHsl(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* hsl, size_t hslStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hsl) && Aligned(hslStride));
            }

            const __m256 KF = _mm256_set1_ps(Base::KF_255_DIV_6);
            const __m256 K255 = _mm256_set1_ps(255.0f);

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    YuvToHsl32x<align>(y + col, u + col, v + col, hsl + 3 * col, KF, K255);
                if (width != alignedWidth)
                {
                    size_t col = width - A;
                    YuvToHsl32x<false>(y + col, u + col, v + col, hsl + 3 * col, KF, K255);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hsl += hslStride;
            }
        }

        void Yuv444pToHsl(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* hsl, size_t hslStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) &&
                Aligned(v) && Aligned(vStride) && Aligned(hsl) && Aligned(hslStride))
                Yuv444pToHsl<true>(y, yStride, u, uStride, v, vStride, width, height, hsl, hslStride);
            else
                Yuv444pToHsl<false>(y, yStride, u, uStride, v, vStride, width, height, hsl, hslStride);
        }
    }
#endif
}
