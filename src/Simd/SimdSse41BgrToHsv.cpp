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
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE __m128i MulDiv32Hsv(__m128i dividend, __m128i divisor, const __m128& scale)
        {
            return _mm_cvttps_epi32(_mm_div_ps(_mm_mul_ps(scale, _mm_cvtepi32_ps(dividend)), _mm_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m128i MulDiv16Hsv(__m128i dividend, __m128i divisor, const __m128& scale)
        {
            const __m128i lo = MulDiv32Hsv(_mm_unpacklo_epi16(dividend, K_ZERO), _mm_unpacklo_epi16(divisor, K_ZERO), scale);
            const __m128i hi = MulDiv32Hsv(_mm_unpackhi_epi16(dividend, K_ZERO), _mm_unpackhi_epi16(divisor, K_ZERO), scale);
            return _mm_packs_epi32(lo, hi);
        }

        SIMD_INLINE void BgrToHsv16(__m128i blue, __m128i green, __m128i red,
            __m128i& hue, __m128i& sat, __m128i& val,
            const __m128& KF_255_DIV_6, const __m128& K_255F)
        {
            __m128i max = MaxI16(red, green, blue);
            __m128i min = MinI16(red, green, blue);
            __m128i range = _mm_sub_epi16(max, min);

            // Hue: determine which channel is the maximum
            __m128i redMaxMask = _mm_cmpeq_epi16(red, max);
            __m128i greenMaxMask = _mm_andnot_si128(redMaxMask, _mm_cmpeq_epi16(green, max));
            __m128i blueMaxMask = _mm_andnot_si128(_mm_or_si128(redMaxMask, greenMaxMask), K_INV_ZERO);

            __m128i hueDividend = _mm_or_si128(
                _mm_and_si128(redMaxMask,
                    _mm_add_epi16(_mm_sub_epi16(green, blue), _mm_mullo_epi16(range, K16_0006))),
                _mm_or_si128(
                    _mm_and_si128(greenMaxMask,
                        _mm_add_epi16(_mm_sub_epi16(blue, red), _mm_mullo_epi16(range, K16_0002))),
                    _mm_and_si128(blueMaxMask,
                        _mm_add_epi16(_mm_sub_epi16(red, green), _mm_mullo_epi16(range, K16_0004)))));

            __m128i safeRange = _mm_max_epi16(range, K16_0001);
            hue = _mm_andnot_si128(_mm_cmpeq_epi16(range, K_ZERO),
                _mm_and_si128(MulDiv16Hsv(hueDividend, safeRange, KF_255_DIV_6), K16_00FF));

            // Value: V = max
            val = max;

            // Saturation: S = 255 * range / max, zero when max == 0
            __m128i safeMax = _mm_max_epi16(max, K16_0001);
            sat = _mm_andnot_si128(_mm_cmpeq_epi16(max, K_ZERO),
                MulDiv16Hsv(range, safeMax, K_255F));
        }

        template <bool align> SIMD_INLINE void BgrToHsv16(const uint8_t* bgr, uint8_t* hsv,
            const __m128& KF_255_DIV_6, const __m128& K_255F)
        {
            __m128i bgr_data[3];
            bgr_data[0] = Load<align>((__m128i*)bgr + 0);
            bgr_data[1] = Load<align>((__m128i*)bgr + 1);
            bgr_data[2] = Load<align>((__m128i*)bgr + 2);

            __m128i blue8 = BgrToBlue(bgr_data);
            __m128i green8 = BgrToGreen(bgr_data);
            __m128i red8 = BgrToRed(bgr_data);

            __m128i blue_lo = _mm_unpacklo_epi8(blue8, K_ZERO);
            __m128i blue_hi = _mm_unpackhi_epi8(blue8, K_ZERO);
            __m128i green_lo = _mm_unpacklo_epi8(green8, K_ZERO);
            __m128i green_hi = _mm_unpackhi_epi8(green8, K_ZERO);
            __m128i red_lo = _mm_unpacklo_epi8(red8, K_ZERO);
            __m128i red_hi = _mm_unpackhi_epi8(red8, K_ZERO);

            __m128i hue_lo, sat_lo, val_lo, hue_hi, sat_hi, val_hi;
            BgrToHsv16(blue_lo, green_lo, red_lo, hue_lo, sat_lo, val_lo, KF_255_DIV_6, K_255F);
            BgrToHsv16(blue_hi, green_hi, red_hi, hue_hi, sat_hi, val_hi, KF_255_DIV_6, K_255F);

            __m128i hue8 = _mm_packus_epi16(hue_lo, hue_hi);
            __m128i sat8 = _mm_packus_epi16(sat_lo, sat_hi);
            __m128i val8 = _mm_packus_epi16(val_lo, val_hi);

            Store<align>((__m128i*)hsv + 0, InterleaveBgr<0>(hue8, sat8, val8));
            Store<align>((__m128i*)hsv + 1, InterleaveBgr<1>(hue8, sat8, val8));
            Store<align>((__m128i*)hsv + 2, InterleaveBgr<2>(hue8, sat8, val8));
        }

        template <bool align> void BgrToHsv(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsv, size_t hsvStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(hsv) && Aligned(hsvStride));

            size_t alignedWidth = AlignLo(width, A);
            const __m128 KF = _mm_set_ps1(Base::KF_255_DIV_6);
            const __m128 K255 = _mm_set_ps1(255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BgrToHsv16<align>(bgr + 3 * col, hsv + 3 * col, KF, K255);
                if (width != alignedWidth)
                    BgrToHsv16<false>(bgr + 3 * (width - A), hsv + 3 * (width - A), KF, K255);
                bgr += bgrStride;
                hsv += hsvStride;
            }
        }

        void BgrToHsv(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsv, size_t hsvStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(hsv) && Aligned(hsvStride))
                BgrToHsv<true>(bgr, width, height, bgrStride, hsv, hsvStride);
            else
                BgrToHsv<false>(bgr, width, height, bgrStride, hsv, hsvStride);
        }
    }
#endif
}
