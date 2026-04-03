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

        SIMD_INLINE void AdjustedYuvToHsv16(__m128i y, __m128i u, __m128i v,
            __m128i& hue, __m128i& sat, __m128i& val,
            const __m128& KF_255_DIV_6, const __m128& K_255F)
        {
            const __m128i red = AdjustedYuvToRed16(y, v);
            const __m128i green = AdjustedYuvToGreen16(y, u, v);
            const __m128i blue = AdjustedYuvToBlue16(y, u);

            const __m128i max = MaxI16(red, green, blue);
            const __m128i min = MinI16(red, green, blue);
            const __m128i range = _mm_sub_epi16(max, min);

            // Hue: same formula as HSL
            const __m128i redMaxMask = _mm_cmpeq_epi16(red, max);
            const __m128i greenMaxMask = _mm_andnot_si128(redMaxMask, _mm_cmpeq_epi16(green, max));
            const __m128i blueMaxMask = _mm_andnot_si128(_mm_or_si128(redMaxMask, greenMaxMask), K_INV_ZERO);

            const __m128i dividend = _mm_or_si128(
                _mm_and_si128(redMaxMask,
                    _mm_add_epi16(_mm_sub_epi16(green, blue), _mm_mullo_epi16(range, K16_0006))),
                _mm_or_si128(
                    _mm_and_si128(greenMaxMask,
                        _mm_add_epi16(_mm_sub_epi16(blue, red), _mm_mullo_epi16(range, K16_0002))),
                    _mm_and_si128(blueMaxMask,
                        _mm_add_epi16(_mm_sub_epi16(red, green), _mm_mullo_epi16(range, K16_0004)))));

            const __m128i safeRange = _mm_max_epi16(range, K16_0001);
            hue = _mm_andnot_si128(_mm_cmpeq_epi16(range, K_ZERO),
                _mm_and_si128(MulDiv16Hsv(dividend, safeRange, KF_255_DIV_6), K16_00FF));

            // Value: V = max
            val = max;

            // Saturation: S = range * 255 / max, zero when max == 0
            const __m128i range_lo = _mm_unpacklo_epi16(range, K_ZERO);
            const __m128i range_hi = _mm_unpackhi_epi16(range, K_ZERO);
            const __m128i max_lo = _mm_unpacklo_epi16(max, K_ZERO);
            const __m128i max_hi = _mm_unpackhi_epi16(max, K_ZERO);

            const __m128i K32_1 = _mm_set1_epi32(1);
            const __m128i safeMax_lo = _mm_max_epi32(max_lo, K32_1);
            const __m128i safeMax_hi = _mm_max_epi32(max_hi, K32_1);

            __m128i sat_lo = _mm_cvttps_epi32(_mm_floor_ps(_mm_div_ps(
                _mm_mul_ps(K_255F, _mm_cvtepi32_ps(range_lo)),
                _mm_cvtepi32_ps(safeMax_lo))));
            __m128i sat_hi = _mm_cvttps_epi32(_mm_floor_ps(_mm_div_ps(
                _mm_mul_ps(K_255F, _mm_cvtepi32_ps(range_hi)),
                _mm_cvtepi32_ps(safeMax_hi))));

            sat_lo = _mm_andnot_si128(_mm_cmpeq_epi32(range_lo, K_ZERO), sat_lo);
            sat_hi = _mm_andnot_si128(_mm_cmpeq_epi32(range_hi, K_ZERO), sat_hi);

            sat = _mm_packs_epi32(sat_lo, sat_hi);
        }

        template <bool align> SIMD_INLINE void YuvToHsv16x(const uint8_t* y, const uint8_t* u, const uint8_t* v,
            uint8_t* hsv, const __m128& KF_255_DIV_6, const __m128& K_255F)
        {
            const __m128i y8 = Load<align>((__m128i*)y);
            const __m128i u8 = Load<align>((__m128i*)u);
            const __m128i v8 = Load<align>((__m128i*)v);

            __m128i hue_lo, sat_lo, val_lo, hue_hi, sat_hi, val_hi;
            AdjustedYuvToHsv16(
                AdjustY16(_mm_unpacklo_epi8(y8, K_ZERO)),
                AdjustUV16(_mm_unpacklo_epi8(u8, K_ZERO)),
                AdjustUV16(_mm_unpacklo_epi8(v8, K_ZERO)),
                hue_lo, sat_lo, val_lo, KF_255_DIV_6, K_255F);
            AdjustedYuvToHsv16(
                AdjustY16(_mm_unpackhi_epi8(y8, K_ZERO)),
                AdjustUV16(_mm_unpackhi_epi8(u8, K_ZERO)),
                AdjustUV16(_mm_unpackhi_epi8(v8, K_ZERO)),
                hue_hi, sat_hi, val_hi, KF_255_DIV_6, K_255F);

            const __m128i hue8 = _mm_packus_epi16(hue_lo, hue_hi);
            const __m128i sat8 = _mm_packus_epi16(sat_lo, sat_hi);
            const __m128i val8 = _mm_packus_epi16(val_lo, val_hi);

            Store<align>((__m128i*)hsv + 0, InterleaveBgr<0>(hue8, sat8, val8));
            Store<align>((__m128i*)hsv + 1, InterleaveBgr<1>(hue8, sat8, val8));
            Store<align>((__m128i*)hsv + 2, InterleaveBgr<2>(hue8, sat8, val8));
        }

        template <bool align> void Yuv444pToHsv(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* hsv, size_t hsvStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hsv) && Aligned(hsvStride));
            }

            const __m128 KF = _mm_set_ps1(Base::KF_255_DIV_6);
            const __m128 K255 = _mm_set_ps1(255.0f);

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    YuvToHsv16x<align>(y + col, u + col, v + col, hsv + 3 * col, KF, K255);
                if (width != alignedWidth)
                {
                    size_t col = width - A;
                    YuvToHsv16x<false>(y + col, u + col, v + col, hsv + 3 * col, KF, K255);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hsv += hsvStride;
            }
        }

        void Yuv444pToHsv(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* hsv, size_t hsvStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) &&
                Aligned(v) && Aligned(vStride) && Aligned(hsv) && Aligned(hsvStride))
                Yuv444pToHsv<true>(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
            else
                Yuv444pToHsv<false>(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
        }
    }
#endif
}
