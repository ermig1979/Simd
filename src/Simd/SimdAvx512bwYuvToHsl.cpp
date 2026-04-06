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
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        SIMD_INLINE __m512i MulDiv32Hsl(const __m512i& dividend, const __m512i& divisor, const __m512& scale)
        {
            return _mm512_cvttps_epi32(_mm512_div_ps(_mm512_mul_ps(scale, _mm512_cvtepi32_ps(dividend)), _mm512_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m512i MulDiv16Hsl(const __m512i& dividend, const __m512i& divisor, const __m512& scale)
        {
            const __m512i lo = MulDiv32Hsl(_mm512_unpacklo_epi16(dividend, K_ZERO), _mm512_unpacklo_epi16(divisor, K_ZERO), scale);
            const __m512i hi = MulDiv32Hsl(_mm512_unpackhi_epi16(dividend, K_ZERO), _mm512_unpackhi_epi16(divisor, K_ZERO), scale);
            return _mm512_packs_epi32(lo, hi);
        }

        SIMD_INLINE void AdjustedYuvToHsl16(const __m512i& y, const __m512i& u, const __m512i& v,
            __m512i& hue, __m512i& sat, __m512i& lgt,
            const __m512& KF_255_DIV_6, const __m512& K_255F)
        {
            const __m512i red = AdjustedYuvToRed16(y, v);
            const __m512i green = AdjustedYuvToGreen16(y, u, v);
            const __m512i blue = AdjustedYuvToBlue16(y, u);

            const __m512i max = _mm512_max_epi16(red, _mm512_max_epi16(green, blue));
            const __m512i min = _mm512_min_epi16(red, _mm512_min_epi16(green, blue));
            const __m512i range = _mm512_sub_epi16(max, min);
            const __m512i sum = _mm512_add_epi16(max, min);

            // Hue: determine which channel is the maximum
            const __mmask32 redMaxMask = _mm512_cmpeq_epi16_mask(red, max);
            const __mmask32 greenMaxMask = (~redMaxMask) & _mm512_cmpeq_epi16_mask(green, max);
            const __mmask32 blueMaxMask = ~(redMaxMask | greenMaxMask);

            __m512i dividend = _mm512_maskz_add_epi16(redMaxMask,
                _mm512_sub_epi16(green, blue), _mm512_mullo_epi16(range, K16_0006));
            dividend = _mm512_mask_add_epi16(dividend, greenMaxMask,
                _mm512_sub_epi16(blue, red), _mm512_mullo_epi16(range, K16_0002));
            dividend = _mm512_mask_add_epi16(dividend, blueMaxMask,
                _mm512_sub_epi16(red, green), _mm512_mullo_epi16(range, K16_0004));

            const __m512i safeRange = _mm512_max_epi16(range, K16_0001);
            hue = _mm512_and_si512(
                MulDiv16Hsl(dividend, safeRange, KF_255_DIV_6),
                _mm512_maskz_set1_epi16(_mm512_cmpneq_epi16_mask(range, K_ZERO), 0xFF));

            // Lightness: L = (max + min) / 2
            lgt = _mm512_srli_epi16(sum, 1);

            // Saturation: S = range * 255 / min(sum, 510 - sum), zero when range == 0
            const __m512i K32_510 = _mm512_set1_epi32(510);
            const __m512i K32_1 = _mm512_set1_epi32(1);

            const __m512i range_lo = _mm512_unpacklo_epi16(range, K_ZERO);
            const __m512i range_hi = _mm512_unpackhi_epi16(range, K_ZERO);
            const __m512i sum_lo = _mm512_unpacklo_epi16(sum, K_ZERO);
            const __m512i sum_hi = _mm512_unpackhi_epi16(sum, K_ZERO);

            const __m512i denom_lo = _mm512_min_epi32(sum_lo, _mm512_sub_epi32(K32_510, sum_lo));
            const __m512i denom_hi = _mm512_min_epi32(sum_hi, _mm512_sub_epi32(K32_510, sum_hi));
            const __m512i denomSafe_lo = _mm512_max_epi32(denom_lo, K32_1);
            const __m512i denomSafe_hi = _mm512_max_epi32(denom_hi, K32_1);

            __m512i sat_lo = _mm512_cvttps_epi32(_mm512_floor_ps(_mm512_div_ps(
                _mm512_mul_ps(K_255F, _mm512_cvtepi32_ps(range_lo)),
                _mm512_cvtepi32_ps(denomSafe_lo))));
            __m512i sat_hi = _mm512_cvttps_epi32(_mm512_floor_ps(_mm512_div_ps(
                _mm512_mul_ps(K_255F, _mm512_cvtepi32_ps(range_hi)),
                _mm512_cvtepi32_ps(denomSafe_hi))));

            sat_lo = _mm512_maskz_mov_epi32(_mm512_cmpneq_epi32_mask(range_lo, K_ZERO), sat_lo);
            sat_hi = _mm512_maskz_mov_epi32(_mm512_cmpneq_epi32_mask(range_hi, K_ZERO), sat_hi);

            sat = _mm512_packs_epi32(sat_lo, sat_hi);
        }

        template <bool align, bool mask> SIMD_INLINE void YuvToHsl64x(const uint8_t* y, const uint8_t* u, const uint8_t* v,
            uint8_t* hsl, const __m512& KF, const __m512& K255,
            __mmask64 tailYuv, const __mmask64 tailHsl[3])
        {
            const __m512i y8 = Load<align, mask>(y, tailYuv);
            const __m512i u8 = Load<align, mask>(u, tailYuv);
            const __m512i v8 = Load<align, mask>(v, tailYuv);

            // Process first 32 pixels (lo half)
            __m512i hue_lo, sat_lo, lgt_lo;
            AdjustedYuvToHsl16(
                AdjustY16(UnpackU8<0>(y8)),
                AdjustUV16(UnpackU8<0>(u8)),
                AdjustUV16(UnpackU8<0>(v8)),
                hue_lo, sat_lo, lgt_lo, KF, K255);

            // Process second 32 pixels (hi half)
            __m512i hue_hi, sat_hi, lgt_hi;
            AdjustedYuvToHsl16(
                AdjustY16(UnpackU8<1>(y8)),
                AdjustUV16(UnpackU8<1>(u8)),
                AdjustUV16(UnpackU8<1>(v8)),
                hue_hi, sat_hi, lgt_hi, KF, K255);

            // Pack each channel from uint16 to uint8 and store interleaved HSL
            const __m512i hue8 = _mm512_packus_epi16(hue_lo, hue_hi);
            const __m512i sat8 = _mm512_packus_epi16(sat_lo, sat_hi);
            const __m512i lgt8 = _mm512_packus_epi16(lgt_lo, lgt_hi);

            Store<align, mask>(hsl + 0 * A, InterleaveBgr<0>(hue8, sat8, lgt8), tailHsl[0]);
            Store<align, mask>(hsl + 1 * A, InterleaveBgr<1>(hue8, sat8, lgt8), tailHsl[1]);
            Store<align, mask>(hsl + 2 * A, InterleaveBgr<2>(hue8, sat8, lgt8), tailHsl[2]);
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

            const __m512 KF = _mm512_set1_ps(Base::KF_255_DIV_6);
            const __m512 K255 = _mm512_set1_ps(255.0f);

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailYuvMask = TailMask64(tail);
            __mmask64 tailHslMasks[3];
            for (size_t i = 0; i < 3; ++i)
                tailHslMasks[i] = TailMask64((ptrdiff_t)(tail * 3) - (ptrdiff_t)(A * i));

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    YuvToHsl64x<align, false>(y + col, u + col, v + col, hsl + 3 * col, KF, K255, tailYuvMask, tailHslMasks);
                if (col < width)
                    YuvToHsl64x<align, true>(y + col, u + col, v + col, hsl + 3 * col, KF, K255, tailYuvMask, tailHslMasks);
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
