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
#include "Simd/SimdInterleave.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        // Per 16-byte lane, extract one channel from the BGRA-like permuted layout:
        //   [B,G,R,B, G,R,B,G, R,B,G,R, X,X,X,X]  (4 pixels per 16-byte lane)
        // Blues at byte offsets 0,3,6,9 within each lane.
        const __m512i K8_HSV_PERMUTED_BGR_TO_BLUE = SIMD_MM512_SETR_EPI8(
            0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K8_HSV_PERMUTED_BGR_TO_GREEN = SIMD_MM512_SETR_EPI8(
            1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K8_HSV_PERMUTED_BGR_TO_RED = SIMD_MM512_SETR_EPI8(
            2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K32_HSV_PACK_CHANNEL_PAIRS = SIMD_MM512_SETR_EPI32(
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        SIMD_INLINE __m512i ExtractChannel16Hsv(const __m512i& perm0, const __m512i& perm1, const __m512i& shuf)
        {
            __m512i c0 = _mm512_shuffle_epi8(perm0, shuf);
            __m512i c1 = _mm512_shuffle_epi8(perm1, shuf);
            __m512i packed = _mm512_permutex2var_epi32(c0, K32_HSV_PACK_CHANNEL_PAIRS, c1);
            return _mm512_cvtepu8_epi16(_mm512_castsi512_si256(packed));
        }

        SIMD_INLINE __m256i PackChannelHsv(const __m512i& ch)
        {
            return _mm512_castsi512_si256(
                _mm512_permutexvar_epi32(K32_PERMUTE_FOR_PACK,
                    _mm512_packus_epi16(ch, K_ZERO)));
        }

        SIMD_INLINE __m512i MulDiv32Hsv(const __m512i& dividend, const __m512i& divisor, const __m512& scale)
        {
            return _mm512_cvttps_epi32(_mm512_div_ps(
                _mm512_mul_ps(scale, _mm512_cvtepi32_ps(dividend)),
                _mm512_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m512i MulDiv16Hsv(const __m512i& dividend, const __m512i& divisor, const __m512& scale)
        {
            const __m512i lo = MulDiv32Hsv(_mm512_unpacklo_epi16(dividend, K_ZERO),
                _mm512_unpacklo_epi16(divisor, K_ZERO), scale);
            const __m512i hi = MulDiv32Hsv(_mm512_unpackhi_epi16(dividend, K_ZERO),
                _mm512_unpackhi_epi16(divisor, K_ZERO), scale);
            return _mm512_packs_epi32(lo, hi);
        }

        // Compute HSV for 32 pixels whose B, G, R channels are provided as 32
        // uint16 values each in a __m512i.
        SIMD_INLINE void BgrToHsv32(__m512i blue, __m512i green, __m512i red,
            __m512i& hue, __m512i& sat, __m512i& val,
            const __m512& KF_255_DIV_6, const __m512& K_255F)
        {
            __m512i max = _mm512_max_epi16(red, _mm512_max_epi16(green, blue));
            __m512i min = _mm512_min_epi16(red, _mm512_min_epi16(green, blue));
            __m512i range = _mm512_sub_epi16(max, min);

            // --- Hue ---
            const __mmask32 redMaxMask = _mm512_cmpeq_epi16_mask(red, max);
            const __mmask32 greenMaxMask = (~redMaxMask) & _mm512_cmpeq_epi16_mask(green, max);
            const __mmask32 blueMaxMask = ~(redMaxMask | greenMaxMask);

            __m512i hueDividend = _mm512_maskz_add_epi16(redMaxMask,
                _mm512_sub_epi16(green, blue), _mm512_mullo_epi16(range, K16_0006));
            hueDividend = _mm512_mask_add_epi16(hueDividend, greenMaxMask,
                _mm512_sub_epi16(blue, red), _mm512_mullo_epi16(range, K16_0002));
            hueDividend = _mm512_mask_add_epi16(hueDividend, blueMaxMask,
                _mm512_sub_epi16(red, green), _mm512_mullo_epi16(range, K16_0004));

            __m512i safeRange = _mm512_max_epi16(range, K16_0001);
            hue = _mm512_and_si512(
                MulDiv16Hsv(hueDividend, safeRange, KF_255_DIV_6),
                _mm512_maskz_set1_epi16(_mm512_cmpneq_epi16_mask(range, K_ZERO), 0xFF));

            // --- Value: V = max ---
            val = max;

            // --- Saturation: S = 255 * range / max, 0 when max == 0 ---
            __m512i safeMax = _mm512_max_epi16(max, K16_0001);
            sat = _mm512_maskz_mov_epi16(_mm512_cmpneq_epi16_mask(max, K_ZERO),
                MulDiv16Hsv(range, safeMax, K_255F));
        }

        // Process 64 BGR pixels and write 64 HSV pixels.
        // tails[0..2]: masks for the three 64-byte BGR input chunks.
        // tails[3..5]: masks for the three 64-byte HSV output chunks (same values).
        template <bool align, bool mask> SIMD_INLINE void BgrToHsv64(
            const uint8_t* bgr, uint8_t* hsv,
            const __m512& KF, const __m512& K255,
            const __mmask64 tails[6])
        {
            // Load 3 × 64 bytes of packed BGR data.
            __m512i bgr0 = Load<align, mask>(bgr + 0 * A, tails[0]);
            __m512i bgr1 = Load<align, mask>(bgr + 1 * A, tails[1]);
            __m512i bgr2 = Load<align, mask>(bgr + 2 * A, tails[2]);

            // Expand packed BGR into 4 BGRA-like groups of 16 pixels each.
            __m512i perm0 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0);
            __m512i perm1 = _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1);
            __m512i perm2 = _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2);
            __m512i perm3 = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2);

            // Extract B, G, R as 32 uint16 values for pixels 0-31 (perm0 + perm1).
            __m512i blue01 = ExtractChannel16Hsv(perm0, perm1, K8_HSV_PERMUTED_BGR_TO_BLUE);
            __m512i green01 = ExtractChannel16Hsv(perm0, perm1, K8_HSV_PERMUTED_BGR_TO_GREEN);
            __m512i red01 = ExtractChannel16Hsv(perm0, perm1, K8_HSV_PERMUTED_BGR_TO_RED);

            // Extract B, G, R as 32 uint16 values for pixels 32-63 (perm2 + perm3).
            __m512i blue23 = ExtractChannel16Hsv(perm2, perm3, K8_HSV_PERMUTED_BGR_TO_BLUE);
            __m512i green23 = ExtractChannel16Hsv(perm2, perm3, K8_HSV_PERMUTED_BGR_TO_GREEN);
            __m512i red23 = ExtractChannel16Hsv(perm2, perm3, K8_HSV_PERMUTED_BGR_TO_RED);

            // Compute H, S, V for pixels 0-31.
            __m512i hue01, sat01, val01;
            BgrToHsv32(blue01, green01, red01, hue01, sat01, val01, KF, K255);

            // Compute H, S, V for pixels 32-63.
            __m512i hue23, sat23, val23;
            BgrToHsv32(blue23, green23, red23, hue23, sat23, val23, KF, K255);

            // Pack each channel from uint16 to uint8 and combine both halves into a
            // single 64-element __m512i.
            __m512i hue8 = _mm512_inserti64x4(
                _mm512_castsi256_si512(PackChannelHsv(hue01)), PackChannelHsv(hue23), 1);
            __m512i sat8 = _mm512_inserti64x4(
                _mm512_castsi256_si512(PackChannelHsv(sat01)), PackChannelHsv(sat23), 1);
            __m512i val8 = _mm512_inserti64x4(
                _mm512_castsi256_si512(PackChannelHsv(val01)), PackChannelHsv(val23), 1);

            // Interleave H, S, V into packed HSV and store 3 × 64 bytes.
            Store<align, mask>(hsv + 0 * A, InterleaveBgr<0>(hue8, sat8, val8), tails[0]);
            Store<align, mask>(hsv + 1 * A, InterleaveBgr<1>(hue8, sat8, val8), tails[1]);
            Store<align, mask>(hsv + 2 * A, InterleaveBgr<2>(hue8, sat8, val8), tails[2]);
        }

        template <bool align> void BgrToHsv(const uint8_t* bgr, size_t width, size_t height,
            size_t bgrStride, uint8_t* hsv, size_t hsvStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(hsv) && Aligned(hsvStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tails[3];
            for (size_t i = 0; i < 3; ++i)
                tails[i] = TailMask64((width - alignedWidth) * 3 - A * i);

            const __m512 KF = _mm512_set1_ps(Base::KF_255_DIV_6);
            const __m512 K255 = _mm512_set1_ps(255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgrToHsv64<align, false>(bgr + col * 3, hsv + col * 3, KF, K255, tails);
                if (col < width)
                    BgrToHsv64<align, true>(bgr + col * 3, hsv + col * 3, KF, K255, tails);
                bgr += bgrStride;
                hsv += hsvStride;
            }
        }

        void BgrToHsv(const uint8_t* bgr, size_t width, size_t height,
            size_t bgrStride, uint8_t* hsv, size_t hsvStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(hsv) && Aligned(hsvStride))
                BgrToHsv<true>(bgr, width, height, bgrStride, hsv, hsvStride);
            else
                BgrToHsv<false>(bgr, width, height, bgrStride, hsv, hsvStride);
        }
    }
#endif
}
