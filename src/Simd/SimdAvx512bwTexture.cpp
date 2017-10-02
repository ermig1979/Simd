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
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i TextureBoostedSaturatedGradient16(const __m512i & difference, const __m512i & saturation, const __m512i & boost)
        {
            return _mm512_mullo_epi16(_mm512_max_epi16(K_ZERO, _mm512_add_epi16(saturation, _mm512_min_epi16(difference, saturation))), boost);
        }

        SIMD_INLINE __m512i TextureBoostedSaturatedGradient8(const __m512i & a, const __m512i & b, const __m512i & saturation, const __m512i & boost)
        {
            __m512i lo = TextureBoostedSaturatedGradient16(SubUnpackedU8<0>(b, a), saturation, boost);
            __m512i hi = TextureBoostedSaturatedGradient16(SubUnpackedU8<1>(b, a), saturation, boost);
            return _mm512_packus_epi16(lo, hi);
        }

        template<bool align, bool mask> SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, uint8_t * dx, uint8_t * dy,
            size_t stride, const __m512i & saturation, const __m512i & boost, __mmask64 tail = -1)
        {
            const __m512i s10 = Load<false, mask>(src - 1, tail);
            const __m512i s12 = Load<false, mask>(src + 1, tail);
            const __m512i s01 = Load<align, mask>(src - stride, tail);
            const __m512i s21 = Load<align, mask>(src + stride, tail);
            Store<align, mask>(dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost), tail);
            Store<align, mask>(dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost), tail);
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(int(2)*saturation*boost <= 0xFF);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i _saturation = _mm512_set1_epi16(saturation);
            __m512i _boost = _mm512_set1_epi16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align, false>(src + col, dx + col, dy + col, srcStride, _saturation, _boost);
                if (col < width)
                    TextureBoostedSaturatedGradient<false, true>(src + col, dx + col, dy + col, srcStride, _saturation, _boost, tailMask);

                dx[0] = 0;
                dy[0] = 0;
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }

        template<bool align, bool mask> SIMD_INLINE void TextureBoostedUv(const uint8_t * src, uint8_t * dst,
            const __m512i & min8, const __m512i & max8, const __m512i & boost16, __mmask64 tail = -1)
        {
            const __m512i _src = Load<align, mask>(src, tail);
            const __m512i saturated = _mm512_sub_epi8(_mm512_max_epu8(min8, _mm512_min_epu8(max8, _src)), min8);
            const __m512i lo = _mm512_mullo_epi16(_mm512_unpacklo_epi8(saturated, K_ZERO), boost16);
            const __m512i hi = _mm512_mullo_epi16(_mm512_unpackhi_epi8(saturated, K_ZERO), boost16);
            Store<align, mask>(dst, _mm512_packus_epi16(lo, hi), tail);
        }

        template<bool align> void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            assert(boost < 0x80);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            int min = 128 - (128 / boost);
            int max = 255 - min;
            __m512i min8 = _mm512_set1_epi8(min);
            __m512i max8 = _mm512_set1_epi8(max);
            __m512i boost16 = _mm512_set1_epi16(boost);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    TextureBoostedUv<align, false>(src + col, dst + col, min8, max8, boost16);
                if (col < width)
                    TextureBoostedUv<false, true>(src + col, dst + col, min8, max8, boost16, tailMask);
                src += srcStride;
                dst += dstStride;
            }
        }

        void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TextureBoostedUv<true>(src, srcStride, width, height, boost, dst, dstStride);
            else
                TextureBoostedUv<false>(src, srcStride, width, height, boost, dst, dstStride);
        }

        SIMD_INLINE void TextureGetDifferenceSum(const __m512i & current, const __m512i & average, __m512i & positive, __m512i & negative)
        {
            positive = _mm512_add_epi64(positive, _mm512_sad_epu8(_mm512_subs_epu8(current, average), K_ZERO));
            negative = _mm512_add_epi64(negative, _mm512_sad_epu8(_mm512_subs_epu8(average, current), K_ZERO));
        }

        template <bool align, bool mask> SIMD_INLINE void TextureGetDifferenceSum(const uint8_t * src, const uint8_t * lo, const uint8_t * hi,
            __m512i & positive, __m512i & negative, __mmask64 tail = -1)
        {
            const __m512i current = Load<align, mask>(src, tail);
            const __m512i _lo = Load<align, mask>(lo, tail);
            const __m512i _hi = Load<align, mask>(hi, tail);
            const __m512i average = _mm512_avg_epu8(_lo, _hi);
            TextureGetDifferenceSum(current, average, positive, negative);
        }

        template <bool align> SIMD_INLINE void TextureGetDifferenceSum4(const uint8_t * src, const uint8_t * lo, const uint8_t * hi, __m512i & positive, __m512i & negative)
        {
            TextureGetDifferenceSum(Load<align>(src + 0 * A), _mm512_avg_epu8(Load<align>(hi + 0 * A), Load<align>(lo + 0 * A)), positive, negative);
            TextureGetDifferenceSum(Load<align>(src + 1 * A), _mm512_avg_epu8(Load<align>(hi + 1 * A), Load<align>(lo + 1 * A)), positive, negative);
            TextureGetDifferenceSum(Load<align>(src + 2 * A), _mm512_avg_epu8(Load<align>(hi + 2 * A), Load<align>(lo + 2 * A)), positive, negative);
            TextureGetDifferenceSum(Load<align>(src + 3 * A), _mm512_avg_epu8(Load<align>(hi + 3 * A), Load<align>(lo + 3 * A)), positive, negative);
        }

        template <bool align> void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            assert(sum != NULL);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride));

            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i positive = _mm512_setzero_si512();
            __m512i negative = _mm512_setzero_si512();
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    TextureGetDifferenceSum4<align>(src + col, lo + col, hi + col, positive, negative);
                for (; col < alignedWidth; col += A)
                    TextureGetDifferenceSum<align, false>(src + col, lo + col, hi + col, positive, negative);
                if (col < width)
                    TextureGetDifferenceSum<align, true>(src + col, lo + col, hi + col, positive, negative, tailMask);
                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
            *sum = ExtractSum<int64_t>(positive) - ExtractSum<int64_t>(negative);
        }

        void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                TextureGetDifferenceSum<true>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
            else
                TextureGetDifferenceSum<false>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        }

        template <bool align> void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height, int shift, uint8_t * dst, size_t dstStride)
        {
            assert(shift > -0xFF && shift < 0xFF && shift != 0);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            if (shift > 0)
            {
                __m512i _shift = _mm512_set1_epi8((char)shift);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                    {
                        Store<align>(dst + col + 0 * A, _mm512_adds_epu8(Load<align>(src + col + 0 * A), _shift));
                        Store<align>(dst + col + 1 * A, _mm512_adds_epu8(Load<align>(src + col + 1 * A), _shift));
                        Store<align>(dst + col + 2 * A, _mm512_adds_epu8(Load<align>(src + col + 2 * A), _shift));
                        Store<align>(dst + col + 3 * A, _mm512_adds_epu8(Load<align>(src + col + 3 * A), _shift));
                    }
                    for (; col < alignedWidth; col += A)
                        Store<align>(dst + col, _mm512_adds_epu8(Load<align>(src + col), _shift));
                    if (col < width)
                        Store<align, true>(dst + col, _mm512_adds_epu8((Load<align, true>(src + col, tailMask)), _shift), tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            if (shift < 0)
            {
                __m512i _shift = _mm512_set1_epi8((char)-shift);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                    {
                        Store<align>(dst + col + 0 * A, _mm512_subs_epu8(Load<align>(src + col + 0 * A), _shift));
                        Store<align>(dst + col + 1 * A, _mm512_subs_epu8(Load<align>(src + col + 1 * A), _shift));
                        Store<align>(dst + col + 2 * A, _mm512_subs_epu8(Load<align>(src + col + 2 * A), _shift));
                        Store<align>(dst + col + 3 * A, _mm512_subs_epu8(Load<align>(src + col + 3 * A), _shift));
                    }
                    for (; col < alignedWidth; col += A)
                        Store<align>(dst + col, _mm512_subs_epu8(Load<align>(src + col), _shift));
                    if (col < width)
                        Store<align, true>(dst + col, _mm512_subs_epu8((Load<align, true>(src + col, tailMask)), _shift), tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }
        }

        void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t * dst, size_t dstStride)
        {
            if (shift == 0)
            {
                if (src != dst)
                    Base::Copy(src, srcStride, width, height, 1, dst, dstStride);
                return;
            }
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TexturePerformCompensation<true>(src, srcStride, width, height, shift, dst, dstStride);
            else
                TexturePerformCompensation<false>(src, srcStride, width, height, shift, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
