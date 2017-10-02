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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE __m128i TextureBoostedSaturatedGradient16(__m128i a, __m128i b, __m128i saturation, const __m128i & boost)
        {
            return _mm_mullo_epi16(_mm_max_epi16(K_ZERO, _mm_add_epi16(saturation, _mm_min_epi16(_mm_sub_epi16(b, a), saturation))), boost);
        }

        SIMD_INLINE __m128i TextureBoostedSaturatedGradient8(__m128i a, __m128i b, __m128i saturation, const __m128i & boost)
        {
            __m128i lo = TextureBoostedSaturatedGradient16(_mm_unpacklo_epi8(a, K_ZERO), _mm_unpacklo_epi8(b, K_ZERO), saturation, boost);
            __m128i hi = TextureBoostedSaturatedGradient16(_mm_unpackhi_epi8(a, K_ZERO), _mm_unpackhi_epi8(b, K_ZERO), saturation, boost);
            return _mm_packus_epi16(lo, hi);
        }

        template<bool align> SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, uint8_t * dx, uint8_t * dy,
            size_t stride, __m128i saturation, __m128i boost)
        {
            const __m128i s10 = Load<false>((__m128i*)(src - 1));
            const __m128i s12 = Load<false>((__m128i*)(src + 1));
            const __m128i s01 = Load<align>((__m128i*)(src - stride));
            const __m128i s21 = Load<align>((__m128i*)(src + stride));
            Store<align>((__m128i*)dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost));
            Store<align>((__m128i*)dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost));
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(width >= A && int(2)*saturation*boost <= 0xFF);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i _saturation = _mm_set1_epi16(saturation);
            __m128i _boost = _mm_set1_epi16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align>(src + col, dx + col, dy + col, srcStride, _saturation, _boost);
                if (width != alignedWidth)
                    TextureBoostedSaturatedGradient<false>(src + width - A, dx + width - A, dy + width - A, srcStride, _saturation, _boost);

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

        template<bool align> SIMD_INLINE void TextureBoostedUv(const uint8_t * src, uint8_t * dst, __m128i min8, __m128i max8, __m128i boost16)
        {
            const __m128i _src = Load<align>((__m128i*)src);
            const __m128i saturated = _mm_sub_epi8(_mm_max_epu8(min8, _mm_min_epu8(max8, _src)), min8);
            const __m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(saturated, K_ZERO), boost16);
            const __m128i hi = _mm_mullo_epi16(_mm_unpackhi_epi8(saturated, K_ZERO), boost16);
            Store<align>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template<bool align> void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A && boost < 0x80);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            int min = 128 - (128 / boost);
            int max = 255 - min;

            __m128i min8 = _mm_set1_epi8(min);
            __m128i max8 = _mm_set1_epi8(max);
            __m128i boost16 = _mm_set1_epi16(boost);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedUv<align>(src + col, dst + col, min8, max8, boost16);
                if (width != alignedWidth)
                    TextureBoostedUv<false>(src + width - A, dst + width - A, min8, max8, boost16);

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

        template <bool align> SIMD_INLINE void TextureGetDifferenceSum(const uint8_t * src, const uint8_t * lo, const uint8_t * hi,
            __m128i & positive, __m128i & negative, const __m128i & mask)
        {
            const __m128i _src = Load<align>((__m128i*)src);
            const __m128i _lo = Load<align>((__m128i*)lo);
            const __m128i _hi = Load<align>((__m128i*)hi);
            const __m128i average = _mm_and_si128(mask, _mm_avg_epu8(_lo, _hi));
            const __m128i current = _mm_and_si128(mask, _src);
            positive = _mm_add_epi64(positive, _mm_sad_epu8(_mm_subs_epu8(current, average), K_ZERO));
            negative = _mm_add_epi64(negative, _mm_sad_epu8(_mm_subs_epu8(average, current), K_ZERO));
        }

        template <bool align> void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            assert(width >= A && sum != NULL);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            __m128i positive = _mm_setzero_si128();
            __m128i negative = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureGetDifferenceSum<align>(src + col, lo + col, hi + col, positive, negative, K_INV_ZERO);
                if (width != alignedWidth)
                    TextureGetDifferenceSum<false>(src + width - A, lo + width - A, hi + width - A, positive, negative, tailMask);
                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
            *sum = ExtractInt64Sum(positive) - ExtractInt64Sum(negative);
        }

        void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                TextureGetDifferenceSum<true>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
            else
                TextureGetDifferenceSum<false>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        }

        template <bool align> void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A && shift > -0xFF && shift < 0xFF && shift != 0);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = src == dst ? ShiftLeft(K_INV_ZERO, A - width + alignedWidth) : K_INV_ZERO;
            if (shift > 0)
            {
                __m128i _shift = _mm_set1_epi8((char)shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        const __m128i _src = Load<align>((__m128i*) (src + col));
                        Store<align>((__m128i*) (dst + col), _mm_adds_epu8(_src, _shift));
                    }
                    if (width != alignedWidth)
                    {
                        const __m128i _src = Load<false>((__m128i*) (src + width - A));
                        Store<false>((__m128i*) (dst + width - A), _mm_adds_epu8(_src, _mm_and_si128(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
            if (shift < 0)
            {
                __m128i _shift = _mm_set1_epi8((char)-shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        const __m128i _src = Load<align>((__m128i*) (src + col));
                        Store<align>((__m128i*) (dst + col), _mm_subs_epu8(_src, _shift));
                    }
                    if (width != alignedWidth)
                    {
                        const __m128i _src = Load<false>((__m128i*) (src + width - A));
                        Store<false>((__m128i*) (dst + width - A), _mm_subs_epu8(_src, _mm_and_si128(_shift, tailMask)));
                    }
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
#endif// SIMD_SSE2_ENABLE
}
