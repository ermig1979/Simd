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

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template<bool increment> __m128i InterferenceChange(__m128i statistic, __m128i value, __m128i saturation);

        template<> SIMD_INLINE __m128i InterferenceChange<true>(__m128i statistic, __m128i value, __m128i saturation)
        {
            return _mm_min_epi16(_mm_add_epi16(statistic, value), saturation);
        }

        template<> SIMD_INLINE __m128i InterferenceChange<false>(__m128i statistic, __m128i value, __m128i saturation)
        {
            return _mm_max_epi16(_mm_sub_epi16(statistic, value), saturation);
        }

        template<bool align, bool increment> SIMD_INLINE void InterferenceChange(int16_t * statistic, __m128i value, __m128i saturation)
        {
            Store<align>((__m128i*)statistic, InterferenceChange<increment>(Load<align>((__m128i*)statistic), value, saturation));
        }

        template <bool align, bool increment> void InterferenceChange(int16_t * statistic, size_t stride, size_t width, size_t height, uint8_t value, int16_t saturation)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(statistic) && Aligned(stride, HA));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, 2 * (HA - width + alignedWidth));

            __m128i _value = _mm_set1_epi16(value);
            __m128i _saturation = _mm_set1_epi16(saturation);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                    InterferenceChange<align, increment>(statistic + col, _value, _saturation);
                if (alignedWidth != width)
                    InterferenceChange<false, increment>(statistic + width - HA, _mm_and_si128(_value, tailMask), _saturation);
                statistic += stride;
            }
        }

        void InterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
        {
            assert(Aligned(stride, 2));

            if (Aligned(statistic) && Aligned(stride))
                InterferenceChange<true, true>((int16_t*)statistic, stride / 2, width, height, increment, saturation);
            else
                InterferenceChange<false, true>((int16_t*)statistic, stride / 2, width, height, increment, saturation);
        }

        void InterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation)
        {
            assert(Aligned(stride, 2));

            if (Aligned(statistic) && Aligned(stride))
                InterferenceChange<true, false>((int16_t*)statistic, stride / 2, width, height, decrement, saturation);
            else
                InterferenceChange<false, false>((int16_t*)statistic, stride / 2, width, height, decrement, saturation);
        }

        template <bool align, bool increment> void InterferenceChangeMasked(int16_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t value, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(statistic) && Aligned(statisticStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);

            __m128i _value = _mm_set1_epi16(value);
            __m128i _saturation = _mm_set1_epi16(saturation);
            __m128i _index = _mm_set1_epi8(index);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _mask = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask + col)), _index);
                    InterferenceChange<align, increment>(statistic + col, _mm_and_si128(_value, _mm_unpacklo_epi8(_mask, _mask)), _saturation);
                    InterferenceChange<align, increment>(statistic + col + HA, _mm_and_si128(_value, _mm_unpackhi_epi8(_mask, _mask)), _saturation);
                }
                if (alignedWidth != width)
                {
                    __m128i _mask = _mm_and_si128(_mm_cmpeq_epi8(Load<false>((__m128i*)(mask + width - A)), _index), tailMask);
                    InterferenceChange<false, increment>(statistic + width - A, _mm_and_si128(_value, _mm_unpacklo_epi8(_mask, _mask)), _saturation);
                    InterferenceChange<false, increment>(statistic + width - HA, _mm_and_si128(_value, _mm_unpackhi_epi8(_mask, _mask)), _saturation);
                }
                statistic += statisticStride;
                mask += maskStride;
            }
        }

        void InterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(Aligned(statisticStride, 2));

            if (Aligned(statistic) && Aligned(statisticStride) && Aligned(mask) && Aligned(maskStride))
                InterferenceChangeMasked<true, true>((int16_t*)statistic, statisticStride / 2, width, height, increment, saturation, mask, maskStride, index);
            else
                InterferenceChangeMasked<false, true>((int16_t*)statistic, statisticStride / 2, width, height, increment, saturation, mask, maskStride, index);
        }

        void InterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(Aligned(statisticStride, 2));

            if (Aligned(statistic) && Aligned(statisticStride) && Aligned(mask) && Aligned(maskStride))
                InterferenceChangeMasked<true, false>((int16_t*)statistic, statisticStride / 2, width, height, decrement, saturation, mask, maskStride, index);
            else
                InterferenceChangeMasked<false, false>((int16_t*)statistic, statisticStride / 2, width, height, decrement, saturation, mask, maskStride, index);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
