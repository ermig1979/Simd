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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool increment> __m512i InterferenceChange(__m512i statistic, __m512i value, __m512i saturation);

        template<> SIMD_INLINE __m512i InterferenceChange<true>(__m512i statistic, __m512i value, __m512i saturation)
        {
            return _mm512_min_epi16(_mm512_add_epi16(statistic, value), saturation);
        }

        template<> SIMD_INLINE __m512i InterferenceChange<false>(__m512i statistic, __m512i value, __m512i saturation)
        {
            return _mm512_max_epi16(_mm512_sub_epi16(statistic, value), saturation);
        }

        template<bool align, bool increment, bool mask> SIMD_INLINE void InterferenceChange(int16_t * statistic, __m512i value, __m512i saturation, __mmask32 tail = -1)
        {
            Store<align, mask>(statistic, InterferenceChange<increment>(Load<align, mask>(statistic, tail), value, saturation), tail);
        }

        template<bool align, bool increment> SIMD_INLINE void InterferenceChange4(int16_t * statistic, __m512i value, __m512i saturation)
        {
            Store<align>(statistic + 0 * HA, InterferenceChange<increment>(Load<align>(statistic + 0 * HA), value, saturation));
            Store<align>(statistic + 1 * HA, InterferenceChange<increment>(Load<align>(statistic + 1 * HA), value, saturation));
            Store<align>(statistic + 2 * HA, InterferenceChange<increment>(Load<align>(statistic + 2 * HA), value, saturation));
            Store<align>(statistic + 3 * HA, InterferenceChange<increment>(Load<align>(statistic + 3 * HA), value, saturation));
        }

        template <bool align, bool increment> void InterferenceChange(int16_t * statistic, size_t stride, size_t width, size_t height, uint8_t value, int16_t saturation)
        {
            if (align)
                assert(Aligned(statistic) && Aligned(stride, HA));

            size_t alignedWidth = AlignLo(width, HA);
            size_t fullAlignedWidth = AlignLo(width, DA);
            __mmask32 tailMask = TailMask32(width - alignedWidth);

            __m512i _value = _mm512_set1_epi16(value);
            __m512i _saturation = _mm512_set1_epi16(saturation);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    InterferenceChange4<align, increment>(statistic + col, _value, _saturation);
                for (; col < alignedWidth; col += HA)
                    InterferenceChange<align, increment, false>(statistic + col, _value, _saturation);
                if (col < width)
                    InterferenceChange<false, increment, true>(statistic + col, _value, _saturation, tailMask);
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

        template<bool increment> __m512i InterferenceChangeMasked(__m512i statistic, __m512i value, __m512i saturation, __mmask32 mask);

        template<> SIMD_INLINE __m512i InterferenceChangeMasked<true>(__m512i statistic, __m512i value, __m512i saturation, __mmask32 mask)
        {
            return _mm512_min_epi16(_mm512_mask_add_epi16(statistic, mask, statistic, value), saturation);
        }

        template<> SIMD_INLINE __m512i InterferenceChangeMasked<false>(__m512i statistic, __m512i value, __m512i saturation, __mmask32 mask)
        {
            return _mm512_max_epi16(_mm512_mask_sub_epi16(statistic, mask, statistic, value), saturation);
        }

        template<bool align, bool increment, bool masked> SIMD_INLINE void InterferenceChangeMasked(int16_t * statistic, __m512i value, __m512i saturation, __mmask32 mask, __mmask32 tail = -1)
        {
            Store<align, masked>(statistic, InterferenceChangeMasked<increment>(Load<align, masked>(statistic, tail), value, saturation, mask), tail);
        }

        template<bool align, bool increment, bool masked> SIMD_INLINE void InterferenceChangeMasked(const uint8_t * mask, __m512i index, int16_t * statistic, __m512i value, __m512i saturation, __mmask64 tail = -1)
        {
            __mmask64 mask0 = _mm512_cmpeq_epi8_mask((Load<align, masked>(mask, tail)), index) & tail;
            InterferenceChangeMasked<align, increment, masked>(statistic + 00, value, saturation, __mmask32(mask0 >> 00), __mmask32(tail >> 00));
            InterferenceChangeMasked<align, increment, masked>(statistic + HA, value, saturation, __mmask32(mask0 >> 32), __mmask32(tail >> 32));
        }

        template<bool align, bool increment> SIMD_INLINE void InterferenceChangeMasked(int16_t * statistic, __m512i value, __m512i saturation, __mmask32 mask)
        {
            Store<align>(statistic, InterferenceChangeMasked<increment>(Load<align>(statistic), value, saturation, mask));
        }

        template<bool align, bool increment> SIMD_INLINE void InterferenceChangeMasked2(const uint8_t * mask, __m512i index, int16_t * statistic, __m512i value, __m512i saturation)
        {
            __mmask64 mask0 = _mm512_cmpeq_epi8_mask(Load<align>(mask + 0), index);
            InterferenceChangeMasked<align, increment>(statistic + 0 * HA, value, saturation, __mmask32(mask0 >> 00));
            InterferenceChangeMasked<align, increment>(statistic + 1 * HA, value, saturation, __mmask32(mask0 >> 32));
            __mmask64 mask1 = _mm512_cmpeq_epi8_mask(Load<align>(mask + A), index);
            InterferenceChangeMasked<align, increment>(statistic + 2 * HA, value, saturation, __mmask32(mask1 >> 00));
            InterferenceChangeMasked<align, increment>(statistic + 3 * HA, value, saturation, __mmask32(mask1 >> 32));
        }

        template <bool align, bool increment> void InterferenceChangeMasked(int16_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t value, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            if (align)
                assert(Aligned(statistic) && Aligned(statisticStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, DA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi16(value);
            __m512i _saturation = _mm512_set1_epi16(saturation);
            __m512i _index = _mm512_set1_epi8(index);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    InterferenceChangeMasked2<align, increment>(mask + col, _index, statistic + col, _value, _saturation);
                for (; col < alignedWidth; col += A)
                    InterferenceChangeMasked<align, increment, false>(mask + col, _index, statistic + col, _value, _saturation);
                if (col < width)
                    InterferenceChangeMasked<align, increment, true>(mask + col, _index, statistic + col, _value, _saturation, tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
