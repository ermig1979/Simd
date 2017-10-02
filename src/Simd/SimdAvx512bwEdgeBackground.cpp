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
#include "Simd/SimdCompare.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const uint8_t * value, uint8_t * background, __mmask64 m = -1)
        {
            const __m512i _value = Load<align, mask>(value, m);
            const __m512i _background = Load<align, mask>(background, m);
            const __mmask64 inc = _mm512_cmpgt_epu8_mask(_value, _background);
            Store<align, mask>(background, _mm512_mask_adds_epu8(_background, inc, _background, K8_01), m);
        }

        template <bool align> void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeSlow<align, false>(value + col, background + col);
                if (col < width)
                    EdgeBackgroundGrowRangeSlow<align, true>(value + col, background + col, tailMask);
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundGrowRangeSlow<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundGrowRangeSlow<false>(value, valueStride, width, height, background, backgroundStride);
        }

        template <bool align, bool mask> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const uint8_t * value, uint8_t * background, __mmask64 m = -1)
        {
            const __m512i _value = Load<align, mask>(value, m);
            const __m512i _background = Load<align, mask>(background, m);
            Store<align, mask>(background, _mm512_max_epu8(_background, _value), m);
        }

        template <bool align> void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeFast<align, false>(value + col, background + col);
                if (col < width)
                    EdgeBackgroundGrowRangeFast<align, true>(value + col, background + col, tailMask);
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundGrowRangeFast<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundGrowRangeFast<false>(value, valueStride, width, height, background, backgroundStride);
        }

        template <bool align, bool mask> SIMD_INLINE void EdgeBackgroundIncrementCount(const uint8_t * value,
            const uint8_t * backgroundValue, uint8_t * backgroundCount, size_t offset, __mmask64 m = -1)
        {
            const __m512i _value = Load<align, mask>(value + offset, m);
            const __m512i _backgroundValue = Load<align, mask>(backgroundValue + offset, m);
            const __m512i _backgroundCount = Load<align, mask>(backgroundCount + offset, m);
            const __mmask64 inc = _mm512_cmpgt_epu8_mask(_value, _backgroundValue);
            Store<align, mask>(backgroundCount + offset, _mm512_mask_adds_epu8(_backgroundCount, inc, _backgroundCount, K8_01), m);
        }

        template <bool align> void EdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
        {
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundIncrementCount<align, false>(value, backgroundValue, backgroundCount, col);
                if (col < width)
                    EdgeBackgroundIncrementCount<align, true>(value, backgroundValue, backgroundCount, col, tailMask);
                value += valueStride;
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        void EdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
        {
            if (Aligned(value) && Aligned(valueStride) &&
                Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride))
                EdgeBackgroundIncrementCount<true>(value, valueStride, width, height,
                    backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
            else
                EdgeBackgroundIncrementCount<false>(value, valueStride, width, height,
                    backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
        }

        SIMD_INLINE __m512i AdjustEdge(const __m512i & count, const __m512i & value, const __m512i & threshold)
        {
            const __mmask64 inc = _mm512_cmpgt_epu8_mask(count, threshold);
            const __mmask64 dec = _mm512_cmplt_epu8_mask(count, threshold);
            __m512i added = _mm512_mask_adds_epu8(value, inc, value, K8_01);
            return _mm512_mask_subs_epu8(added, dec, added, K8_01);
        }

        template <bool align, bool mask> SIMD_INLINE void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, uint8_t * backgroundValue, const __m512i & threshold, __mmask64 m = -1)
        {
            const __m512i _backgroundCount = Load<align, mask>(backgroundCount, m);
            const __m512i _backgroundValue = Load<align, mask>(backgroundValue, m);
            Store<align, mask>(backgroundValue, AdjustEdge(_backgroundCount, _backgroundValue, threshold), m);
            Store<align, mask>(backgroundCount, K_ZERO, m);
        }

        template <bool align> void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
        {
            if (align)
            {
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
            }

            const __m512i _threshold = _mm512_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRange<align, false>(backgroundCount + col, backgroundValue + col, _threshold);
                if (col < width)
                    EdgeBackgroundAdjustRange<align, true>(backgroundCount + col, backgroundValue + col, _threshold, tailMask);
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
        {
            if (Aligned(backgroundValue) && Aligned(backgroundValueStride) &&
                Aligned(backgroundCount) && Aligned(backgroundCountStride))
                EdgeBackgroundAdjustRange<true>(backgroundCount, backgroundCountStride, width, height,
                    backgroundValue, backgroundValueStride, threshold);
            else
                EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundCountStride, width, height,
                    backgroundValue, backgroundValueStride, threshold);
        }

        template <bool align, bool mask> SIMD_INLINE void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, uint8_t * backgroundValue,
            const uint8_t * pmask, const __m512i & threshold, __mmask64 m = -1)
        {
            const __m512i _mask = Load<align, mask>(pmask, m);
            const __mmask64 mm = _mm512_cmpneq_epu8_mask(_mask, K_ZERO) & m;

            const __m512i _backgroundCount = Load<align, mask>(backgroundCount, m);
            const __m512i _backgroundValue = Load<align, mask>(backgroundValue, m);

            Store<align, true>(backgroundValue, AdjustEdge(_backgroundCount, _backgroundValue, threshold), mm);
            Store<align, mask>(backgroundCount, K_ZERO, m);
        }

        template <bool align> void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            if (align)
            {
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            const __m512i _threshold = _mm512_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRangeMasked<align, false>(backgroundCount + col, backgroundValue + col, mask + col, _threshold);
                if (col < width)
                    EdgeBackgroundAdjustRangeMasked<align, true>(backgroundCount + col, backgroundValue + col, mask + col, _threshold, tailMask);
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
                mask += maskStride;
            }
        }

        void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            if (Aligned(backgroundValue) && Aligned(backgroundValueStride) &&
                Aligned(backgroundCount) && Aligned(backgroundCountStride) &&
                Aligned(mask) && Aligned(maskStride))
                EdgeBackgroundAdjustRangeMasked<true>(backgroundCount, backgroundCountStride, width, height,
                    backgroundValue, backgroundValueStride, threshold, mask, maskStride);
            else
                EdgeBackgroundAdjustRangeMasked<false>(backgroundCount, backgroundCountStride, width, height,
                    backgroundValue, backgroundValueStride, threshold, mask, maskStride);
        }

        template <bool align, bool masked> SIMD_INLINE void EdgeBackgroundShiftRangeMasked(const uint8_t * value, uint8_t * background, const uint8_t * mask, __mmask64 tail = -1)
        {
            const __m512i _mask = Load<align, masked>(mask, tail);
            const __mmask64 mmask = _mm512_cmpneq_epu8_mask(_mask, K_ZERO) & tail;
            const __m512i _value = Load<align, masked>(value, tail);
            Store<align, true>(background, _value, mmask);
        }

        template <bool align> void EdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
        {
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    EdgeBackgroundShiftRangeMasked<align, false>(value + col, background + col, mask + col);
                if (col < width)
                    EdgeBackgroundShiftRangeMasked<align, true>(value + col, background + col, mask + col, tailMask);
                value += valueStride;
                background += backgroundStride;
                mask += maskStride;
            }
        }

        void EdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride) && Aligned(mask) && Aligned(maskStride))
                EdgeBackgroundShiftRangeMasked<true>(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
            else
                EdgeBackgroundShiftRangeMasked<false>(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
