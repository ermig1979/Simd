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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const uint8_t * value, uint8_t * background, __m256i tailMask)
        {
            const __m256i _value = Load<align>((__m256i*)value);
            const __m256i _background = Load<align>((__m256i*)background);
            const __m256i inc = _mm256_and_si256(tailMask, Greater8u(_value, _background));
            Store<align>((__m256i*)background, _mm256_adds_epu8(_background, inc));
        }

        template <bool align> void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeSlow<align>(value + col, background + col, K8_01);
                if (alignedWidth != width)
                    EdgeBackgroundGrowRangeSlow<false>(value + width - A, background + width - A, tailMask);
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

        template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const uint8_t * value, uint8_t * background)
        {
            const __m256i _value = Load<align>((__m256i*)value);
            const __m256i _background = Load<align>((__m256i*)background);
            Store<align>((__m256i*)background, _mm256_max_epu8(_background, _value));
        }

        template <bool align> void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeFast<align>(value + col, background + col);
                if (alignedWidth != width)
                    EdgeBackgroundGrowRangeFast<false>(value + width - A, background + width - A);
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

        template <bool align> SIMD_INLINE void EdgeBackgroundIncrementCount(const uint8_t * value,
            const uint8_t * backgroundValue, uint8_t * backgroundCount, size_t offset, __m256i tailMask)
        {
            const __m256i _value = Load<align>((__m256i*)(value + offset));
            const __m256i _backgroundValue = Load<align>((__m256i*)(backgroundValue + offset));
            const __m256i _backgroundCount = Load<align>((__m256i*)(backgroundCount + offset));

            const __m256i inc = _mm256_and_si256(tailMask, Greater8u(_value, _backgroundValue));

            Store<align>((__m256i*)(backgroundCount + offset), _mm256_adds_epu8(_backgroundCount, inc));
        }

        template <bool align> void EdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundIncrementCount<align>(value, backgroundValue, backgroundCount, col, K8_01);
                if (alignedWidth != width)
                    EdgeBackgroundIncrementCount<false>(value, backgroundValue, backgroundCount, width - A, tailMask);
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

        SIMD_INLINE __m256i AdjustEdge(const __m256i &count, const __m256i & value, const __m256i & mask, const __m256i & threshold)
        {
            const __m256i inc = _mm256_and_si256(mask, Greater8u(count, threshold));
            const __m256i dec = _mm256_and_si256(mask, Lesser8u(count, threshold));
            return _mm256_subs_epu8(_mm256_adds_epu8(value, inc), dec);
        }

        template <bool align> SIMD_INLINE void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, uint8_t * backgroundValue,
            size_t offset, const __m256i & threshold, const __m256i & mask)
        {
            const __m256i _backgroundCount = Load<align>((__m256i*)(backgroundCount + offset));
            const __m256i _backgroundValue = Load<align>((__m256i*)(backgroundValue + offset));

            Store<align>((__m256i*)(backgroundValue + offset), AdjustEdge(_backgroundCount, _backgroundValue, mask, threshold));
            Store<align>((__m256i*)(backgroundCount + offset), K_ZERO);
        }

        template <bool align> void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
            }

            const __m256i _threshold = _mm256_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRange<align>(backgroundCount, backgroundValue, col, _threshold, K8_01);
                if (alignedWidth != width)
                    EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundValue, width - A, _threshold, tailMask);
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

        template <bool align> SIMD_INLINE void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, uint8_t * backgroundValue,
            const uint8_t * mask, size_t offset, const __m256i & threshold, const __m256i & tailMask)
        {
            const __m256i _mask = Load<align>((const __m256i*)(mask + offset));
            EdgeBackgroundAdjustRange<align>(backgroundCount, backgroundValue, offset, threshold, _mm256_and_si256(_mask, tailMask));
        }

        template <bool align> void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
                assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            const __m256i _threshold = _mm256_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRangeMasked<align>(backgroundCount, backgroundValue, mask, col, _threshold, K8_01);
                if (alignedWidth != width)
                    EdgeBackgroundAdjustRangeMasked<false>(backgroundCount, backgroundValue, mask, width - A, _threshold, tailMask);
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

        template <bool align> SIMD_INLINE void EdgeBackgroundShiftRangeMasked(const uint8_t * value, uint8_t * background, const uint8_t * mask, size_t offset)
        {
            const __m256i _value = Load<align>((__m256i*)(value + offset));
            const __m256i _background = Load<align>((__m256i*)(background + offset));
            const __m256i _mask = Load<align>((const __m256i*)(mask + offset));
            Store<align>((__m256i*)(background + offset), _mm256_or_si256(_mm256_and_si256(_mask, _value), _mm256_andnot_si256(_mask, _background)));
        }

        template <bool align> void EdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundShiftRangeMasked<align>(value, background, mask, col);
                if (alignedWidth != width)
                    EdgeBackgroundShiftRangeMasked<false>(value, background, mask, width - A);
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
#endif// SIMD_AVX2_ENABLE
}
