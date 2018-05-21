/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, __m256i tailMask)
        {
            const __m256i _value = Load<align>((__m256i*)value);
            const __m256i _lo = Load<align>((__m256i*)lo);
            const __m256i _hi = Load<align>((__m256i*)hi);

            const __m256i inc = _mm256_and_si256(tailMask, Greater8u(_value, _hi));
            const __m256i dec = _mm256_and_si256(tailMask, Lesser8u(_value, _lo));

            Store<align>((__m256i*)lo, _mm256_subs_epu8(_lo, dec));
            Store<align>((__m256i*)hi, _mm256_adds_epu8(_hi, inc));
        }

        template <bool align> void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow<align>(value + col, lo + col, hi + col, K8_01);
                if (alignedWidth != width)
                    BackgroundGrowRangeSlow<false>(value + width - A, lo + width - A, hi + width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                BackgroundGrowRangeSlow<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
            else
                BackgroundGrowRangeSlow<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
        }

        template <bool align> SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t * value, uint8_t * lo, uint8_t * hi)
        {
            const __m256i _value = Load<align>((__m256i*)value);
            const __m256i _lo = Load<align>((__m256i*)lo);
            const __m256i _hi = Load<align>((__m256i*)hi);

            Store<align>((__m256i*)lo, _mm256_min_epu8(_lo, _value));
            Store<align>((__m256i*)hi, _mm256_max_epu8(_hi, _value));
        }

        template <bool align> void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundGrowRangeFast<align>(value + col, lo + col, hi + col);
                if (alignedWidth != width)
                    BackgroundGrowRangeFast<false>(value + width - A, lo + width - A, hi + width - A);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                BackgroundGrowRangeFast<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
            else
                BackgroundGrowRangeFast<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
        }

        template <bool align> SIMD_INLINE void BackgroundIncrementCount(const uint8_t * value,
            const uint8_t * loValue, const uint8_t * hiValue, uint8_t * loCount, uint8_t * hiCount, size_t offset, __m256i tailMask)
        {
            const __m256i _value = Load<align>((__m256i*)(value + offset));
            const __m256i _loValue = Load<align>((__m256i*)(loValue + offset));
            const __m256i _loCount = Load<align>((__m256i*)(loCount + offset));
            const __m256i _hiValue = Load<align>((__m256i*)(hiValue + offset));
            const __m256i _hiCount = Load<align>((__m256i*)(hiCount + offset));

            const __m256i incLo = _mm256_and_si256(tailMask, Lesser8u(_value, _loValue));
            const __m256i incHi = _mm256_and_si256(tailMask, Greater8u(_value, _hiValue));

            Store<align>((__m256i*)(loCount + offset), _mm256_adds_epu8(_loCount, incLo));
            Store<align>((__m256i*)(hiCount + offset), _mm256_adds_epu8(_hiCount, incHi));
        }

        template <bool align> void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
            uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
                assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundIncrementCount<align>(value, loValue, hiValue, loCount, hiCount, col, K8_01);
                if (alignedWidth != width)
                    BackgroundIncrementCount<false>(value, loValue, hiValue, loCount, hiCount, width - A, tailMask);
                value += valueStride;
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
            uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
        {
            if (Aligned(value) && Aligned(valueStride) &&
                Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) &&
                Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride))
                BackgroundIncrementCount<true>(value, valueStride, width, height,
                    loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
            else
                BackgroundIncrementCount<false>(value, valueStride, width, height,
                    loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
        }

        SIMD_INLINE __m256i AdjustLo(const __m256i &count, const __m256i & value, const __m256i & mask, const __m256i & threshold)
        {
            const __m256i dec = _mm256_and_si256(mask, Greater8u(count, threshold));
            const __m256i inc = _mm256_and_si256(mask, Lesser8u(count, threshold));
            return _mm256_subs_epu8(_mm256_adds_epu8(value, inc), dec);
        }

        SIMD_INLINE __m256i AdjustHi(const __m256i &count, const __m256i & value, const __m256i & mask, const __m256i & threshold)
        {
            const __m256i inc = _mm256_and_si256(mask, Greater8u(count, threshold));
            const __m256i dec = _mm256_and_si256(mask, Lesser8u(count, threshold));
            return _mm256_subs_epu8(_mm256_adds_epu8(value, inc), dec);
        }

        template <bool align> SIMD_INLINE void BackgroundAdjustRange(uint8_t * loCount, uint8_t * loValue,
            uint8_t * hiCount, uint8_t * hiValue, size_t offset, const __m256i & threshold, const __m256i & mask)
        {
            const __m256i _loCount = Load<align>((__m256i*)(loCount + offset));
            const __m256i _loValue = Load<align>((__m256i*)(loValue + offset));
            const __m256i _hiCount = Load<align>((__m256i*)(hiCount + offset));
            const __m256i _hiValue = Load<align>((__m256i*)(hiValue + offset));

            Store<align>((__m256i*)(loValue + offset), AdjustLo(_loCount, _loValue, mask, threshold));
            Store<align>((__m256i*)(hiValue + offset), AdjustHi(_hiCount, _hiValue, mask, threshold));
            Store<align>((__m256i*)(loCount + offset), K_ZERO);
            Store<align>((__m256i*)(hiCount + offset), K_ZERO);
        }

        template <bool align> void BackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
                assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
            }

            const __m256i _threshold = _mm256_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, col, _threshold, K8_01);
                if (alignedWidth != width)
                    BackgroundAdjustRange<false>(loCount, loValue, hiCount, hiValue, width - A, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        void BackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
        {
            if (Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) &&
                Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride))
                BackgroundAdjustRange<true>(loCount, loCountStride, width, height, loValue, loValueStride,
                    hiCount, hiCountStride, hiValue, hiValueStride, threshold);
            else
                BackgroundAdjustRange<false>(loCount, loCountStride, width, height, loValue, loValueStride,
                    hiCount, hiCountStride, hiValue, hiValueStride, threshold);
        }

        template <bool align> SIMD_INLINE void BackgroundAdjustRangeMasked(uint8_t * loCount, uint8_t * loValue, uint8_t * hiCount, uint8_t * hiValue,
            const uint8_t * mask, size_t offset, const __m256i & threshold, const __m256i & tailMask)
        {
            const __m256i _mask = Load<align>((const __m256i*)(mask + offset));
            BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, offset, threshold, _mm256_and_si256(_mask, tailMask));
        }

        template <bool align> void BackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
                assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            const __m256i _threshold = _mm256_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundAdjustRangeMasked<align>(loCount, loValue, hiCount, hiValue, mask, col, _threshold, K8_01);
                if (alignedWidth != width)
                    BackgroundAdjustRangeMasked<false>(loCount, loValue, hiCount, hiValue, mask, width - A, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
                mask += maskStride;
            }
        }

        void BackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            if (Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) &&
                Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride) &&
                Aligned(mask) && Aligned(maskStride))
                BackgroundAdjustRangeMasked<true>(loCount, loCountStride, width, height, loValue, loValueStride,
                    hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
            else
                BackgroundAdjustRangeMasked<false>(loCount, loCountStride, width, height, loValue, loValueStride,
                    hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
        }

        template <bool align> SIMD_INLINE void BackgroundShiftRange(const uint8_t * value, uint8_t * lo, uint8_t * hi, size_t offset, __m256i mask)
        {
            const __m256i _value = Load<align>((__m256i*)(value + offset));
            const __m256i _lo = Load<align>((__m256i*)(lo + offset));
            const __m256i _hi = Load<align>((__m256i*)(hi + offset));

            const __m256i add = _mm256_and_si256(mask, _mm256_subs_epu8(_value, _hi));
            const __m256i sub = _mm256_and_si256(mask, _mm256_subs_epu8(_lo, _value));

            Store<align>((__m256i*)(lo + offset), _mm256_subs_epu8(_mm256_adds_epu8(_lo, add), sub));
            Store<align>((__m256i*)(hi + offset), _mm256_subs_epu8(_mm256_adds_epu8(_hi, add), sub));
        }

        template <bool align> void BackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRange<align>(value, lo, hi, col, K_INV_ZERO);
                if (alignedWidth != width)
                    BackgroundShiftRange<false>(value, lo, hi, width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void BackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                BackgroundShiftRange<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
            else
                BackgroundShiftRange<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
        }

        template <bool align> SIMD_INLINE void BackgroundShiftRangeMasked(const uint8_t * value, uint8_t * lo, uint8_t * hi, const uint8_t * mask,
            size_t offset, __m256i tailMask)
        {
            const __m256i _mask = Load<align>((const __m256i*)(mask + offset));
            BackgroundShiftRange<align>(value, lo, hi, offset, _mm256_and_si256(_mask, tailMask));
        }

        template <bool align> void BackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRangeMasked<align>(value, lo, hi, mask, col, K_INV_ZERO);
                if (alignedWidth != width)
                    BackgroundShiftRangeMasked<false>(value, lo, hi, mask, width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                mask += maskStride;
            }
        }

        void BackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) &&
                Aligned(hi) && Aligned(hiStride) && Aligned(mask) && Aligned(maskStride))
                BackgroundShiftRangeMasked<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
            else
                BackgroundShiftRangeMasked<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
        }

        template <bool align> SIMD_INLINE void BackgroundInitMask(const uint8_t * src, uint8_t * dst, const __m256i & index, const __m256i & value)
        {
            __m256i _mask = _mm256_cmpeq_epi8(Load<align>((__m256i*)src), index);
            __m256i _old = _mm256_andnot_si256(_mask, Load<align>((__m256i*)dst));
            __m256i _new = _mm256_and_si256(_mask, value);
            Store<align>((__m256i*)dst, _mm256_or_si256(_old, _new));
        }

        template <bool align> void BackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i _index = _mm256_set1_epi8(index);
            __m256i _value = _mm256_set1_epi8(value);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundInitMask<align>(src + col, dst + col, _index, _value);
                if (alignedWidth != width)
                    BackgroundInitMask<false>(src + width - A, dst + width - A, _index, _value);
                src += srcStride;
                dst += dstStride;
            }
        }

        void BackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                BackgroundInitMask<true>(src, srcStride, width, height, index, value, dst, dstStride);
            else
                BackgroundInitMask<false>(src, srcStride, width, height, index, value, dst, dstStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
