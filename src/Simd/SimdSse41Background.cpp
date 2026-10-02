/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, __m128i tailMask)
        {
            const __m128i _value = _mm_loadu_si128((__m128i*)value);
            const __m128i _lo = _mm_loadu_si128((__m128i*)lo);
            const __m128i _hi = _mm_loadu_si128((__m128i*)hi);

            const __m128i inc = _mm_and_si128(tailMask, Greater8u(_value, _hi));
            const __m128i dec = _mm_and_si128(tailMask, Lesser8u(_value, _lo));

            _mm_storeu_si128((__m128i*)lo, _mm_subs_epu8(_lo, dec));
            _mm_storeu_si128((__m128i*)hi, _mm_adds_epu8(_hi, inc));
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow(value + col, lo + col, hi + col, K8_01);
                if (alignedWidth != width)
                    BackgroundGrowRangeSlow(value + width - A, lo + width - A, hi + width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t * value, uint8_t * lo, uint8_t * hi)
        {
            const __m128i _value = _mm_loadu_si128((__m128i*)value);
            const __m128i _lo = _mm_loadu_si128((__m128i*)lo);
            const __m128i _hi = _mm_loadu_si128((__m128i*)hi);

            _mm_storeu_si128((__m128i*)lo, _mm_min_epu8(_lo, _value));
            _mm_storeu_si128((__m128i*)hi, _mm_max_epu8(_hi, _value));
        }

        void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundGrowRangeFast(value + col, lo + col, hi + col);
                if (alignedWidth != width)
                    BackgroundGrowRangeFast(value + width - A, lo + width - A, hi + width - A);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundIncrementCount(const uint8_t * value,
            const uint8_t * loValue, const uint8_t * hiValue, uint8_t * loCount, uint8_t * hiCount, size_t offset, __m128i tailMask)
        {
            const __m128i _value = _mm_loadu_si128((__m128i*)(value + offset));
            const __m128i _loValue = _mm_loadu_si128((__m128i*)(loValue + offset));
            const __m128i _loCount = _mm_loadu_si128((__m128i*)(loCount + offset));
            const __m128i _hiValue = _mm_loadu_si128((__m128i*)(hiValue + offset));
            const __m128i _hiCount = _mm_loadu_si128((__m128i*)(hiCount + offset));

            const __m128i incLo = _mm_and_si128(tailMask, Lesser8u(_value, _loValue));
            const __m128i incHi = _mm_and_si128(tailMask, Greater8u(_value, _hiValue));

            _mm_storeu_si128((__m128i*)(loCount + offset), _mm_adds_epu8(_loCount, incLo));
            _mm_storeu_si128((__m128i*)(hiCount + offset), _mm_adds_epu8(_hiCount, incHi));
        }

        void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
            uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundIncrementCount(value, loValue, hiValue, loCount, hiCount, col, K8_01);
                if (alignedWidth != width)
                    BackgroundIncrementCount(value, loValue, hiValue, loCount, hiCount, width - A, tailMask);
                value += valueStride;
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128i AdjustLo(const __m128i & count, const __m128i & value, const __m128i & mask, const __m128i & threshold)
        {
            const __m128i dec = _mm_and_si128(mask, Greater8u(count, threshold));
            const __m128i inc = _mm_and_si128(mask, Lesser8u(count, threshold));
            return _mm_subs_epu8(_mm_adds_epu8(value, inc), dec);
        }

        SIMD_INLINE __m128i AdjustHi(const __m128i & count, const __m128i & value, const __m128i & mask, const __m128i & threshold)
        {
            const __m128i inc = _mm_and_si128(mask, Greater8u(count, threshold));
            const __m128i dec = _mm_and_si128(mask, Lesser8u(count, threshold));
            return _mm_subs_epu8(_mm_adds_epu8(value, inc), dec);
        }

        SIMD_INLINE void BackgroundAdjustRange(uint8_t * loCount, uint8_t * loValue,
            uint8_t * hiCount, uint8_t * hiValue, size_t offset, const __m128i & threshold, const __m128i & mask)
        {
            const __m128i _loCount = _mm_loadu_si128((__m128i*)(loCount + offset));
            const __m128i _loValue = _mm_loadu_si128((__m128i*)(loValue + offset));
            const __m128i _hiCount = _mm_loadu_si128((__m128i*)(hiCount + offset));
            const __m128i _hiValue = _mm_loadu_si128((__m128i*)(hiValue + offset));

            _mm_storeu_si128((__m128i*)(loValue + offset), AdjustLo(_loCount, _loValue, mask, threshold));
            _mm_storeu_si128((__m128i*)(hiValue + offset), AdjustHi(_hiCount, _hiValue, mask, threshold));
            _mm_storeu_si128((__m128i*)(loCount + offset), K_ZERO);
            _mm_storeu_si128((__m128i*)(hiCount + offset), K_ZERO);
        }

        void BackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
        {
            assert(width >= A);

            const __m128i _threshold = _mm_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundAdjustRange(loCount, loValue, hiCount, hiValue, col, _threshold, K8_01);
                if (alignedWidth != width)
                    BackgroundAdjustRange(loCount, loValue, hiCount, hiValue, width - A, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundAdjustRangeMasked(uint8_t * loCount, uint8_t * loValue, uint8_t * hiCount, uint8_t * hiValue,
            const uint8_t * mask, size_t offset, const __m128i & threshold, const __m128i & tailMask)
        {
            const __m128i _mask = _mm_loadu_si128((const __m128i*)(mask + offset));
            BackgroundAdjustRange(loCount, loValue, hiCount, hiValue, offset, threshold, _mm_and_si128(_mask, tailMask));
        }

        void BackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);

            const __m128i _threshold = _mm_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundAdjustRangeMasked(loCount, loValue, hiCount, hiValue, mask, col, _threshold, K8_01);
                if (alignedWidth != width)
                    BackgroundAdjustRangeMasked(loCount, loValue, hiCount, hiValue, mask, width - A, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
                mask += maskStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundShiftRange(const uint8_t * value, uint8_t * lo, uint8_t * hi, size_t offset, __m128i mask)
        {
            const __m128i _value = _mm_loadu_si128((__m128i*)(value + offset));
            const __m128i _lo = _mm_loadu_si128((__m128i*)(lo + offset));
            const __m128i _hi = _mm_loadu_si128((__m128i*)(hi + offset));

            const __m128i add = _mm_and_si128(mask, _mm_subs_epu8(_value, _hi));
            const __m128i sub = _mm_and_si128(mask, _mm_subs_epu8(_lo, _value));

            _mm_storeu_si128((__m128i*)(lo + offset), _mm_subs_epu8(_mm_adds_epu8(_lo, add), sub));
            _mm_storeu_si128((__m128i*)(hi + offset), _mm_subs_epu8(_mm_adds_epu8(_hi, add), sub));
        }

        void BackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRange(value, lo, hi, col, K_INV_ZERO);
                if (alignedWidth != width)
                    BackgroundShiftRange(value, lo, hi, width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundShiftRangeMasked(const uint8_t * value, uint8_t * lo, uint8_t * hi, const uint8_t * mask,
            size_t offset, __m128i tailMask)
        {
            const __m128i _mask = _mm_loadu_si128((const __m128i*)(mask + offset));
            BackgroundShiftRange(value, lo, hi, offset, _mm_and_si128(_mask, tailMask));
        }

        void BackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRangeMasked(value, lo, hi, mask, col, K_INV_ZERO);
                if (alignedWidth != width)
                    BackgroundShiftRangeMasked(value, lo, hi, mask, width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                mask += maskStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundInitMask(const uint8_t * src, uint8_t * dst, const __m128i & index, const __m128i & value)
        {
            __m128i _mask = _mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)src), index);
            __m128i _old = _mm_andnot_si128(_mask, _mm_loadu_si128((__m128i*)dst));
            __m128i _new = _mm_and_si128(_mask, value);
            _mm_storeu_si128((__m128i*)dst, _mm_or_si128(_old, _new));
        }

        void BackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            __m128i _index = _mm_set1_epi8(index);
            __m128i _value = _mm_set1_epi8(value);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundInitMask(src + col, dst + col, _index, _value);
                if (alignedWidth != width)
                    BackgroundInitMask(src + width - A, dst + width - A, _index, _value);
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
