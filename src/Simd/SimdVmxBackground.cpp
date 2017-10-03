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
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align, bool first>
        SIMD_INLINE void BackgroundGrowRangeSlow(const Loader<align> & value, const Loader<align> & loSrc, const Loader<align> &  hiSrc,
            v128_u8 mask, Storer<align> & loDst, Storer<align> &  hiDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _lo = Load<align, first>(loSrc);
            const v128_u8 _hi = Load<align, first>(hiSrc);

            const v128_u8 inc = vec_and(mask, vec_cmpgt(_value, _hi));
            const v128_u8 dec = vec_and(mask, vec_cmplt(_value, _lo));

            Store<align, first>(loDst, vec_subs(_lo, dec));
            Store<align, first>(hiDst, vec_adds(_hi, inc));
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
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _loSrc(lo), _hiSrc(hi);
                Storer<align> _loDst(lo), _hiDst(hi);
                BackgroundGrowRangeSlow<align, true>(_value, _loSrc, _hiSrc, K8_01, _loDst, _hiDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow<align, false>(_value, _loSrc, _hiSrc, K8_01, _loDst, _hiDst);
                Flush(_loDst, _hiDst);
                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _loSrc(lo + width - A), _hiSrc(hi + width - A);
                    Storer<false> _loDst(lo + width - A), _hiDst(hi + width - A);
                    BackgroundGrowRangeSlow<false, true>(_value, _loSrc, _hiSrc, tailMask, _loDst, _hiDst);
                    Flush(_loDst, _hiDst);
                }
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

        template <bool align, bool first>
        SIMD_INLINE void BackgroundGrowRangeFast(const Loader<align> & value, const Loader<align> & loSrc, const Loader<align> &  hiSrc,
            Storer<align> & loDst, Storer<align> &  hiDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _lo = Load<align, first>(loSrc);
            const v128_u8 _hi = Load<align, first>(hiSrc);

            Store<align, first>(loDst, vec_min(_lo, _value));
            Store<align, first>(hiDst, vec_max(_hi, _value));
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
                Loader<align> _value(value), _loSrc(lo), _hiSrc(hi);
                Storer<align> _loDst(lo), _hiDst(hi);
                BackgroundGrowRangeFast<align, true>(_value, _loSrc, _hiSrc, _loDst, _hiDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundGrowRangeFast<align, false>(_value, _loSrc, _hiSrc, _loDst, _hiDst);
                Flush(_loDst, _hiDst);
                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _loSrc(lo + width - A), _hiSrc(hi + width - A);
                    Storer<false> _loDst(lo + width - A), _hiDst(hi + width - A);
                    BackgroundGrowRangeFast<false, true>(_value, _loSrc, _hiSrc, _loDst, _hiDst);
                    Flush(_loDst, _hiDst);
                }
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

        template <bool align, bool first>
        SIMD_INLINE void BackgroundIncrementCount(const Loader<align> & value, const Loader<align> & loValue, const Loader<align> & hiValue,
            const Loader<align> & loCountSrc, const Loader<align> & hiCountSrc, v128_u8 mask, Storer<align> & loCountDst, Storer<align> & hiCountDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _loValue = Load<align, first>(loValue);
            const v128_u8 _loCount = Load<align, first>(loCountSrc);
            const v128_u8 _hiValue = Load<align, first>(hiValue);
            const v128_u8 _hiCount = Load<align, first>(hiCountSrc);

            const v128_u8 incLo = vec_and(mask, vec_cmplt(_value, _loValue));
            const v128_u8 incHi = vec_and(mask, vec_cmpgt(_value, _hiValue));

            Store<align, first>(loCountDst, vec_adds(_loCount, incLo));
            Store<align, first>(hiCountDst, vec_adds(_hiCount, incHi));
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
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _loValue(loValue), _hiValue(hiValue), _loCountSrc(loCount), _hiCountSrc(hiCount);
                Storer<align> _loCountDst(loCount), _hiCountDst(hiCount);
                BackgroundIncrementCount<align, true>(_value, _loValue, _hiValue, _loCountSrc, _hiCountSrc, K8_01, _loCountDst, _hiCountDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundIncrementCount<align, false>(_value, _loValue, _hiValue, _loCountSrc, _hiCountSrc, K8_01, _loCountDst, _hiCountDst);
                Flush(_loCountDst, _hiCountDst);
                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _loValue(loValue + width - A), _hiValue(hiValue + width - A),
                        _loCountSrc(loCount + width - A), _hiCountSrc(hiCount + width - A);
                    Storer<false> _loCountDst(loCount + width - A), _hiCountDst(hiCount + width - A);
                    BackgroundIncrementCount<false, true>(_value, _loValue, _hiValue, _loCountSrc, _hiCountSrc, tailMask, _loCountDst, _hiCountDst);
                    Flush(_loCountDst, _hiCountDst);
                }
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

        SIMD_INLINE v128_u8 AdjustLo(const v128_u8 & count, const v128_u8 & value, const v128_u8 & mask, const v128_u8 & threshold)
        {
            const v128_u8 dec = vec_and(mask, vec_cmpgt(count, threshold));
            const v128_u8 inc = vec_and(mask, vec_cmplt(count, threshold));
            return vec_subs(vec_adds(value, inc), dec);
        }

        SIMD_INLINE v128_u8 AdjustHi(const v128_u8 & count, const v128_u8 & value, const v128_u8 & mask, const v128_u8 & threshold)
        {
            const v128_u8 inc = vec_and(mask, vec_cmpgt(count, threshold));
            const v128_u8 dec = vec_and(mask, vec_cmplt(count, threshold));
            return vec_subs(vec_adds(value, inc), dec);
        }

        template <bool align, bool first>
        SIMD_INLINE void BackgroundAdjustRange(const Loader<align> & loCountSrc, const Loader<align> & loValueSrc,
            const Loader<align> & hiCountSrc, const Loader<align> & hiValueSrc, const v128_u8 & threshold, const v128_u8 & mask,
            Storer<align> & loCountDst, Storer<align> & loValueDst, Storer<align> & hiCountDst, Storer<align> & hiValueDst)
        {
            const v128_u8 _loCount = Load<align, first>(loCountSrc);
            const v128_u8 _loValue = Load<align, first>(loValueSrc);
            const v128_u8 _hiCount = Load<align, first>(hiCountSrc);
            const v128_u8 _hiValue = Load<align, first>(hiValueSrc);

            Store<align, first>(loValueDst, AdjustLo(_loCount, _loValue, mask, threshold));
            Store<align, first>(hiValueDst, AdjustHi(_hiCount, _hiValue, mask, threshold));
            Store<align, first>(loCountDst, K8_00);
            Store<align, first>(hiCountDst, K8_00);
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

            const v128_u8 _threshold = SetU8(threshold);
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _loCountSrc(loCount), _loValueSrc(loValue), _hiCountSrc(hiCount), _hiValueSrc(hiValue);
                Storer<align> _loCountDst(loCount), _loValueDst(loValue), _hiCountDst(hiCount), _hiValueDst(hiValue);
                BackgroundAdjustRange<align, true>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc,
                    _threshold, K8_01, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundAdjustRange<align, false>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc,
                        _threshold, K8_01, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                Flush(_loValueDst, _hiValueDst, _loCountDst, _hiCountDst);

                if (alignedWidth != width)
                {
                    Loader<false> _loCountSrc(loCount + width - A), _loValueSrc(loValue + width - A), _hiCountSrc(hiCount + width - A), _hiValueSrc(hiValue + width - A);
                    Storer<false> _loCountDst(loCount + width - A), _loValueDst(loValue + width - A), _hiCountDst(hiCount + width - A), _hiValueDst(hiValue + width - A);
                    BackgroundAdjustRange<false, true>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc,
                        _threshold, tailMask, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                    Flush(_loValueDst, _hiValueDst, _loCountDst, _hiCountDst);
                }

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

        template <bool align, bool first> SIMD_INLINE void BackgroundAdjustRangeMasked(const Loader<align> & loCountSrc, const Loader<align> & loValueSrc,
            const Loader<align> & hiCountSrc, const Loader<align> & hiValueSrc, const Loader<align> & mask, const v128_u8 & threshold, const v128_u8 & tailMask,
            Storer<align> & loCountDst, Storer<align> & loValueDst, Storer<align> & hiCountDst, Storer<align> & hiValueDst)
        {
            const v128_u8 _mask = vec_and(Load<align, first>(mask), tailMask);
            BackgroundAdjustRange<align, first>(loCountSrc, loValueSrc, hiCountSrc, hiValueSrc, threshold, _mask, loCountDst, loValueDst, hiCountDst, hiValueDst);
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

            const v128_u8 _threshold = SetU8(threshold);
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _loCountSrc(loCount), _loValueSrc(loValue), _hiCountSrc(hiCount), _hiValueSrc(hiValue), _mask(mask);
                Storer<align> _loCountDst(loCount), _loValueDst(loValue), _hiCountDst(hiCount), _hiValueDst(hiValue);
                BackgroundAdjustRangeMasked<align, true>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc, _mask,
                    _threshold, K8_01, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundAdjustRangeMasked<align, false>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc, _mask,
                        _threshold, K8_01, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                Flush(_loValueDst, _hiValueDst, _loCountDst, _hiCountDst);

                if (alignedWidth != width)
                {
                    Loader<false> _loCountSrc(loCount + width - A), _loValueSrc(loValue + width - A),
                        _hiCountSrc(hiCount + width - A), _hiValueSrc(hiValue + width - A), _mask(mask + width - A);
                    Storer<false> _loCountDst(loCount + width - A), _loValueDst(loValue + width - A), _hiCountDst(hiCount + width - A), _hiValueDst(hiValue + width - A);
                    BackgroundAdjustRangeMasked<false, true>(_loCountSrc, _loValueSrc, _hiCountSrc, _hiValueSrc, _mask,
                        _threshold, tailMask, _loCountDst, _loValueDst, _hiCountDst, _hiValueDst);
                    Flush(_loValueDst, _hiValueDst, _loCountDst, _hiCountDst);
                }

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

        template <bool align, bool first>
        SIMD_INLINE void BackgroundShiftRange(const Loader<align> & value, const Loader<align> & loSrc, const Loader<align> &  hiSrc,
            v128_u8 mask, Storer<align> & loDst, Storer<align> &  hiDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _lo = Load<align, first>(loSrc);
            const v128_u8 _hi = Load<align, first>(hiSrc);

            const v128_u8 add = vec_and(mask, vec_subs(_value, _hi));
            const v128_u8 sub = vec_and(mask, vec_subs(_lo, _value));

            Store<align, first>(loDst, vec_subs(vec_adds(_lo, add), sub));
            Store<align, first>(hiDst, vec_subs(vec_adds(_hi, add), sub));
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
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _loSrc(lo), _hiSrc(hi);
                Storer<align> _loDst(lo), _hiDst(hi);
                BackgroundShiftRange<align, true>(_value, _loSrc, _hiSrc, K8_FF, _loDst, _hiDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundShiftRange<align, false>(_value, _loSrc, _hiSrc, K8_FF, _loDst, _hiDst);
                Flush(_loDst, _hiDst);

                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _loSrc(lo + width - A), _hiSrc(hi + width - A);
                    Storer<false> _loDst(lo + width - A), _hiDst(hi + width - A);
                    BackgroundShiftRange<false, true>(_value, _loSrc, _hiSrc, tailMask, _loDst, _hiDst);
                    Flush(_loDst, _hiDst);
                }

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

        template <bool align, bool first> SIMD_INLINE void BackgroundShiftRangeMasked(const Loader<align> & value,
            const Loader<align> & loSrc, const Loader<align> & hiSrc, const Loader<align> & mask,
            v128_u8 tailMask, Storer<align> & loDst, Storer<align> &  hiDst)
        {
            const v128_u8 _mask = vec_and(Load<align, first>(mask), tailMask);
            BackgroundShiftRange<align, first>(value, loSrc, hiSrc, _mask, loDst, hiDst);
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
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _loSrc(lo), _hiSrc(hi), _mask(mask);
                Storer<align> _loDst(lo), _hiDst(hi);
                BackgroundShiftRangeMasked<align, true>(_value, _loSrc, _hiSrc, _mask, K8_FF, _loDst, _hiDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundShiftRangeMasked<align, false>(_value, _loSrc, _hiSrc, _mask, K8_FF, _loDst, _hiDst);
                Flush(_loDst, _hiDst);

                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _loSrc(lo + width - A), _hiSrc(hi + width - A), _mask(mask + width - A);
                    Storer<false> _loDst(lo + width - A), _hiDst(hi + width - A);
                    BackgroundShiftRangeMasked<false, true>(_value, _loSrc, _hiSrc, _mask, tailMask, _loDst, _hiDst);
                    Flush(_loDst, _hiDst);
                }

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

        template <bool align, bool first> SIMD_INLINE void BackgroundInitMask(const Loader<align> & src, Loader<align> & dstSrc,
            const v128_u8 & index, const v128_u8 & value, Storer<align> & dstDst)
        {
            v128_u8 _mask = (v128_u8)vec_cmpeq(Load<align, first>(src), index);
            v128_u8 _old = Load<align, first>(dstSrc);
            v128_u8 _new = vec_and(_mask, value);
            Store<align, first>(dstDst, vec_sel(_old, _new, _mask));
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
            v128_u8 _index = SetU8(index);
            v128_u8 _value = SetU8(value);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _src(src), _dstSrc(dst);
                Storer<align> _dstDst(dst);
                BackgroundInitMask<align, true>(_src, _dstSrc, _index, _value, _dstDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    BackgroundInitMask<align, false>(_src, _dstSrc, _index, _value, _dstDst);
                Flush(_dstDst);

                if (alignedWidth != width)
                {
                    Loader<false> _src(src + width - A), _dstSrc(dst + width - A);
                    Storer<false> _dstDst(dst + width - A);
                    BackgroundInitMask<false, true>(_src, _dstSrc, _index, _value, _dstDst);
                    Flush(_dstDst);
                }

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
#endif// SIMD_VMX_ENABLE
}
