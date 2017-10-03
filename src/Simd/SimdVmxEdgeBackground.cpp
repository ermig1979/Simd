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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const Loader<align> & value, const Loader<align> & backgroundSrc, v128_u8 mask, Storer<align> & backgroundDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            const v128_u8 inc = vec_and(mask, vec_cmpgt(_value, _background));
            Store<align, first>(backgroundDst, vec_adds(_background, inc));
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
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _backgroundSrc(background);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundGrowRangeSlow<align, true>(_value, _backgroundSrc, K8_01, _backgroundDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeSlow<align, false>(_value, _backgroundSrc, K8_01, _backgroundDst);
                Flush(_backgroundDst);

                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _backgroundSrc(background + width - A);
                    Storer<false> _backgroundDst(background + width - A);
                    EdgeBackgroundGrowRangeSlow<false, true>(_value, _backgroundSrc, tailMask, _backgroundDst);
                    Flush(_backgroundDst);
                }

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

        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundGrowRangeFast(const Loader<align> & value, const Loader<align> & backgroundSrc, Storer<align> & backgroundDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            Store<align, first>(backgroundDst, vec_max(_background, _value));
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
                Loader<align> _value(value), _backgroundSrc(background);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundGrowRangeFast<align, true>(_value, _backgroundSrc, _backgroundDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeFast<align, false>(_value, _backgroundSrc, _backgroundDst);
                Flush(_backgroundDst);

                if (alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _backgroundSrc(background + width - A);
                    Storer<false> _backgroundDst(background + width - A);
                    EdgeBackgroundGrowRangeFast<false, true>(_value, _backgroundSrc, _backgroundDst);
                    Flush(_backgroundDst);
                }

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

        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundIncrementCount(const Loader<align> & value, const Loader<align> & backgroundValue,
            const Loader<align> & backgroundCountSrc, v128_u8 mask, Storer<align> & backgroundCountDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _backgroundValue = Load<align, first>(backgroundValue);
            const v128_u8 _backgroundCount = Load<align, first>(backgroundCountSrc);

            const v128_u8 inc = vec_and(mask, vec_cmpgt(_value, _backgroundValue));

            Store<align, first>(backgroundCountDst, vec_adds(_backgroundCount, inc));
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
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _backgroundValue(backgroundValue), _backgroundCountSrc(backgroundCount);
                Storer<align> _backgroundCountDst(backgroundCount);
                EdgeBackgroundIncrementCount<align, true>(_value, _backgroundValue, _backgroundCountSrc, K8_01, _backgroundCountDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundIncrementCount<align, false>(_value, _backgroundValue, _backgroundCountSrc, K8_01, _backgroundCountDst);
                Flush(_backgroundCountDst);

                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    Loader<false> _value(value + col), _backgroundValue(backgroundValue + col), _backgroundCountSrc(backgroundCount + col);
                    Storer<false> _backgroundCountDst(backgroundCount + col);
                    EdgeBackgroundIncrementCount<false, true>(_value, _backgroundValue, _backgroundCountSrc, tailMask, _backgroundCountDst);
                    Flush(_backgroundCountDst);
                }

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

        SIMD_INLINE v128_u8 AdjustEdge(const v128_u8 & count, const v128_u8 & value, const v128_u8 & mask, const v128_u8 & threshold)
        {
            const v128_u8 inc = vec_and(mask, vec_cmpgt(count, threshold));
            const v128_u8 dec = vec_and(mask, vec_cmplt(count, threshold));
            return vec_subs(vec_adds(value, inc), dec);
        }

        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundAdjustRange(const Loader<align> & countSrc, const Loader<align> & valueSrc,
            const v128_u8 & threshold, const v128_u8 & mask, Storer<align> & countDst, Storer<align> & valueDst)
        {
            const v128_u8 _count = Load<align, first>(countSrc);
            const v128_u8 _value = Load<align, first>(valueSrc);

            Store<align, first>(valueDst, AdjustEdge(_count, _value, mask, threshold));
            Store<align, first>(countDst, K8_00);
        }

        template <bool align> void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride));
            }

            const v128_u8 _threshold = SetU8(threshold);
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _backgroundCountSrc(backgroundCount), _backgroundValueSrc(backgroundValue);
                Storer<align> _backgroundCountDst(backgroundCount), _backgroundValueDst(backgroundValue);
                EdgeBackgroundAdjustRange<align, true>(_backgroundCountSrc, _backgroundValueSrc,
                    _threshold, K8_01, _backgroundCountDst, _backgroundValueDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRange<align, false>(_backgroundCountSrc, _backgroundValueSrc,
                        _threshold, K8_01, _backgroundCountDst, _backgroundValueDst);
                Flush(_backgroundValueDst, _backgroundCountDst);

                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    Loader<false> _backgroundCountSrc(backgroundCount + col), _backgroundValueSrc(backgroundValue + col);
                    Storer<false> _backgroundCountDst(backgroundCount + col), _backgroundValueDst(backgroundValue + col);
                    EdgeBackgroundAdjustRange<false, true>(_backgroundCountSrc, _backgroundValueSrc,
                        _threshold, tailMask, _backgroundCountDst, _backgroundValueDst);
                    Flush(_backgroundValueDst, _backgroundCountDst);
                }

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

        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundAdjustRangeMasked(const Loader<align> & backgroundCountSrc, const Loader<align> & backgroundValueSrc,
            const Loader<align> & mask, const v128_u8 & threshold, const v128_u8 & tailMask,
            Storer<align> & backgroundCountDst, Storer<align> & backgroundValueDst)
        {
            const v128_u8 _mask = vec_and(Load<align, first>(mask), tailMask);
            EdgeBackgroundAdjustRange<align, first>(backgroundCountSrc, backgroundValueSrc, threshold, _mask, backgroundCountDst, backgroundValueDst);
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

            const v128_u8 _threshold = SetU8(threshold);
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _backgroundCountSrc(backgroundCount), _backgroundValueSrc(backgroundValue), _mask(mask);
                Storer<align> _backgroundCountDst(backgroundCount), _backgroundValueDst(backgroundValue);
                EdgeBackgroundAdjustRangeMasked<align, true>(_backgroundCountSrc, _backgroundValueSrc, _mask,
                    _threshold, K8_01, _backgroundCountDst, _backgroundValueDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundAdjustRangeMasked<align, false>(_backgroundCountSrc, _backgroundValueSrc, _mask,
                        _threshold, K8_01, _backgroundCountDst, _backgroundValueDst);
                Flush(_backgroundValueDst, _backgroundCountDst);

                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    Loader<false> _backgroundCountSrc(backgroundCount + col), _backgroundValueSrc(backgroundValue + col), _mask(mask + col);
                    Storer<false> _backgroundCountDst(backgroundCount + col), _backgroundValueDst(backgroundValue + col);
                    EdgeBackgroundAdjustRangeMasked<false, true>(_backgroundCountSrc, _backgroundValueSrc, _mask,
                        _threshold, tailMask, _backgroundCountDst, _backgroundValueDst);
                    Flush(_backgroundValueDst, _backgroundCountDst);
                }

                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
                mask += maskStride;
            }
        }

        void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            if (Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride) &&
                Aligned(mask) && Aligned(maskStride))
                EdgeBackgroundAdjustRangeMasked<true>(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride,
                    threshold, mask, maskStride);
            else
                EdgeBackgroundAdjustRangeMasked<false>(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride,
                    threshold, mask, maskStride);
        }

        template <bool align, bool first>
        SIMD_INLINE void EdgeBackgroundShiftRangeMasked(const Loader<align> & value, const Loader<align> & backgroundSrc, const Loader<align> & mask, Storer<align> & backgroundDst)
        {
            const v128_u8 _mask = Load<align, first>(mask);
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            Store<align, first>(backgroundDst, vec_sel(_background, _value, _mask));
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
                Loader<align> _value(value), _backgroundSrc(background), _mask(mask);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundShiftRangeMasked<align, true>(_value, _backgroundSrc, _mask, _backgroundDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundShiftRangeMasked<align, false>(_value, _backgroundSrc, _mask, _backgroundDst);
                Flush(_backgroundDst);

                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    Loader<false> _value(value + col), _backgroundSrc(background + col), _mask(mask + col);
                    Storer<false> _backgroundDst(background + col);
                    EdgeBackgroundShiftRangeMasked<false, true>(_value, _backgroundSrc, _mask, _backgroundDst);
                    Flush(_backgroundDst);
                }

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
#endif// SIMD_VMX_ENABLE
}
