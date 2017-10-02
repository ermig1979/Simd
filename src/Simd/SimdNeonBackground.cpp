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
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, uint8x16_t mask)
        {
            const uint8x16_t _value = Load<align>(value);
            const uint8x16_t _lo = Load<align>(lo);
            const uint8x16_t _hi = Load<align>(hi);

            const uint8x16_t inc = vandq_u8(mask, vcgtq_u8(_value, _hi));
            const uint8x16_t dec = vandq_u8(mask, vcltq_u8(_value, _lo));

            Store<align>(lo, vqsubq_u8(_lo, dec));
            Store<align>(hi, vqaddq_u8(_hi, inc));
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
            uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
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
            const uint8x16_t _value = Load<align>(value);
            const uint8x16_t _lo = Load<align>(lo);
            const uint8x16_t _hi = Load<align>(hi);

            Store<align>(lo, vminq_u8(_lo, _value));
            Store<align>(hi, vmaxq_u8(_hi, _value));
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
            const uint8_t * loValue, const uint8_t * hiValue, uint8_t * loCount, uint8_t * hiCount, size_t offset, uint8x16_t mask)
        {
            const uint8x16_t _value = Load<align>(value + offset);
            const uint8x16_t _loValue = Load<align>(loValue + offset);
            const uint8x16_t _loCount = Load<align>(loCount + offset);
            const uint8x16_t _hiValue = Load<align>(hiValue + offset);
            const uint8x16_t _hiCount = Load<align>(hiCount + offset);

            const uint8x16_t incLo = vandq_u8(mask, vcltq_u8(_value, _loValue));
            const uint8x16_t incHi = vandq_u8(mask, vcgtq_u8(_value, _hiValue));

            Store<align>(loCount + offset, vqaddq_u8(_loCount, incLo));
            Store<align>(hiCount + offset, vqaddq_u8(_hiCount, incHi));
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
            uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
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

        SIMD_INLINE uint8x16_t AdjustLo(const uint8x16_t & count, const uint8x16_t & value, const uint8x16_t & mask, const uint8x16_t & threshold)
        {
            const uint8x16_t dec = vandq_u8(mask, vcgtq_u8(count, threshold));
            const uint8x16_t inc = vandq_u8(mask, vcltq_u8(count, threshold));
            return vqsubq_u8(vqaddq_u8(value, inc), dec);
        }

        SIMD_INLINE uint8x16_t AdjustHi(const uint8x16_t & count, const uint8x16_t & value, const uint8x16_t & mask, const uint8x16_t & threshold)
        {
            const uint8x16_t inc = vandq_u8(mask, vcgtq_u8(count, threshold));
            const uint8x16_t dec = vandq_u8(mask, vcltq_u8(count, threshold));
            return vqsubq_u8(vqaddq_u8(value, inc), dec);
        }

        template <bool align> SIMD_INLINE void BackgroundAdjustRange(uint8_t * loCount, uint8_t * loValue,
            uint8_t * hiCount, uint8_t * hiValue, size_t offset, const uint8x16_t & threshold, const uint8x16_t & mask)
        {
            const uint8x16_t _loCount = Load<align>(loCount + offset);
            const uint8x16_t _loValue = Load<align>(loValue + offset);
            const uint8x16_t _hiCount = Load<align>(hiCount + offset);
            const uint8x16_t _hiValue = Load<align>(hiValue + offset);

            Store<align>(loValue + offset, AdjustLo(_loCount, _loValue, mask, threshold));
            Store<align>(hiValue + offset, AdjustHi(_hiCount, _hiValue, mask, threshold));
            Store<align>(loCount + offset, K8_00);
            Store<align>(hiCount + offset, K8_00);
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

            const uint8x16_t _threshold = vld1q_dup_u8(&threshold);
            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
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
            const uint8_t * mask, size_t offset, const uint8x16_t & threshold, const uint8x16_t & tailMask)
        {
            const uint8x16_t _mask = Load<align>(mask + offset);
            BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, offset, threshold, vandq_u8(_mask, tailMask));
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

            const uint8x16_t _threshold = vld1q_dup_u8(&threshold);
            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
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

        template <bool align> SIMD_INLINE void BackgroundShiftRange(const uint8_t * value, uint8_t * lo, uint8_t * hi, size_t offset, uint8x16_t mask)
        {
            const uint8x16_t _value = Load<align>(value + offset);
            const uint8x16_t _lo = Load<align>(lo + offset);
            const uint8x16_t _hi = Load<align>(hi + offset);

            const uint8x16_t add = vandq_u8(mask, vqsubq_u8(_value, _hi));
            const uint8x16_t sub = vandq_u8(mask, vqsubq_u8(_lo, _value));

            Store<align>(lo + offset, vqsubq_u8(vqaddq_u8(_lo, add), sub));
            Store<align>(hi + offset, vqsubq_u8(vqaddq_u8(_hi, add), sub));
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
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRange<align>(value, lo, hi, col, K8_FF);
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
            size_t offset, uint8x16_t tailMask)
        {
            const uint8x16_t _mask = Load<align>(mask + offset);
            BackgroundShiftRange<align>(value, lo, hi, offset, vandq_u8(_mask, tailMask));
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
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    BackgroundShiftRangeMasked<align>(value, lo, hi, mask, col, K8_FF);
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

        template <bool align> SIMD_INLINE void BackgroundInitMask(const uint8_t * src, uint8_t * dst, const uint8x16_t & index, const uint8x16_t & value)
        {
            uint8x16_t _mask = vceqq_u8(Load<align>(src), index);
            uint8x16_t _old = Load<align>(dst);
            Store<align>(dst, vbslq_u8(_mask, value, _old));
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
            uint8x16_t _index = vld1q_dup_u8(&index);
            uint8x16_t _value = vld1q_dup_u8(&value);
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
#endif// SIMD_NEON_ENABLE
}
