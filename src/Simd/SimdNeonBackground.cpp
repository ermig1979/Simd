/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
            }

			size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow<align>(value + col, lo + col, hi + col, K8_01);
                if(alignedWidth != width)
                    BackgroundGrowRangeSlow<false>(value + width - A, lo + width - A, hi + width - A, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
             uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
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
    }
#endif// SIMD_NEON_ENABLE
}