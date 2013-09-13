/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBackground.h"

namespace Simd
{
	namespace Base
	{
		void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(value[col] < lo[col])
						lo[col]--;
					if(value[col] > hi[col])
						hi[col]++;
				}
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(value[col] < lo[col])
						lo[col] = value[col];
					if(value[col] > hi[col])
						hi[col] = value[col];
				}
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
			uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(value[col] < loValue[col] && loCount[col] < 0xFF)
						loCount[col]++;
					if(value[col] > hiValue[col] && hiCount[col] < 0xFF)
						hiCount[col]++;
				}
				value += valueStride;
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
			}
		}

		SIMD_INLINE void AdjustLo(const uchar & count, uchar & value, int threshold)
		{
			if(count > threshold)
			{
				if(value > 0)
					value--;
			}
			else if(count < threshold)
			{
				if(value < 0xFF)
					value++;
			}
		}

		SIMD_INLINE void AdjustHi(const uchar & count, uchar & value, int threshold)
		{
			if(count > threshold)
			{
				if(value < 0xFF)
					value++;
			}
			else if(count < threshold)
			{
				if(value > 0)
					value--;
			}
		}

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					AdjustLo(loCount[col], loValue[col], threshold);
					AdjustHi(hiCount[col], hiValue[col], threshold);
					loCount[col] = 0;
					hiCount[col] = 0;
				}
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
			}
		}

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(mask[col])
					{
						AdjustLo(loCount[col], loValue[col], threshold);
						AdjustHi(hiCount[col], hiValue[col], threshold);
					}
					loCount[col] = 0;
					hiCount[col] = 0;
				}
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
				mask += maskStride;
			}
		}

		SIMD_INLINE void BackgroundShiftRange(const uchar & value, uchar & lo, uchar & hi)
		{
			int add = int(value) - int(hi);
			int sub = int(lo) - int(value);
			if(add > 0)
			{
				lo = Min(lo + add, 0xFF);
				hi = Min(hi + add, 0xFF);
			}
			if(sub > 0)
			{
				lo = Max(lo - sub, 0);
				hi = Max(hi - sub, 0);
			}
		}

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
					BackgroundShiftRange(value[col], lo[col], hi[col]);
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(mask[col])
						BackgroundShiftRange(value[col], lo[col], hi[col]);
				}
				value += valueStride;
				lo += loStride;
				hi += hiStride;
				mask += maskStride;
			}
		}

		void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
			uchar index, uchar value, uchar * dst, size_t dstStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
				{
					if(src[col] == index)
						dst[col] = value;
				}
				src += srcStride;
				dst += dstStride;
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> SIMD_INLINE void BackgroundGrowRangeSlow(const uchar * value, uchar * lo, uchar * hi, __m128i tailMask)
		{
			const __m128i _value = Load<align>((__m128i*)value);
			const __m128i _lo = Load<align>((__m128i*)lo);
			const __m128i _hi = Load<align>((__m128i*)hi);

			const __m128i inc = _mm_and_si128(tailMask, GreaterThenU8(_value, _hi));
			const __m128i dec = _mm_and_si128(tailMask, LesserThenU8(_value, _lo));

			Store<align>((__m128i*)lo, _mm_subs_epu8(_lo, dec));
			Store<align>((__m128i*)hi, _mm_adds_epu8(_hi, inc));
		}

		template <bool align> void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(lo) && Aligned(loStride));
				assert(Aligned(hi) && Aligned(hiStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
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

		void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
				BackgroundGrowRangeSlow<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
			else
				BackgroundGrowRangeSlow<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
		}

		template <bool align> SIMD_INLINE void BackgroundGrowRangeFast(const uchar * value, uchar * lo, uchar * hi)
		{
			const __m128i _value = Load<align>((__m128i*)value);
			const __m128i _lo = Load<align>((__m128i*)lo);
			const __m128i _hi = Load<align>((__m128i*)hi);

			Store<align>((__m128i*)lo, _mm_min_epu8(_lo, _value));
			Store<align>((__m128i*)hi, _mm_max_epu8(_hi, _value));
		}

		template <bool align> void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(lo) && Aligned(loStride));
				assert(Aligned(hi) && Aligned(hiStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundGrowRangeFast<align>(value + col, lo + col, hi + col);
				if(alignedWidth != width)
					BackgroundGrowRangeFast<false>(value + width - A, lo + width - A, hi + width - A);
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
				BackgroundGrowRangeFast<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
			else
				BackgroundGrowRangeFast<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
		}

		template <bool align> SIMD_INLINE void BackgroundIncrementCount(const uchar * value, 
			const uchar * loValue, const uchar * hiValue, uchar * loCount, uchar * hiCount, size_t offset, __m128i tailMask)
		{
			const __m128i _value = Load<align>((__m128i*)(value + offset));
			const __m128i _loValue = Load<align>((__m128i*)(loValue + offset));
			const __m128i _loCount = Load<align>((__m128i*)(loCount + offset));
			const __m128i _hiValue = Load<align>((__m128i*)(hiValue + offset));
			const __m128i _hiCount = Load<align>((__m128i*)(hiCount + offset));

			const __m128i incLo = _mm_and_si128(tailMask, LesserThenU8(_value, _loValue));
			const __m128i incHi = _mm_and_si128(tailMask, GreaterThenU8(_value, _hiValue));

			Store<align>((__m128i*)(loCount + offset), _mm_adds_epu8(_loCount, incLo));
			Store<align>((__m128i*)(hiCount + offset), _mm_adds_epu8(_hiCount, incHi));
		}

		template <bool align> void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
			uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
				assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundIncrementCount<align>(value, loValue, hiValue, loCount, hiCount, col, K8_01);
				if(alignedWidth != width)
					BackgroundIncrementCount<false>(value, loValue, hiValue, loCount, hiCount, width - A, tailMask);
				value += valueStride;
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
			}
		}

		void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
			uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
		{
			if(Aligned(value) && Aligned(valueStride) && 
				Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) && 
				Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride))
				BackgroundIncrementCount<true>(value, valueStride, width, height,
				loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
			else
				BackgroundIncrementCount<false>(value, valueStride, width, height,
				loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
		}

		SIMD_INLINE __m128i AdjustLo(const __m128i &count, const __m128i & value, const __m128i & mask, const __m128i & threshold)
		{
			const __m128i dec = _mm_and_si128(mask, GreaterThenU8(count, threshold));
			const __m128i inc = _mm_and_si128(mask, LesserThenU8(count, threshold));
			return _mm_subs_epu8(_mm_adds_epu8(value, inc), dec);
		}

		SIMD_INLINE __m128i AdjustHi(const __m128i &count, const __m128i & value, const __m128i & mask, const __m128i & threshold)
		{
			const __m128i inc = _mm_and_si128(mask, GreaterThenU8(count, threshold));
			const __m128i dec = _mm_and_si128(mask, LesserThenU8(count, threshold));
			return _mm_subs_epu8(_mm_adds_epu8(value, inc), dec);
		}

		template <bool align> SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, uchar * loValue, 
			uchar * hiCount, uchar * hiValue, size_t offset, const __m128i & threshold, const __m128i & mask)
		{
			const __m128i _loCount = Load<align>((__m128i*)(loCount + offset));
			const __m128i _loValue = Load<align>((__m128i*)(loValue + offset));
			const __m128i _hiCount = Load<align>((__m128i*)(hiCount + offset));
			const __m128i _hiValue = Load<align>((__m128i*)(hiValue + offset));

			Store<align>((__m128i*)(loValue + offset), AdjustLo(_loCount, _loValue, mask, threshold));
			Store<align>((__m128i*)(hiValue + offset), AdjustHi(_hiCount, _hiValue, mask, threshold));
			Store<align>((__m128i*)(loCount + offset), K_ZERO);
			Store<align>((__m128i*)(hiCount + offset), K_ZERO);
		}

		template <bool align> void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
				assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
			}

			const __m128i _threshold = _mm_set1_epi8((char)threshold);
			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, col, _threshold, K8_01);
				if(alignedWidth != width)
					BackgroundAdjustRange<false>(loCount, loValue, hiCount, hiValue, width - A, _threshold, tailMask);
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
			}
		}

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold)
		{
			if(	Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) && 
				Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride))
				BackgroundAdjustRange<true>(loCount, loCountStride, width, height, loValue, loValueStride, 
				hiCount, hiCountStride, hiValue, hiValueStride, threshold);
			else
				BackgroundAdjustRange<false>(loCount, loCountStride, width, height, loValue, loValueStride, 
				hiCount, hiCountStride, hiValue, hiValueStride, threshold);
		}

		template <bool align> SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, uchar * loValue, uchar * hiCount, uchar * hiValue, 
			const uchar * mask, size_t offset, const __m128i & threshold, const __m128i & tailMask)
		{
			const __m128i _mask = Load<align>((const __m128i*)(mask + offset));
			BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, offset, threshold, _mm_and_si128(_mask, tailMask));
		}

		template <bool align> void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
				assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			const __m128i _threshold = _mm_set1_epi8((char)threshold);
			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundAdjustRange<align>(loCount, loValue, hiCount, hiValue, mask, col, _threshold, K8_01);
				if(alignedWidth != width)
					BackgroundAdjustRange<false>(loCount, loValue, hiCount, hiValue, mask, width - A, _threshold, tailMask);
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
				mask += maskStride;
			}		
		}

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
		{
			if(	Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) && 
				Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride) &&
				Aligned(mask) && Aligned(maskStride))
				BackgroundAdjustRange<true>(loCount, loCountStride, width, height, loValue, loValueStride, 
				hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
			else
				BackgroundAdjustRange<false>(loCount, loCountStride, width, height, loValue, loValueStride, 
				hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
		}

		template <bool align> SIMD_INLINE void BackgroundShiftRange(const uchar * value, uchar * lo, uchar * hi, size_t offset, __m128i mask)
		{
			const __m128i _value = Load<align>((__m128i*)(value + offset));
			const __m128i _lo = Load<align>((__m128i*)(lo + offset));
			const __m128i _hi = Load<align>((__m128i*)(hi + offset));

			const __m128i add = _mm_and_si128(mask, _mm_subs_epu8(_value, _hi));
			const __m128i sub = _mm_and_si128(mask, _mm_subs_epu8(_lo, _value));

			Store<align>((__m128i*)(lo + offset), _mm_subs_epu8(_mm_adds_epu8(_lo, add), sub));
			Store<align>((__m128i*)(hi + offset), _mm_subs_epu8(_mm_adds_epu8(_hi, add), sub));
		}

		template <bool align> void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(lo) && Aligned(loStride));
				assert(Aligned(hi) && Aligned(hiStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundShiftRange<align>(value, lo, hi, col, K_INV_ZERO);
				if(alignedWidth != width)
					BackgroundShiftRange<false>(value, lo, hi, width - A, tailMask);
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
				BackgroundShiftRange<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
			else
				BackgroundShiftRange<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
		}

		template <bool align> SIMD_INLINE void BackgroundShiftRange(const uchar * value, uchar * lo, uchar * hi, const uchar * mask, 
			size_t offset, __m128i tailMask)
		{
			const __m128i _mask = Load<align>((const __m128i*)(mask + offset));
			BackgroundShiftRange<align>(value, lo, hi, offset, _mm_and_si128(_mask, tailMask));
		}

		template <bool align> void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(lo) && Aligned(loStride));
				assert(Aligned(hi) && Aligned(hiStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundShiftRange<align>(value, lo, hi, mask, col, K_INV_ZERO);
				if(alignedWidth != width)
					BackgroundShiftRange<false>(value, lo, hi, mask, width - A, tailMask);
				value += valueStride;
				lo += loStride;
				hi += hiStride;
				mask += maskStride;
			}
		}

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && 
				Aligned(hi) && Aligned(hiStride) &&  Aligned(mask) && Aligned(maskStride))
				BackgroundShiftRange<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
			else
				BackgroundShiftRange<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
		}

		template <bool align> SIMD_INLINE void BackgroundInitMask(const uchar * src, uchar * dst, const __m128i & index, const __m128i & value)
		{
			__m128i _mask = _mm_cmpeq_epi8(Load<align>((__m128i*)src), index);
			__m128i _old = _mm_andnot_si128(_mask, Load<align>((__m128i*)dst));
			__m128i _new = _mm_and_si128(_mask, value);
			Store<align>((__m128i*)dst, _mm_or_si128(_old, _new));
		}

		template <bool align> void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
			uchar index, uchar value, uchar * dst, size_t dstStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(src) && Aligned(srcStride));
				assert(Aligned(dst) && Aligned(dstStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i _index = _mm_set1_epi8(index);
			__m128i _value = _mm_set1_epi8(value);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					BackgroundInitMask<align>(src + col, dst + col, _index, _value);
				if(alignedWidth != width)
					BackgroundInitMask<false>(src + width - A, dst + width - A, _index, _value);
				src += srcStride;
				dst += dstStride;
			}
		}

		void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
			uchar index, uchar value, uchar * dst, size_t dstStride)
		{
			if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
				BackgroundInitMask<true>(src, srcStride, width, height, index, value, dst, dstStride);
			else
				BackgroundInitMask<false>(src, srcStride, width, height, index, value, dst, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
	}

	void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
	}

	void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
		const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
		uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundIncrementCount(value, valueStride, width, height,
            loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundIncrementCount(value, valueStride, width, height,
			loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundIncrementCount(value, valueStride, width, height,
			loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
	}

	void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
		uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
		uchar * hiValue, size_t hiValueStride, uchar threshold)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
            hiCount, hiCountStride, hiValue, hiValueStride, threshold);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
			hiCount, hiCountStride, hiValue, hiValueStride, threshold);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
			hiCount, hiCountStride, hiValue, hiValueStride, threshold);
	}

	void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
		uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
		uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
            hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
			hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
			hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
	}

	void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
	}

	void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, 
            mask, maskStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, 
			mask, maskStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, 
			maskStride);
	}

	void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
		uchar index, uchar value, uchar * dst, size_t dstStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
	}
}