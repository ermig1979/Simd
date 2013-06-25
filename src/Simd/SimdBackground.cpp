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
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> SIMD_INLINE void BackgroundGrowRangeSlow(const uchar * value, uchar * lo, uchar * hi, __m128i incDecMask)
		{
			const __m128i _value = Load<align>((__m128i*)value);
			const __m128i _lo = Load<align>((__m128i*)lo);
			const __m128i _hi = Load<align>((__m128i*)hi);

			const __m128i inc = _mm_and_si128(incDecMask, GreaterThenU8(_value, _hi));
			const __m128i dec = _mm_and_si128(incDecMask, LesserThenU8(_value, _lo));

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
			const uchar * loValue, const uchar * hiValue, uchar * loCount, uchar * hiCount, size_t offset, __m128i incMask)
		{
			const __m128i _value = Load<align>((__m128i*)(value + offset));
			const __m128i _loValue = Load<align>((__m128i*)(loValue + offset));
			const __m128i _loCount = Load<align>((__m128i*)(loCount + offset));
			const __m128i _hiValue = Load<align>((__m128i*)(hiValue + offset));
			const __m128i _hiCount = Load<align>((__m128i*)(hiCount + offset));

			const __m128i incLo = _mm_and_si128(incMask, LesserThenU8(_value, _loValue));
			const __m128i incHi = _mm_and_si128(incMask, GreaterThenU8(_value, _hiValue));

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
	}
#endif// SIMD_SSE2_ENABLE

	void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
	{
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
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::BackgroundIncrementCount(value, valueStride, width, height,
			loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::BackgroundIncrementCount(value, valueStride, width, height,
			loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
	}

	void BackgroundGrowRangeSlow(const View & value, View & lo, View & hi)
	{
		assert(value.width == lo.width && value.height == lo.height && value.width == hi.width && value.height == hi.height);
		assert(value.format == View::Gray8 && lo.format == View::Gray8 && hi.format == View::Gray8);

		BackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
	}

	void BackgroundGrowRangeFast(const View & value, View & lo, View & hi)
	{
		assert(value.width == lo.width && value.height == lo.height && value.width == hi.width && value.height == hi.height);
		assert(value.format == View::Gray8 && lo.format == View::Gray8 && hi.format == View::Gray8);

		BackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
	}

	void BackgroundIncrementCount(const View & value, const View & loValue, const View & hiValue, 
		View & loCount, View & hiCount)
	{
		assert(value.width == loValue.width && value.height == loValue.height && 
			value.width == hiValue.width && value.height == hiValue.height &&
			value.width == loCount.width && value.height == loCount.height && 
			value.width == hiCount.width && value.height == hiCount.height &&);
		assert(value.format == View::Gray8 && loValue.format == View::Gray8 && hiValue.format == View::Gray8 && 
			loCount.format == View::Gray8 && hiCount.format == View::Gray8);

		BackgroundIncrementCount(value.data, value.stride, value.width, value.height,
			loValue.data, loValue.stride, hiValue.data, hiValue.stride,
			loCount.data, loCount.stride, hiCount.data, hiCount.stride);
	}
}