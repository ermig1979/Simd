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
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> void BackgroundGrowRangeSlow(const uchar * value, uchar * lo, uchar * hi, __m128i incDecMask)
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

		template <bool align> void BackgroundGrowRangeFast(const uchar * value, uchar * lo, uchar * hi)
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
}