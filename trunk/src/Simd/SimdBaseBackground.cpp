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
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"

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
}