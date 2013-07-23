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
#ifndef __SimdBackground_h__
#define __SimdBackground_h__

#include "Simd/SimdView.h"

namespace Simd
{
	namespace Base
	{
		void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
			uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride);

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold);

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride);

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride);

		void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
			uchar index, uchar value, uchar * dst, size_t dstStride);
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
			uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride);

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold);

		void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
			uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
			uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride);

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

		void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride);

		void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
			uchar index, uchar value, uchar * dst, size_t dstStride);
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

        void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

        void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
            const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
            uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride);

        void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
            uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
            uchar * hiValue, size_t hiValueStride, uchar threshold);

        void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
            uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
            uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride);

        void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

        void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride);

        void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar index, uchar value, uchar * dst, size_t dstStride);
    }
#endif// SIMD_AVX2_ENABLE

	void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

	void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

	void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
		const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
		uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride);

	void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
		uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
		uchar * hiValue, size_t hiValueStride, uchar threshold);

	void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
		uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
		uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride);

	void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride);

	void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
		uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride);

	void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
		uchar index, uchar value, uchar * dst, size_t dstStride);

	void BackgroundGrowRangeSlow(const View & value, View & lo, View & hi);

	void BackgroundGrowRangeFast(const View & value, View & lo, View & hi);

	void BackgroundIncrementCount(const View & value, const View & loValue, const View & hiValue, 
		View & loCount, View & hiCount);

	void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, 
		uchar threshold);

	void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, 
		uchar threshold, const View & mask);

	void BackgroundShiftRange(const View & value, View & lo, View & hi);

	void BackgroundShiftRange(const View & value, View & lo, View & hi, const View & mask);

	void BackgroundInitMask(const View & src, uchar index, uchar value, View & dst);
}
#endif//__SimdBackground_h__
