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
#ifndef __SimdShiftBilinear_h__
#define __SimdShiftBilinear_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
	namespace Base
	{
        const int LINEAR_SHIFT = 4;
        const int LINEAR_ROUND_TERM = 1 << (LINEAR_SHIFT - 1);

        const int BILINEAR_SHIFT = LINEAR_SHIFT*2;
        const int BILINEAR_ROUND_TERM = 1 << (BILINEAR_SHIFT - 1);

        const int FRACTION_RANGE = 1 << LINEAR_SHIFT;
        const double FRACTION_ROUND_TERM = 0.5/FRACTION_RANGE;

        void CommonShiftAction(
            const uchar * & src, size_t srcStride, size_t & width, size_t & height, size_t channelCount, 
            const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * & dst, size_t dstStride,
            int & fDx, int & fDy);

		void ShiftBilinear(
			const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
			const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
			size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride);
	}

#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
		void ShiftBilinear(
			const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
			const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
			size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride);
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        void ShiftBilinear(
            const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
            const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride);
    }
#endif// SIMD_AVX2_ENABLE

	void ShiftBilinear(
		const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
		const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
		size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride);
}
#endif//__SimdShiftBilinear_h__