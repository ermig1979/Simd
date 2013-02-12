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
#ifndef __SimdBgrToGray_h__
#define __SimdBgrToGray_h__

#include "Simd/SimdView.h"

namespace Simd
{
    namespace Base
    {
        const int BGR_TO_GRAY_AVERAGING_SHIFT = 14;
        const int BGR_TO_GRAY_ROUND_TERM = 1 << (BGR_TO_GRAY_AVERAGING_SHIFT - 1);
        const int BLUE_TO_GRAY_WEIGHT = int(0.114f*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);
        const int GREEN_TO_GRAY_WEIGHT = int(0.587f*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);
        const int RED_TO_GRAY_WEIGHT = int(0.299f*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);

        SIMD_INLINE int BgrToGray(int blue, int green, int red)
        {
            return (BLUE_TO_GRAY_WEIGHT*blue + GREEN_TO_GRAY_WEIGHT*green + 
                RED_TO_GRAY_WEIGHT*red + BGR_TO_GRAY_ROUND_TERM) >> BGR_TO_GRAY_AVERAGING_SHIFT;
        }

        void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride);
    }
#endif// SIMD_SSE2_ENABLE

    void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride);

	void BgrToGray(const View & bgr, View & gray);
}
#endif//__SimdBgrToGray_h__