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
#ifndef __SimdYuvToBgra_h__
#define __SimdYuvToBgra_h__

#include "Simd/SimdView.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
		SIMD_INLINE void YuvToBgra(int y, int u, int v, int alpha, uchar * bgra)
		{
			bgra[0] = YuvToBlue(y, u);
			bgra[1] = YuvToGreen(y, u, v);
			bgra[2] = YuvToRed(y, v);
			bgra[3] = alpha;
		}

        void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha = 0xFF);

		void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);

		void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
		void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);

		void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);
    }
#endif// SIMD_SSE2_ENABLE

	void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
		const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha = 0xFF);

	void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);

	void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha = 0xFF);

	void Yuv444ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha = 0xFF);

	void Yuv420ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha = 0xFF);
}
#endif//__SimdYuvToBgra_h__