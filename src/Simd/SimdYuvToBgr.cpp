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
#include "Simd/SimdInit.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
	namespace Base
	{
		SIMD_INLINE void Yuv420ToBgr(const uchar *y, int u, int v, uchar * bgr)
		{
			YuvToBgr(y[0], u, v, bgr);
			YuvToBgr(y[1], u, v, bgr + 3);
		}

		void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride)
		{
			assert((width%2 == 0) && (height%2 == 0) && (width >= 2) && (height >= 2));

			for(size_t row = 0; row < height; row += 2)
			{
				for(size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
				{
					int u_ = u[colUV];
					int v_ = v[colUV];
					Yuv420ToBgr(y + colY, u_, v_, bgr + colBgr);
					Yuv420ToBgr(y + yStride + colY, u_, v_, bgr + bgrStride + colBgr);
				}
				y += 2*yStride;
				u += uStride;
				v += vStride;
				bgr += 2*bgrStride;
			}
		}

		void Yuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0, colBgr = 0; col < width; col++, colBgr += 3)
					YuvToBgr(y[col], u[col], v[col], bgr + colBgr);
				y += yStride;
				u += uStride;
				v += vStride;
				bgr += bgrStride;
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
	}
#endif// SIMD_SSE2_ENABLE

	void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride)
	{
		Base::Yuv420ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
	}

	void Yuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride)
	{
		Base::Yuv444ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
	}

	void Yuv444ToBgr(const View & y, const View & u, const View & v, View & bgr)
	{
		assert(y.width == u.width && y.height == u.height && y.format == u.format);
		assert(y.width == v.width && y.height == v.height && y.format == v.format);
		assert(y.width == bgr.width && y.height == bgr.height);
		assert(y.format == View::Gray8 && bgr.format == View::Bgr24);

		Yuv444ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, 
			y.width, y.height, bgr.data, bgr.stride);
	}

	void Yuv420ToBgr(const View & y, const View & u, const View & v, View & bgr)
	{
		assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
		assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
		assert(y.width == bgr.width && y.height == bgr.height);
		assert(y.format == View::Gray8 && bgr.format == View::Bgr24);

		Yuv420ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, 
			y.width, y.height, bgr.data, bgr.stride);
	}
}