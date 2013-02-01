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
#include "Simd/SimdYuvToBgra.h"

namespace Simd
{
    namespace Base
    {
		SIMD_INLINE void Yuv420ToBgra(const uchar *y, int u, int v, int alpha, uchar * bgra)
		{
			YuvToBgra(y[0], u, v, alpha, bgra);
			YuvToBgra(y[1], u, v, alpha, bgra + 4);
		}

        void RowYuv444ToBgra(uchar *bgra, size_t width, const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            const int *end = y + width;
            for(;y < end; y += 1, u += 1, v += 1, bgra += 4)
            {
                int y0 = y[0] << shift;
                int u0 = u[0] << shift;
                int v0 = v[0] << shift;
                bgra[0] = YuvToBlue(y0, u0);
                bgra[1] = YuvToGreen(y0, u0, v0);
                bgra[2] = YuvToRed(y0, v0);
                bgra[3] = alpha;
            }
        }

        void Yuv444ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            for(size_t row  = 0; row < height; ++row)
            {
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += width;
                v += width;
            }
        }

        void Yuv422ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            assert(height%2 == 0);

            size_t uv_height = height/2;
            for(size_t row  = 0; row < uv_height; ++row)
            {
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += width;
                v += width;
            }
        }

        void RowYuv420ToBgra(uchar *bgra, size_t width, const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            const int *end = y + width;
            for(;y < end; y += 2, u += 1, v += 1, bgra += 8)
            {
                int y0 = y[0] << shift;
                int u0 = u[0] << shift;
                int v0 = v[0] << shift;
                bgra[0] = YuvToBlue(y0, u0);
                bgra[1] = YuvToGreen(y0, u0, v0);
                bgra[2] = YuvToRed(y0, v0);
                bgra[3] = alpha;
                int y1 = y[1] << shift;
                bgra[4] = YuvToBlue(y1, u0);
                bgra[5] = YuvToGreen(y1, u0, v0);
                bgra[6] = YuvToRed(y1, v0);
                bgra[7] = alpha;
            }
        }

        void Yuv420ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            assert(width%2 == 0 && height%2 == 0);

            size_t uv_width = width/2; 
            size_t uv_height = height/2;
            for(size_t row = 0; row < uv_height; ++row)
            {
                RowYuv420ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                RowYuv420ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += uv_width;
                v += uv_width;
            }
        }

        void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha)
        {
            assert(precision >= 8 && (dx == 1 || dx == 2) && (dy == 1 || dy == 2) && (dy != 1 || dx != 2));

            if(dy == 2)
            {
                if(dx == 2)
                    Yuv420ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
                else
                    Yuv422ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
            }
            else
                Yuv444ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
        }

		void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			assert((width%2 == 0) && (height%2 == 0) && (width >= 2) && (height >= 2));

			for(size_t row = 0; row < height; row += 2)
			{
				for(size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
				{
					int u_ = u[colUV];
					int v_ = v[colUV];
					Yuv420ToBgra(y + colY, u_, v_, alpha, bgra + colBgra);
					Yuv420ToBgra(y + yStride + colY, u_, v_, alpha, bgra + bgraStride + colBgra);
				}
				y += 2*yStride;
				u += uStride;
				v += vStride;
				bgra += 2*bgraStride;
			}
		}

		void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0, colBgra = 0; col < width; col++, colBgra += 4)
					YuvToBgra(y[col], u[col], v[col], alpha, bgra + colBgra);
				y += yStride;
				u += uStride;
				v += vStride;
				bgra += bgraStride;
			}
		}
   }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
    }
#endif// SIMD_SSE2_ENABLE

    IntYuvToBgraPtr IntYuvToBgra = Base::YuvToBgra;
        //SIMD_INIT_FUNCTION_PTR(IntYuvToBgraPtr, Sse2::YuvToBgra, Base::YuvToBgra);

}