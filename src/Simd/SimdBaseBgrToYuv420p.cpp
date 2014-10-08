/*
* Simd Library.
*
* Copyright (c) 2014-2014 Antonenka Mikhail.
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
#include "Simd/SimdConversion.h"
#include "Simd/SimdBase.h"

namespace Simd
{
	namespace Base
	{
		SIMD_INLINE void BgrRowToYRow(const uint8_t * bgr, uint8_t * y)
		{
			y[0] = BgrToYLuminance(bgr[0], bgr[1], bgr[2]);
			y[1] = BgrToYLuminance(bgr[3], bgr[4], bgr[5]);
		}

		SIMD_INLINE void BgrUnitToYuv420p(const uint8_t * bgr, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, uint8_t * v)
		{
			//Bgr Unit to Y unit (unit consists of 2 rows by 2 pixels: 2x2 = 4pixels)
			BgrRowToYRow(bgr, y);
			BgrRowToYRow(bgr + bgrStride, y + yStride);

			//filling U and V planes based on left upper pixel Rgb, rather then averaging all pixels in current unit
			u[0] = BgrToUChrominance(bgr[0], bgr[1], bgr[2]);
			v[0] = BgrToVChrominance(bgr[0], bgr[1], bgr[2]);
		}

		void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
			uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
		{
			assert((width%2 == 0) && (height%2 == 0) && (width >= 2) && (height >= 2));

			for(size_t row = 0; row < height; row += 2)
			{
				for(size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
				{
					BgrUnitToYuv420p(bgr + colBgr, bgrStride, y + colY, yStride, u + colUV, v + colUV);
				}
				y += 2*yStride;
				u += uStride;
				v += vStride;
				bgr += 2*bgrStride;
			}
		}
	}
}