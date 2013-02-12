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
#include "Simd/SimdInterleaveBgra.h"

namespace Simd
{
    namespace Base
    {
        void InterleaveBgra(
            uchar *bgra, size_t size, 
            const int *blue, int bluePrecision, bool blueSigned, 
            const int *green, int greenPrecision, bool greenSigned,
            const int *red, int redPrecision, bool redSigned,
            uchar alpha)
        {
            assert(bluePrecision > 0 && greenPrecision > 0 && redPrecision > 0 && 
				bluePrecision < 32 && greenPrecision < 32 && redPrecision < 32);
            assert((bluePrecision >= 8 && greenPrecision >= 8 && redPrecision >= 8) || 
				(bluePrecision <= 8 && greenPrecision <= 8 && redPrecision <= 8)); 
 
            int blueAdjust = blueSigned ? 1 << (bluePrecision - 1) : 0;
            int greenAdjust = greenSigned ? 1 << (greenPrecision - 1) : 0;
            int redAdjust = redSigned ? 1 << (redPrecision - 1) : 0;
            if(bluePrecision >= 8 && greenPrecision >= 8 && redPrecision >= 8)
            {
                int blueShift = bluePrecision - 8;
                int greenShift = greenPrecision - 8;
                int redShift = redPrecision - 8;
                for(size_t i = 0; i < size; ++i, bgra += 4)
                {
                    bgra[0] = (blue[i] + blueAdjust) >> blueShift;
                    bgra[1] = (green[i] + greenAdjust) >> greenShift;
                    bgra[2] = (red[i] + redAdjust) >> redShift;
                    bgra[3] = alpha;
                }
            }
            else
            {
                int blueShift = 8 - bluePrecision;
                int greenShift = 8 - greenPrecision;
                int redShift = 8 - redPrecision;
                for(size_t i = 0; i < size; ++i, bgra += 4)
                {
                    bgra[0] = (blue[i] + blueAdjust) << blueShift;
                    bgra[1] = (green[i] + greenAdjust) << greenShift;
                    bgra[2] = (red[i] + redAdjust) << redShift;
                    bgra[3] = alpha;
                }
            }
        }

        void InterleaveBgra(uchar *bgra, size_t size, 
            const int *gray, int grayPrecision, bool graySigned, 
            uchar alpha)
        {
            assert(grayPrecision > 0 && grayPrecision < 32);

            int grayAdjust = graySigned ? 1 << (grayPrecision - 1) : 0;
            if(grayPrecision >= 8)
            {
                int grayShift = grayPrecision - 8;
                for(size_t i = 0; i < size; ++i, bgra += 4)
                {
                    int value = (gray[i] + grayAdjust) >> grayShift;
                    bgra[0] = value;
                    bgra[1] = value;
                    bgra[2] = value;
                    bgra[3] = alpha;
                }
            }
            else
            {
                int grayShift = 8 - grayPrecision;
                for(size_t i = 0; i < size; ++i, bgra += 4)
                {
                    int value = (gray[i] + grayAdjust) << grayShift;
                    bgra[0] = value;
                    bgra[1] = value;
                    bgra[2] = value;
                    bgra[3] = alpha;
                }
            }
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
    }
#endif// SIMD_SSE2_ENABLE

	void InterleaveBgra(
		uchar *bgra, size_t size, 
		const int *blue, int bluePrecision, bool blueSigned, 
		const int *green, int greenPrecision, bool greenSigned,
		const int *red, int redPrecision, bool redSigned,
		uchar alpha)
	{
		Base::InterleaveBgra(bgra, size, blue, bluePrecision, blueSigned, 
			green, greenPrecision, greenSigned, red, redPrecision, redSigned, alpha);
	}

	void InterleaveBgra(uchar *bgra, size_t size, 
		const int *gray, int grayPrecision, bool graySigned, 
		uchar alpha)
	{
		Base::InterleaveBgra(bgra, size, gray, grayPrecision, graySigned, alpha);
	}
}