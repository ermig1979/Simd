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
#include "Simd/SimdConst.h"
#include "Simd/SimdBgrToBgra.h"
#include "Simd/SimdBgraToGray.h"
#include "Simd/SimdBgrToGray.h"

namespace Simd
{
    namespace Base
    {
        void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
        {
			for(size_t row = 0; row < height; ++row)
			{
				const uchar * pBgr = bgr + row*bgrStride;
				uchar * pGray = gray + row*grayStride;
				for(const uchar *pGrayEnd = pGray + width; pGray < pGrayEnd; pGray += 1, pBgr += 3)
				{
					*pGray = BgrToGray(pBgr[0], pBgr[1], pBgr[2]);
				}
			}
        }
    }

#if defined(SIMD_SSE2_ENABLE) || defined(SIMD_AVX2_ENABLE)
    namespace
    {
        struct Buffer
        {
            Buffer(size_t width)
            {
                _p = Allocate(sizeof(uchar)*4*width);
                bgra = (uchar*)_p;
            }

            ~Buffer()
            {
                Free(_p);
            }

            uchar * bgra;
        private:
            void *_p;
        };	
    }
#endif//defined(SIMD_SSE2_ENABLE) || defined(SIMD_AVX2_ENABLE)

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
        {
            assert(width >= A);

            Buffer buffer(width);

			for(size_t row = 1; row < height; ++row)
			{
				Base::BgrToBgra(bgr, width, buffer.bgra, false, false);
				Sse2::BgraToGray(buffer.bgra, width, 1, 4*width, gray, width);
				bgr += bgrStride;
				gray += grayStride;
			}
			Base::BgrToBgra(bgr, width, buffer.bgra, false, true);
			Sse2::BgraToGray(buffer.bgra, width, 1, 4*width, gray, width);
        }
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
        {
            assert(width >= A);

            Buffer buffer(width);

            for(size_t row = 1; row < height; ++row)
            {
                Base::BgrToBgra(bgr, width, buffer.bgra, false, false);
                Avx2::BgraToGray(buffer.bgra, width, 1, 4*width, gray, width);
                bgr += bgrStride;
                gray += grayStride;
            }
            Base::BgrToBgra(bgr, width, buffer.bgra, false, true);
            Avx2::BgraToGray(buffer.bgra, width, 1, 4*width, gray, width);
        }
    }
#endif//SIMD_Avx2_ENABLE

    void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
        else
#endif//SIMD_SSE2_ENABLE       
            Base::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    }

	void BgrToGray(const View & bgr, View & gray)
	{
		assert(bgr.width == gray.width && bgr.height == gray.height);
		assert(bgr.format == View::Bgr24 && gray.format == View::Gray8);

		BgrToGray(bgr.data, bgr.width, bgr.height, bgr.stride, gray.data, gray.stride);
	}
}