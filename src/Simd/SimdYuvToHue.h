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
#ifndef __SimdYuvToHue_h__
#define __SimdYuvToHue_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
	namespace Base
	{
        const float KF_255_DIV_6 = 255.0f/6.0f;

		void Yuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * hue, size_t hueStride);

		void Yuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * hue, size_t hueStride);
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		void Yuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * hue, size_t hueStride);

		void Yuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * hue, size_t hueStride);
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void Yuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
            size_t width, size_t height, uchar * hue, size_t hueStride);

        void Yuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
            size_t width, size_t height, uchar * hue, size_t hueStride);
    }
#endif// SIMD_AVX2_ENABLE

	void Yuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * hue, size_t hueStride);

	void Yuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * hue, size_t hueStride);
}
#endif//__SimdYuvToHue_h__