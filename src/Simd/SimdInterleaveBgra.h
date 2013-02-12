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
#ifndef __SimdInterleaveBgra_h__
#define __SimdInterleaveBgra_h__

#include "Simd/SimdTypes.h"
#include "Simd/SimdConst.h"

namespace Simd
{
    namespace Base
    {
        void InterleaveBgra(
            uchar *bgra, size_t size, 
            const int *blue, int bluePrecision, bool blueSigned, 
            const int *green, int greenPrecision, bool greenSigned,
            const int *red, int redPrecision, bool redSigned,
            uchar alpha = 0xFF);

        void InterleaveBgra(uchar *bgra, size_t size, 
            const int *gray, int grayPrecision, bool graySigned, 
            uchar alpha = 0xFF);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
		SIMD_INLINE __m128i InterleaveBgra32(__m128i blue32, __m128i green32, __m128i red32)
		{
			return _mm_packus_epi16(_mm_packs_epi32(blue32, green32), _mm_packs_epi32(red32, K32_000000FF));
		}
    }
#endif// SIMD_SSE2_ENABLE

	void InterleaveBgra(
		uchar *bgra, size_t size, 
		const int *blue, int bluePrecision, bool blueSigned, 
		const int *green, int greenPrecision, bool greenSigned,
		const int *red, int redPrecision, bool redSigned,
		uchar alpha = 0xFF);

	void InterleaveBgra(uchar *bgra, size_t size, 
		const int *gray, int grayPrecision, bool graySigned, 
		uchar alpha = 0xFF);
}
#endif//__SimdInterleaveBgra_h__
