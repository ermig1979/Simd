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
#ifndef __SimdReduceGray5x5_h__
#define __SimdReduceGray5x5_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
	namespace Base
	{
		void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true);
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true);
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true);
    }
#endif// SIMD_AVX2_ENABLE

	void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
		uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true);
}
#endif//__SimdReduceGray5x5_h__