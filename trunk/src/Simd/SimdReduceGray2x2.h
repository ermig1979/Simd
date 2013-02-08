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
#ifndef __SimdReduceGray2x2_h__
#define __SimdReduceGray2x2_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int Average(int a, int b)
        {
            return (a + b + 1) >> 1;
        }

        SIMD_INLINE int Average(int a, int b, int c, int d)
        {
            return (a + b + c + d + 2) >> 2;
        }

        void ReduceGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        void ReduceGray2x2A(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        void ReduceGray2x2U(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

		void ReduceGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);
}
#endif// SIMD_SSE2_ENABLE

    typedef void (*ReduceGray2x2Ptr)(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);
    extern ReduceGray2x2Ptr ReduceGray2x2A;

    void ReduceGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);
}
#endif//__SimdReduceGray2x2_h__