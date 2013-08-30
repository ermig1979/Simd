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
#ifndef __SimdStatistic_h__
#define __SimdStatistic_h__

#include "Simd/SimdView.h"

namespace Simd
{
	namespace Base
	{
		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average);

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

        void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);
    }

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average);

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

        void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
            uchar * min, uchar * max, uchar * average);

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

        void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);
    }
#endif// SIMD_AVX2_ENABLE

	void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
		uchar * min, uchar * max, uchar * average);

    void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
        uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

    void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

    void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

	void GetStatistic(const View & src, uchar * min, uchar * max, uchar * average);

    void GetMoments(const View & mask, uchar index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

    void GetRowSums(const View & src, uint * sums);

    void GetColSums(const View & src, uint * sums);
}
#endif//__SimdStatistic_h__
