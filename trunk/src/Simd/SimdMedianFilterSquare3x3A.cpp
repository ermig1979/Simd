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
#include "Simd/SimdMath.h"
#include "Simd/SimdMedianFilterSquare3x3.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare3x3(const uchar* y[3], size_t offset, __m256i a[9])
        {
            LoadNose3<align, step>(y[0] + offset, a + 0);
            LoadNose3<align, step>(y[1] + offset, a + 3);
            LoadNose3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare3x3(const uchar* y[3], size_t offset, __m256i a[9])
        {
            LoadBody3<align, step>(y[0] + offset, a + 0);
            LoadBody3<align, step>(y[1] + offset, a + 3);
            LoadBody3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare3x3(const uchar* y[3], size_t offset, __m256i a[9])
        {
            LoadTail3<align, step>(y[0] + offset, a + 0);
            LoadTail3<align, step>(y[1] + offset, a + 3);
            LoadTail3<align, step>(y[2] + offset, a + 6);
        }

        SIMD_INLINE void Sort9(__m256i a[9])
        {
            SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]); 
            SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[6], a[7]);
            SortU8(a[1], a[2]); SortU8(a[4], a[5]); SortU8(a[7], a[8]); 
            SortU8(a[0], a[3]); SortU8(a[5], a[8]); SortU8(a[4], a[7]);
            SortU8(a[3], a[6]); SortU8(a[1], a[4]); SortU8(a[2], a[5]); 
            SortU8(a[4], a[7]); SortU8(a[4], a[2]); SortU8(a[6], a[4]);
            SortU8(a[4], a[2]);
        }

        template <bool align, size_t step> void MedianFilterSquare3x3(
            const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
        {
            assert(step*width >= A);

            const uchar * y[3];
            __m256i a[9];

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            for(size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 1);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                if(row < 1)
                    y[0] = y[1];
                if(row >= height - 1)
                    y[2] = y[1];

                LoadNoseSquare3x3<align, step>(y, 0, a);
                Sort9(a);
                Store<align>((__m256i*)(dst), a[4]);

                for(size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare3x3<align, step>(y, col, a);
                    Sort9(a);
                    Store<align>((__m256i*)(dst + col), a[4]);
                }

                size_t col = size - A;
                LoadTailSquare3x3<align, step>(y, col, a);
                Sort9(a);
                Store<align>((__m256i*)(dst + col), a[4]);
            }
        }

        template <bool align> void MedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
            size_t channelCount, uchar * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch(channelCount)
            {
            case 1: MedianFilterSquare3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterSquare3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterSquare3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterSquare3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void MedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
            size_t channelCount, uchar * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MedianFilterSquare3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                MedianFilterSquare3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}