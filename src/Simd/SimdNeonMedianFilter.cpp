/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
	namespace Neon
	{
        template <bool align, size_t step> SIMD_INLINE void LoadNoseRhomb3x3(const uint8_t* y[3], size_t offset, uint8x16_t a[5])
        {
            a[0] = Load<align>(y[0] + offset);
            LoadNose3<align, step>(y[1] + offset, a + 1);
			a[4] = Load<align>(y[2] + offset);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodyRhomb3x3(const uint8_t* y[3], size_t offset, uint8x16_t a[5])
        {
            a[0] = Load<align>(y[0] + offset);
            LoadBody3<align, step>(y[1] + offset, a + 1);
            a[4] = Load<align>(y[2] + offset);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailRhomb3x3(const uint8_t* y[3], size_t offset, uint8x16_t a[5])
        {
            a[0] = Load<align>(y[0] + offset);
            LoadTail3<align, step>(y[1] + offset, a + 1);
            a[4] = Load<align>(y[2] + offset);
        }

        SIMD_INLINE void PartialSort5(uint8x16_t a[5])
        {
            SortU8(a[2], a[3]); 
            SortU8(a[1], a[2]);
            SortU8(a[2], a[3]); 
            a[4] = vmaxq_u8(a[1], a[4]); 
            a[0] = vminq_u8(a[0], a[3]);
            SortU8(a[2], a[0]); 
            a[2] = vmaxq_u8(a[4], a[2]);
            a[2] = vminq_u8(a[2], a[0]);
        }

        template <bool align, size_t step> void MedianFilterRhomb3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);

            const uint8_t * y[3];
			uint8x16_t a[5];

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

                LoadNoseRhomb3x3<align, step>(y, 0, a);
                PartialSort5(a);
                Store<align>(dst, a[2]);

                for(size_t col = A; col < bodySize; col += A)
                {
                    LoadBodyRhomb3x3<align, step>(y, col, a);
                    PartialSort5(a);
                    Store<align>(dst + col, a[2]);
                }

                size_t col = size - A;
                LoadTailRhomb3x3<align, step>(y, col, a);
                PartialSort5(a);
                Store<align>(dst + col, a[2]);
            }
        }

        template <bool align> void MedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch(channelCount)
            {
            case 1: MedianFilterRhomb3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MedianFilterRhomb3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MedianFilterRhomb3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MedianFilterRhomb3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void MedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MedianFilterRhomb3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                MedianFilterRhomb3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
	}
#endif// SIMD_NEON_ENABLE
}