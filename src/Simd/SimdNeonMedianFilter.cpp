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

		template <bool align, size_t step> SIMD_INLINE void LoadNoseRhomb5x5(const uint8_t* y[5], size_t offset, uint8x16_t a[13])
		{
			a[0] = Load<align>(y[0] + offset);
			LoadNose3<align, step>(y[1] + offset, a + 1);
			LoadNose5<align, step>(y[2] + offset, a + 4);
			LoadNose3<align, step>(y[3] + offset, a + 9);
			a[12] = Load<align>(y[4] + offset);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBodyRhomb5x5(const uint8_t* y[5], size_t offset, uint8x16_t a[13])
		{
			a[0] = Load<align>(y[0] + offset);
			LoadBody3<align, step>(y[1] + offset, a + 1);
			LoadBody5<align, step>(y[2] + offset, a + 4);
			LoadBody3<align, step>(y[3] + offset, a + 9);
			a[12] = Load<align>(y[4] + offset);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTailRhomb5x5(const uint8_t* y[5], size_t offset, uint8x16_t a[13])
		{
			a[0] = Load<align>(y[0] + offset);
			LoadTail3<align, step>(y[1] + offset, a + 1);
			LoadTail5<align, step>(y[2] + offset, a + 4);
			LoadTail3<align, step>(y[3] + offset, a + 9);
			a[12] = Load<align>(y[4] + offset);
		}

		SIMD_INLINE void PartialSort13(uint8x16_t a[13])
		{
			SortU8(a[0], a[1]); SortU8(a[3], a[4]); SortU8(a[2], a[4]);
			SortU8(a[2], a[3]); SortU8(a[6], a[7]); SortU8(a[5], a[7]);
			SortU8(a[5], a[6]); SortU8(a[9], a[10]); SortU8(a[8], a[10]);
			SortU8(a[8], a[9]); SortU8(a[11], a[12]); SortU8(a[5], a[8]);
			SortU8(a[2], a[8]); SortU8(a[2], a[5]); SortU8(a[6], a[9]);
			SortU8(a[3], a[9]); SortU8(a[3], a[6]); SortU8(a[7], a[10]);
			SortU8(a[4], a[10]); SortU8(a[4], a[7]); SortU8(a[3], a[12]);
			SortU8(a[0], a[9]);
			a[1] = vminq_u8(a[1], a[10]);
			a[1] = vminq_u8(a[1], a[7]);
			a[1] = vminq_u8(a[1], a[9]);
			a[11] = vmaxq_u8(a[5], a[11]);
			a[11] = vmaxq_u8(a[3], a[11]);
			a[11] = vmaxq_u8(a[2], a[11]);
			SortU8(a[0], a[6]); SortU8(a[1], a[8]); SortU8(a[6], a[8]);
			a[4] = vminq_u8(a[4], a[8]);
			SortU8(a[0], a[1]); SortU8(a[4], a[6]); SortU8(a[0], a[4]);
			a[11] = vmaxq_u8(a[0], a[11]);
			SortU8(a[6], a[11]);
			a[1] = vminq_u8(a[1], a[11]);
			SortU8(a[1], a[4]); SortU8(a[6], a[12]);
			a[6] = vmaxq_u8(a[1], a[6]);
			a[4] = vminq_u8(a[4], a[12]);
			a[6] = vmaxq_u8(a[4], a[6]);
		}

		template <bool align, size_t step> void MedianFilterRhomb5x5(
			const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
		{
			assert(step*(width - 2) >= A);

			const uint8_t * y[5];
			uint8x16_t a[13];

			size_t size = step*width;
			size_t bodySize = Simd::AlignHi(size, A) - A;

			for (size_t row = 0; row < height; ++row, dst += dstStride)
			{
				y[0] = src + srcStride*(row - 2);
				y[1] = y[0] + srcStride;
				y[2] = y[1] + srcStride;
				y[3] = y[2] + srcStride;
				y[4] = y[3] + srcStride;
				if (row < 2)
				{
					if (row < 1)
						y[1] = y[2];
					y[0] = y[1];
				}
				if (row >= height - 2)
				{
					if (row >= height - 1)
						y[3] = y[2];
					y[4] = y[3];
				}

				LoadNoseRhomb5x5<align, step>(y, 0, a);
				PartialSort13(a);
				Store<align>(dst, a[6]);

				for (size_t col = A; col < bodySize; col += A)
				{
					LoadBodyRhomb5x5<align, step>(y, col, a);
					PartialSort13(a);
					Store<align>(dst + col, a[6]);
				}

				size_t col = size - A;
				LoadTailRhomb5x5<false, step>(y, col, a);
				PartialSort13(a);
				Store<false>(dst + col, a[6]);
			}
		}

		template <bool align> void MedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			size_t channelCount, uint8_t * dst, size_t dstStride)
		{
			assert(channelCount > 0 && channelCount <= 4);

			switch (channelCount)
			{
			case 1: MedianFilterRhomb5x5<align, 1>(src, srcStride, width, height, dst, dstStride); break;
			case 2: MedianFilterRhomb5x5<align, 2>(src, srcStride, width, height, dst, dstStride); break;
			case 3: MedianFilterRhomb5x5<align, 3>(src, srcStride, width, height, dst, dstStride); break;
			case 4: MedianFilterRhomb5x5<align, 4>(src, srcStride, width, height, dst, dstStride); break;
			}
		}

		void MedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			size_t channelCount, uint8_t * dst, size_t dstStride)
		{
			if (Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
				MedianFilterRhomb5x5<true>(src, srcStride, width, height, channelCount, dst, dstStride);
			else
				MedianFilterRhomb5x5<false>(src, srcStride, width, height, channelCount, dst, dstStride);
		}
	}
#endif// SIMD_NEON_ENABLE
}