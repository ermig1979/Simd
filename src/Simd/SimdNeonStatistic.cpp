/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
             uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;
			uint64x2_t fullSum = K64_0000000000000000;
			uint8x16_t _min = K8_FF;
			uint8x16_t _max = K8_00;
            for(size_t row = 0; row < height; ++row)
            {
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						_min = vminq_u8(_min, _src);
						_max = vmaxq_u8(_max, _src);
						blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
					}
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
				}
                if(width - alignedWidth)
                {
                    const uint8x16_t _src = Load<false>(src + width - A);
					_min = vminq_u8(_min, _src);
					_max = vmaxq_u8(_max, _src);
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8( _src)));
                }
				fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
				src += stride;
            }

            uint8_t min_buffer[A], max_buffer[A];
            Store<false>(min_buffer, _min);
			Store<false>(max_buffer, _max);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum(fullSum) + width*height/2)/(width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
             uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if(Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }
    }
#endif// SIMD_NEON_ENABLE
}