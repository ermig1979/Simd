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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template<bool align> SIMD_INLINE void ChangeIndex(uint8_t * mask, const uint8x16_t & oldIndex, const uint8x16_t & newIndex)
        {
			uint8x16_t _mask = Load<align>(mask);
            Store<align>(mask, vbslq_u8(vceqq_u8(_mask, oldIndex), newIndex, _mask));
        }

        template<bool align> void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
			if (align)
				assert(Aligned(mask) && Aligned(stride));

			uint8x16_t _oldIndex = vdupq_n_u8(oldIndex);
			uint8x16_t _newIndex = vdupq_n_u8(newIndex);
            size_t alignedWidth = Simd::AlignLo(width, A);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                    ChangeIndex<align>(mask + col, _oldIndex, _newIndex);
                if(alignedWidth != width)
                    ChangeIndex<false>(mask + width - A, _oldIndex, _newIndex);
                mask += stride;
            }
        }

        void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            if(Aligned(mask) && Aligned(stride))
                SegmentationChangeIndex<true>(mask, stride, width, height, oldIndex, newIndex);
            else
                SegmentationChangeIndex<false>(mask, stride, width, height, oldIndex, newIndex);
        }

		template<bool align> SIMD_INLINE void FillSingleHoles(uint8_t * mask, ptrdiff_t stride, const uint8x16_t & index)
		{
			uint8x16_t up = vceqq_u8(Load<align>(mask - stride), index);
			uint8x16_t left = vceqq_u8(Load<false>(mask - 1), index);
			uint8x16_t right = vceqq_u8(Load<false>(mask + 1), index);
			uint8x16_t down = vceqq_u8(Load<align>(mask + stride), index);
			Store<align>(mask, vbslq_u8(vandq_u8(vandq_u8(up, left), vandq_u8(right, down)), index, Load<align>(mask)));
		}

		template<bool align> void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
		{
			assert(width > A + 2 && height > 2);
			if (align)
				assert(Aligned(mask) && Aligned(stride));

			height -= 1;
			width -= 1;
			uint8x16_t _index = vdupq_n_u8(index);
			size_t alignedWidth = Simd::AlignLo(width, A);
			for (size_t row = 1; row < height; ++row)
			{
				mask += stride;

				FillSingleHoles<false>(mask + 1, stride, _index);

				for (size_t col = A; col < alignedWidth; col += A)
					FillSingleHoles<align>(mask + col, stride, _index);

				if (alignedWidth != width)
					FillSingleHoles<false>(mask + width - A, stride, _index);
			}
		}

		void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
		{
			if (Aligned(mask) && Aligned(stride))
				SegmentationFillSingleHoles<true>(mask, stride, width, height, index);
			else
				SegmentationFillSingleHoles<false>(mask, stride, width, height, index);
		}
    }
#endif//SIMD_NEON_ENABLE
}