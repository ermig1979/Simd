/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        template<bool align, bool masked> SIMD_INLINE void ChangeIndex(uint8_t * mask, __m512i oldIndex, __m512i newIndex, __mmask64 tail = -1)
        {
			Store<align, true>(mask, newIndex, _mm512_cmpeq_epi8_mask((Load<align, masked>(mask, tail)), oldIndex)&tail);
        }

		template<bool align> SIMD_INLINE void ChangeIndex4(uint8_t * mask, __m512i oldIndex, __m512i newIndex)
		{
			Store<align, true>(mask + 0 * A, newIndex, _mm512_cmpeq_epi8_mask(Load<align>(mask + 0 * A), oldIndex));
			Store<align, true>(mask + 1 * A, newIndex, _mm512_cmpeq_epi8_mask(Load<align>(mask + 1 * A), oldIndex));
			Store<align, true>(mask + 2 * A, newIndex, _mm512_cmpeq_epi8_mask(Load<align>(mask + 2 * A), oldIndex));
			Store<align, true>(mask + 3 * A, newIndex, _mm512_cmpeq_epi8_mask(Load<align>(mask + 3 * A), oldIndex));
		}

        template<bool align> void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
			if (align)
				assert(Aligned(mask) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			size_t fullAlignedWidth = Simd::AlignLo(width, QA);
			__mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _oldIndex = _mm512_set1_epi8((char)oldIndex);
            __m512i _newIndex = _mm512_set1_epi8((char)newIndex);

            for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
				for (; col < fullAlignedWidth; col += QA)
					ChangeIndex4<align>(mask + col, _oldIndex, _newIndex);
				for (; col < alignedWidth; col += A)
					ChangeIndex<align, false>(mask + col, _oldIndex, _newIndex);
				if (col < width)
					ChangeIndex<align, true>(mask + col, _oldIndex, _newIndex, tailMask);
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

		template<bool align, bool masked> SIMD_INLINE void FillSingleHoles(uint8_t * mask, ptrdiff_t stride, __m512i index, __mmask64 edge = -1)
		{
			__mmask64 up = _mm512_cmpeq_epi8_mask((Load<align, masked>(mask - stride, edge)), index);
			__mmask64 left = _mm512_cmpeq_epi8_mask((Load<false, masked>(mask - 1, edge)), index);
			__mmask64 right = _mm512_cmpeq_epi8_mask((Load<false, masked>(mask + 1, edge)), index);
			__mmask64 down = _mm512_cmpeq_epi8_mask((Load<align, masked>(mask + stride, edge)), index);
			Store<align, true>(mask, index, up & left & right & down & edge);
		}

		template<bool align> void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
		{
			assert(width > 2 && height > 2);

			__m512i _index = _mm512_set1_epi8((char)index);
			size_t alignedWidth = Simd::AlignLo(width - 1, A);
			__mmask64 noseMask = NoseMask64(A - 1);
			__mmask64 tailMask = TailMask64(width - 1 - alignedWidth);
			if (alignedWidth < A)
				noseMask = noseMask&tailMask;

			for (size_t row = 2; row < height; ++row)
			{
				mask += stride;
				size_t col = A;
				FillSingleHoles<align, true>(mask, stride, _index, noseMask);
				for (; col < alignedWidth; col += A)
					FillSingleHoles<align, false>(mask + col, stride, _index);
				if (col < width)
					FillSingleHoles<align, true>(mask + col, stride, _index, tailMask);
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
#endif//SIMD_AVX512BW_ENABLE
}
