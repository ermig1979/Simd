/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdSse2.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        template<bool align> SIMD_INLINE void FillSingleHoles(uint8_t * mask, ptrdiff_t stride, __m128i index)
        {
            const __m128i up = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask - stride)), index);
            const __m128i left = _mm_cmpeq_epi8(Load<false>((__m128i*)(mask - 1)), index);
            const __m128i right = _mm_cmpeq_epi8(Load<false>((__m128i*)(mask + 1)), index);
            const __m128i down = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask + stride)), index);
            StoreMasked<align>((__m128i*)mask, index, _mm_and_si128(_mm_and_si128(up, left), _mm_and_si128(right, down)));
        }

        template<bool align> void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > 2 && height > 2);

            height -= 1;
            width -= 1;
            __m128i _index = _mm_set1_epi8((char)index);
            size_t alignedWidth = Simd::AlignLo(width, A);
            for(size_t row = 1; row < height; ++row)
            {
                mask += stride;

                FillSingleHoles<false>(mask + 1, stride, _index);

                for(size_t col = A; col < alignedWidth; col += A)
                    FillSingleHoles<align>(mask + col, stride, _index);

                if(alignedWidth != width )
                    FillSingleHoles<false>(mask + width - A, stride, _index);
            }
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            if(Aligned(mask) && Aligned(stride))
                SegmentationFillSingleHoles<true>(mask, stride, width, height, index);
            else
                SegmentationFillSingleHoles<false>(mask, stride, width, height, index);
        }
    }
#endif//SIMD_SSE2_ENABLE
}