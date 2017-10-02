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
            assert(width > A + 2 && height > 2);

            height -= 1;
            width -= 1;
            __m128i _index = _mm_set1_epi8((char)index);
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

        template<bool align> SIMD_INLINE void ChangeIndex(uint8_t * mask, __m128i oldIndex, __m128i newIndex)
        {
            __m128i _mask = Load<align>((__m128i*)mask);
            Store<align>((__m128i*)mask, Combine(_mm_cmpeq_epi8(_mask, oldIndex), newIndex, _mask));
        }

        template<bool align> void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            __m128i _oldIndex = _mm_set1_epi8((char)oldIndex);
            __m128i _newIndex = _mm_set1_epi8((char)newIndex);
            size_t alignedWidth = Simd::AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    ChangeIndex<align>(mask + col, _oldIndex, _newIndex);
                if (alignedWidth != width)
                    ChangeIndex<false>(mask + width - A, _oldIndex, _newIndex);
                mask += stride;
            }
        }

        void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            if (Aligned(mask) && Aligned(stride))
                SegmentationChangeIndex<true>(mask, stride, width, height, oldIndex, newIndex);
            else
                SegmentationChangeIndex<false>(mask, stride, width, height, oldIndex, newIndex);
        }

        SIMD_INLINE void SegmentationPropagate2x2(const __m128i & parentOne, const __m128i & parentAll,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m128i & index, const __m128i & invalid, const __m128i & empty, const __m128i & threshold)
        {
            const __m128i _difference0 = Load<false>((__m128i*)(difference0 + childCol));
            const __m128i _difference1 = Load<false>((__m128i*)(difference1 + childCol));
            const __m128i _child0 = Load<false>((__m128i*)(child0 + childCol));
            const __m128i _child1 = Load<false>((__m128i*)(child1 + childCol));
            const __m128i condition0 = _mm_or_si128(parentAll, _mm_and_si128(parentOne, Greater8u(_difference0, threshold)));
            const __m128i condition1 = _mm_or_si128(parentAll, _mm_and_si128(parentOne, Greater8u(_difference1, threshold)));
            Store<false>((__m128i*)(child0 + childCol), Combine(Lesser8u(_child0, invalid), Combine(condition0, index, empty), _child0));
            Store<false>((__m128i*)(child1 + childCol), Combine(Lesser8u(_child1, invalid), Combine(condition1, index, empty), _child1));
        }

        template<bool align> SIMD_INLINE void SegmentationPropagate2x2(const uint8_t * parent0, const uint8_t * parent1, size_t parentCol,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m128i & index, const __m128i & invalid, const __m128i & empty, const __m128i & threshold)
        {
            const __m128i parent00 = _mm_cmpeq_epi8(Load<align>((__m128i*)(parent0 + parentCol)), index);
            const __m128i parent01 = _mm_cmpeq_epi8(Load<false>((__m128i*)(parent0 + parentCol + 1)), index);
            const __m128i parent10 = _mm_cmpeq_epi8(Load<align>((__m128i*)(parent1 + parentCol)), index);
            const __m128i parent11 = _mm_cmpeq_epi8(Load<false>((__m128i*)(parent1 + parentCol + 1)), index);
            const __m128i parentOne = _mm_or_si128(_mm_or_si128(parent00, parent01), _mm_or_si128(parent10, parent11));
            const __m128i parentAll = _mm_and_si128(_mm_and_si128(parent00, parent01), _mm_and_si128(parent10, parent11));

            SegmentationPropagate2x2(_mm_unpacklo_epi8(parentOne, parentOne), _mm_unpacklo_epi8(parentAll, parentAll),
                difference0, difference1, child0, child1, childCol, index, invalid, empty, threshold);

            SegmentationPropagate2x2(_mm_unpackhi_epi8(parentOne, parentOne), _mm_unpackhi_epi8(parentAll, parentAll),
                difference0, difference1, child0, child1, childCol + A, index, invalid, empty, threshold);
        }

        template<bool align> void SegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height,
            uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            assert(width >= A + 1 && height >= 2);
            height--;
            width--;

            size_t alignedWidth = Simd::AlignLo(width, A);
            __m128i index = _mm_set1_epi8((char)currentIndex);
            __m128i invalid = _mm_set1_epi8((char)invalidIndex);
            __m128i empty = _mm_set1_epi8((char)emptyIndex);
            __m128i threshold = _mm_set1_epi8((char)differenceThreshold);

            for (size_t parentRow = 0, childRow = 1; parentRow < height; ++parentRow, childRow += 2)
            {
                const uint8_t * parent0 = parent + parentRow*parentStride;
                const uint8_t * parent1 = parent0 + parentStride;
                const uint8_t * difference0 = difference + childRow*differenceStride;
                const uint8_t * difference1 = difference0 + differenceStride;
                uint8_t * child0 = child + childRow*childStride;
                uint8_t * child1 = child0 + childStride;

                for (size_t parentCol = 0, childCol = 1; parentCol < alignedWidth; parentCol += A, childCol += DA)
                    SegmentationPropagate2x2<align>(parent0, parent1, parentCol, difference0, difference1,
                        child0, child1, childCol, index, invalid, empty, threshold);
                if (alignedWidth != width)
                    SegmentationPropagate2x2<false>(parent0, parent1, width - A, difference0, difference1,
                        child0, child1, (width - A) * 2 + 1, index, invalid, empty, threshold);
            }
        }

        void SegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height,
            uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            if (Aligned(parent) && Aligned(parentStride))
                SegmentationPropagate2x2<true>(parent, parentStride, width, height, child, childStride,
                    difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
            else
                SegmentationPropagate2x2<false>(parent, parentStride, width, height, child, childStride,
                    difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
        }
    }
#endif//SIMD_SSE2_ENABLE
}
