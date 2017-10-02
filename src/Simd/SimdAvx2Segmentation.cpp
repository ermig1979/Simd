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
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template<bool align> SIMD_INLINE void FillSingleHoles(uint8_t * mask, ptrdiff_t stride, __m256i index)
        {
            const __m256i up = _mm256_cmpeq_epi8(Load<align>((__m256i*)(mask - stride)), index);
            const __m256i left = _mm256_cmpeq_epi8(Load<false>((__m256i*)(mask - 1)), index);
            const __m256i right = _mm256_cmpeq_epi8(Load<false>((__m256i*)(mask + 1)), index);
            const __m256i down = _mm256_cmpeq_epi8(Load<align>((__m256i*)(mask + stride)), index);
            StoreMasked<align>((__m256i*)mask, index, _mm256_and_si256(_mm256_and_si256(up, left), _mm256_and_si256(right, down)));
        }

        template<bool align> void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > A + 2 && height > 2);

            height -= 1;
            width -= 1;
            __m256i _index = _mm256_set1_epi8((char)index);
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

        template<bool align> SIMD_INLINE void ChangeIndex(uint8_t * mask, __m256i oldIndex, __m256i newIndex)
        {
            __m256i _mask = Load<align>((__m256i*)mask);
            Store<align>((__m256i*)mask, _mm256_blendv_epi8(_mask, newIndex, _mm256_cmpeq_epi8(_mask, oldIndex)));
        }

        template<bool align> void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            __m256i _oldIndex = _mm256_set1_epi8((char)oldIndex);
            __m256i _newIndex = _mm256_set1_epi8((char)newIndex);
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

        SIMD_INLINE void SegmentationPropagate2x2(const __m256i & parentOne, const __m256i & parentAll,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m256i & index, const __m256i & invalid, const __m256i & empty, const __m256i & threshold)
        {
            const __m256i _difference0 = Load<false>((__m256i*)(difference0 + childCol));
            const __m256i _difference1 = Load<false>((__m256i*)(difference1 + childCol));
            const __m256i _child0 = Load<false>((__m256i*)(child0 + childCol));
            const __m256i _child1 = Load<false>((__m256i*)(child1 + childCol));
            const __m256i condition0 = _mm256_or_si256(parentAll, _mm256_and_si256(parentOne, Greater8u(_difference0, threshold)));
            const __m256i condition1 = _mm256_or_si256(parentAll, _mm256_and_si256(parentOne, Greater8u(_difference1, threshold)));
            Store<false>((__m256i*)(child0 + childCol), _mm256_blendv_epi8(_child0, _mm256_blendv_epi8(empty, index, condition0), Lesser8u(_child0, invalid)));
            Store<false>((__m256i*)(child1 + childCol), _mm256_blendv_epi8(_child1, _mm256_blendv_epi8(empty, index, condition1), Lesser8u(_child1, invalid)));
        }

        template<bool align> SIMD_INLINE void SegmentationPropagate2x2(const uint8_t * parent0, const uint8_t * parent1, size_t parentCol,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m256i & index, const __m256i & invalid, const __m256i & empty, const __m256i & threshold)
        {
            const __m256i parent00 = _mm256_cmpeq_epi8(Load<align>((__m256i*)(parent0 + parentCol)), index);
            const __m256i parent01 = _mm256_cmpeq_epi8(Load<false>((__m256i*)(parent0 + parentCol + 1)), index);
            const __m256i parent10 = _mm256_cmpeq_epi8(Load<align>((__m256i*)(parent1 + parentCol)), index);
            const __m256i parent11 = _mm256_cmpeq_epi8(Load<false>((__m256i*)(parent1 + parentCol + 1)), index);
            const __m256i parentOne = _mm256_permute4x64_epi64(_mm256_or_si256(_mm256_or_si256(parent00, parent01), _mm256_or_si256(parent10, parent11)), 0xD8);
            const __m256i parentAll = _mm256_permute4x64_epi64(_mm256_and_si256(_mm256_and_si256(parent00, parent01), _mm256_and_si256(parent10, parent11)), 0xD8);

            SegmentationPropagate2x2(_mm256_unpacklo_epi8(parentOne, parentOne), _mm256_unpacklo_epi8(parentAll, parentAll),
                difference0, difference1, child0, child1, childCol, index, invalid, empty, threshold);

            SegmentationPropagate2x2(_mm256_unpackhi_epi8(parentOne, parentOne), _mm256_unpackhi_epi8(parentAll, parentAll),
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
            __m256i index = _mm256_set1_epi8((char)currentIndex);
            __m256i invalid = _mm256_set1_epi8((char)invalidIndex);
            __m256i empty = _mm256_set1_epi8((char)emptyIndex);
            __m256i threshold = _mm256_set1_epi8((char)differenceThreshold);

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

        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, __m256i index)
        {
            for (size_t col = 0; col < alignedSize; col += A)
            {
                if (!_mm256_testz_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)(mask + col)), index), K_INV_ZERO))
                    return true;
            }
            if (alignedSize != fullSize)
            {
                if (!_mm256_testz_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)(mask + fullSize - A)), index), K_INV_ZERO))
                    return true;
            }
            return false;
        }

        SIMD_INLINE bool ColsHasIndex(const uint8_t * mask, size_t stride, size_t size, __m256i index, uint8_t * cols)
        {
            __m256i _cols = _mm256_setzero_si256();
            for (size_t row = 0; row < size; ++row)
            {
                _cols = _mm256_or_si256(_cols, _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)mask), index));
                mask += stride;
            }
            _mm256_storeu_si256((__m256i*)cols, _cols);
            return !_mm256_testz_si256(_cols, K_INV_ZERO);
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*right - *left >= A && *bottom > *top);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            size_t fullWidth = *right - *left;
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            __m256i _index = _mm256_set1_epi8(index);
            bool search = true;
            for (ptrdiff_t row = *top; search && row < *bottom; ++row)
            {
                if (RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index))
                {
                    search = false;
                    *top = row;
                }
            }

            if (search)
            {
                *left = 0;
                *top = 0;
                *right = 0;
                *bottom = 0;
                return;
            }

            search = true;
            for (ptrdiff_t row = *bottom - 1; search && row >= *top; --row)
            {
                if (RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index))
                {
                    search = false;
                    *bottom = row + 1;
                }
            }

            search = true;
            for (ptrdiff_t col = *left; search && col < *left + alignedWidth; col += A)
            {
                uint8_t cols[A];
                if (ColsHasIndex(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols))
                {
                    for (size_t i = 0; i < A; i++)
                    {
                        if (cols[i])
                        {
                            *left = col + i;
                            break;
                        }
                    }
                    search = false;
                    break;
                }
            }

            search = true;
            for (ptrdiff_t col = *right; search && col > *left; col -= A)
            {
                uint8_t cols[A];
                if (ColsHasIndex(mask + (*top)*stride + col - A, stride, *bottom - *top, _index, cols))
                {
                    for (ptrdiff_t i = A - 1; i >= 0; i--)
                    {
                        if (cols[i])
                        {
                            *right = col - A + i + 1;
                            break;
                        }
                    }
                    search = false;
                    break;
                }
            }
        }
    }
#endif//SIMD_AVX2_ENABLE
}
