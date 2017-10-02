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

        SIMD_INLINE void SegmentationPropagate2x2(const uint8x16_t & parentOne, const uint8x16_t & parentAll,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const uint8x16_t & index, const uint8x16_t & invalid, const uint8x16_t & empty, const uint8x16_t & threshold)
        {
            const uint8x16_t _difference0 = Load<false>(difference0 + childCol);
            const uint8x16_t _difference1 = Load<false>(difference1 + childCol);
            const uint8x16_t _child0 = Load<false>(child0 + childCol);
            const uint8x16_t _child1 = Load<false>(child1 + childCol);
            const uint8x16_t condition0 = vorrq_u8(parentAll, vandq_u8(parentOne, vcgtq_u8(_difference0, threshold)));
            const uint8x16_t condition1 = vorrq_u8(parentAll, vandq_u8(parentOne, vcgtq_u8(_difference1, threshold)));
            Store<false>(child0 + childCol, vbslq_u8(vcltq_u8(_child0, invalid), vbslq_u8(condition0, index, empty), _child0));
            Store<false>(child1 + childCol, vbslq_u8(vcltq_u8(_child1, invalid), vbslq_u8(condition1, index, empty), _child1));
        }

        template<bool align> SIMD_INLINE void SegmentationPropagate2x2(const uint8_t * parent0, const uint8_t * parent1, size_t parentCol,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const uint8x16_t & index, const uint8x16_t & invalid, const uint8x16_t & empty, const uint8x16_t & threshold)
        {
            const uint8x16_t parent00 = vceqq_u8(Load<align>(parent0 + parentCol), index);
            const uint8x16_t parent01 = vceqq_u8(Load<false>(parent0 + parentCol + 1), index);
            const uint8x16_t parent10 = vceqq_u8(Load<align>(parent1 + parentCol), index);
            const uint8x16_t parent11 = vceqq_u8(Load<false>(parent1 + parentCol + 1), index);
            const uint8x16_t parentOne = vorrq_u8(vorrq_u8(parent00, parent01), vorrq_u8(parent10, parent11));
            const uint8x16_t parentAll = vandq_u8(vandq_u8(parent00, parent01), vandq_u8(parent10, parent11));

            SegmentationPropagate2x2(Stretch2<0>(parentOne), Stretch2<0>(parentAll),
                difference0, difference1, child0, child1, childCol, index, invalid, empty, threshold);

            SegmentationPropagate2x2(Stretch2<1>(parentOne), Stretch2<1>(parentAll),
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
            uint8x16_t index = vdupq_n_u8(currentIndex);
            uint8x16_t invalid = vdupq_n_u8(invalidIndex);
            uint8x16_t empty = vdupq_n_u8(emptyIndex);
            uint8x16_t threshold = vdupq_n_u8(differenceThreshold);

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

        SIMD_INLINE bool IsNotZero(uint8x16_t value)
        {
            uint32x2_t tmp = (uint32x2_t)vorr_u8(Half<0>(value), Half<1>(value));
            return vget_lane_u32(vpmax_u32(tmp, tmp), 0);
        }

        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, uint8x16_t index)
        {
            for (size_t col = 0; col < alignedSize; col += A)
            {
                if (IsNotZero(vceqq_u8(Load<false>(mask + col), index)))
                    return true;
            }
            if (alignedSize != fullSize)
            {
                if (IsNotZero(vceqq_u8(Load<false>(mask + fullSize - A), index)))
                    return true;
            }
            return false;
        }

        SIMD_INLINE bool ColsHasIndex(const uint8_t * mask, size_t stride, size_t size, uint8x16_t index, uint8_t * cols)
        {
            uint8x16_t _cols = K8_00;
            for (size_t row = 0; row < size; ++row)
            {
                _cols = vorrq_u8(_cols, vceqq_u8(Load<false>(mask), index));
                mask += stride;
            }
            Store<false>(cols, _cols);
            return IsNotZero(_cols);
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*right - *left >= (ptrdiff_t)A && *bottom > *top);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            size_t fullWidth = *right - *left;
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            uint8x16_t _index = vdupq_n_u8(index);
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
#endif//SIMD_NEON_ENABLE
}
