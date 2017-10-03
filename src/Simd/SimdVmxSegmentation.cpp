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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, v128_u8 index)
        {
            for (size_t col = 0; col < alignedSize; col += A)
            {
                if (vec_any_eq(Load<false>(mask + col), index))
                    return true;
            }
            if (alignedSize != fullSize)
            {
                if (vec_any_eq(Load<false>(mask + fullSize - A), index))
                    return true;
            }
            return false;
        }

        SIMD_INLINE bool ColsHasIndex(const uint8_t * mask, size_t stride, size_t size, v128_u8 index, uint8_t * cols)
        {
            v128_u8 _cols = K8_00;
            for (size_t row = 0; row < size; ++row)
            {
                _cols = vec_or(_cols, (v128_u8)vec_cmpeq(Load<false>(mask), index));
                mask += stride;
            }
            Store<true>(cols, _cols);
            return vec_any_eq(_cols, K8_FF);
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*right - *left >= (ptrdiff_t)A && *bottom > *top);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            size_t fullWidth = *right - *left;
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            v128_u8 _index = SetU8(index);
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
                SIMD_ALIGNED(16) uint8_t cols[A];
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
                SIMD_ALIGNED(16) uint8_t cols[A];
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

        template<bool align> SIMD_INLINE v128_u8 FillSingleHoles(uint8_t * src, ptrdiff_t stride, const v128_u8 & index)
        {
            const v128_u8 current = Load<align>(src);
            const v128_u8 up = (v128_u8)vec_cmpeq(Load<align>(src - stride), index);
            const v128_u8 left = (v128_u8)vec_cmpeq(Load<false>(src - 1), index);
            const v128_u8 right = (v128_u8)vec_cmpeq(Load<false>(src + 1), index);
            const v128_u8 down = (v128_u8)vec_cmpeq(Load<align>(src + stride), index);
            return vec_sel(current, index, vec_and(vec_and(up, left), vec_and(right, down)));
        }

        template<bool align> void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > A + 2 && height > 2);

            height -= 1;
            width -= 1;
            v128_u8 _index = SetU8(index);
            size_t alignedWidth = Simd::AlignLo(width, A);
            for (size_t row = 1; row < height; ++row)
            {
                mask += stride;

                Store<false>(mask + 1, FillSingleHoles<false>(mask + 1, stride, _index));

                if (width > DA + 2)
                {
                    Storer<align> _dst(mask + A);
                    _dst.First(FillSingleHoles<align>(mask + A, stride, _index));
                    for (size_t col = DA; col < alignedWidth; col += A)
                        _dst.Next(FillSingleHoles<align>(mask + col, stride, _index));
                    Flush(_dst);
                }

                if (alignedWidth != width)
                    Store<false>(mask + width - A, FillSingleHoles<false>(mask + width - A, stride, _index));
            }
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            if (Aligned(mask) && Aligned(stride))
                SegmentationFillSingleHoles<true>(mask, stride, width, height, index);
            else
                SegmentationFillSingleHoles<false>(mask, stride, width, height, index);
        }

        SIMD_INLINE v128_u8 ChangeIndex(const v128_u8 & mask, const v128_u8 & oldIndex, const v128_u8 & newIndex)
        {
            return vec_sel(mask, newIndex, (v128_u8)vec_cmpeq(mask, oldIndex));
        }

        template<bool align> void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            v128_u8 _oldIndex = SetU8(oldIndex);
            v128_u8 _newIndex = SetU8(newIndex);
            size_t alignedWidth = Simd::AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(mask);
                _dst.First(ChangeIndex(Load<align>(mask), _oldIndex, _newIndex));
                for (size_t col = A; col < alignedWidth; col += A)
                    _dst.Next(ChangeIndex(Load<align>(mask + col), _oldIndex, _newIndex));
                Flush(_dst);

                if (alignedWidth != width)
                    Store<false>(mask + width - A, ChangeIndex(Load<false>(mask + width - A), _oldIndex, _newIndex));
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

        template<bool first>
        SIMD_INLINE void SegmentationPropagate2x2(const v128_u8 & parentOne, const v128_u8 & parentAll,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * childSrc0, uint8_t * childSrc1, size_t childCol,
            const v128_u8 & index, const v128_u8 & invalid, const v128_u8 & empty, const v128_u8 & threshold,
            Storer<false> & childDst0, Storer<false> & childDst1)
        {
            const v128_u8 _difference0 = Load<false>(difference0 + childCol);
            const v128_u8 _difference1 = Load<false>(difference1 + childCol);
            const v128_u8 child0 = Load<false>(childSrc0 + childCol);
            const v128_u8 child1 = Load<false>(childSrc1 + childCol);
            const v128_u8 condition0 = vec_or(parentAll, vec_and(parentOne, vec_cmpgt(_difference0, threshold)));
            const v128_u8 condition1 = vec_or(parentAll, vec_and(parentOne, vec_cmpgt(_difference1, threshold)));
            Store<false, first>(childDst0, vec_sel(child0, vec_sel(empty, index, condition0), vec_cmplt(child0, invalid)));
            Store<false, first>(childDst1, vec_sel(child1, vec_sel(empty, index, condition1), vec_cmplt(child1, invalid)));
        }

        template<bool align, bool first> SIMD_INLINE void SegmentationPropagate2x2(const uint8_t * parent0, const uint8_t * parent1, size_t parentCol,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * childSrc0, uint8_t * childSrc1, size_t childCol,
            const v128_u8 & index, const v128_u8 & invalid, const v128_u8 & empty, const v128_u8 & threshold,
            Storer<false> & childDst0, Storer<false> & childDst1)
        {
            const v128_u8 parent00 = (v128_u8)vec_cmpeq(Load<align>(parent0 + parentCol), index);
            const v128_u8 parent01 = (v128_u8)vec_cmpeq(Load<false>(parent0 + parentCol + 1), index);
            const v128_u8 parent10 = (v128_u8)vec_cmpeq(Load<align>(parent1 + parentCol), index);
            const v128_u8 parent11 = (v128_u8)vec_cmpeq(Load<false>(parent1 + parentCol + 1), index);
            const v128_u8 parentOne = vec_or(vec_or(parent00, parent01), vec_or(parent10, parent11));
            const v128_u8 parentAll = vec_and(vec_and(parent00, parent01), vec_and(parent10, parent11));

            SegmentationPropagate2x2<first>((v128_u8)UnpackLoU8(parentOne, parentOne), (v128_u8)UnpackLoU8(parentAll, parentAll),
                difference0, difference1, childSrc0, childSrc1, childCol, index, invalid, empty, threshold, childDst0, childDst1);

            SegmentationPropagate2x2<false>((v128_u8)UnpackHiU8(parentOne, parentOne), (v128_u8)UnpackHiU8(parentAll, parentAll),
                difference0, difference1, childSrc0, childSrc1, childCol + A, index, invalid, empty, threshold, childDst0, childDst1);
        }

        template<bool align> void SegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height,
            uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            assert(width >= A + 1 && height >= 2);
            height--;
            width--;

            size_t alignedWidth = Simd::AlignLo(width, A);
            v128_u8 index = SetU8(currentIndex);
            v128_u8 invalid = SetU8(invalidIndex);
            v128_u8 empty = SetU8(emptyIndex);
            v128_u8 threshold = SetU8(differenceThreshold);

            for (size_t parentRow = 0, childRow = 1; parentRow < height; ++parentRow, childRow += 2)
            {
                const uint8_t * parent0 = parent + parentRow*parentStride;
                const uint8_t * parent1 = parent0 + parentStride;
                const uint8_t * difference0 = difference + childRow*differenceStride;
                const uint8_t * difference1 = difference0 + differenceStride;
                uint8_t * child0 = child + childRow*childStride;
                uint8_t * child1 = child0 + childStride;

                Storer<false> childDst0(child0 + 1), childDst1(child1 + 1);
                SegmentationPropagate2x2<align, true>(parent0, parent1, 0, difference0, difference1,
                    child0, child1, 1, index, invalid, empty, threshold, childDst0, childDst1);
                for (size_t parentCol = A, childCol = DA + 1; parentCol < alignedWidth; parentCol += A, childCol += DA)
                    SegmentationPropagate2x2<align, false>(parent0, parent1, parentCol, difference0, difference1,
                        child0, child1, childCol, index, invalid, empty, threshold, childDst0, childDst1);
                Flush(childDst0, childDst1);

                if (alignedWidth != width)
                {
                    size_t childCol = (width - A) * 2 + 1;
                    Storer<false> childDst0(child0 + childCol), childDst1(child1 + childCol);
                    SegmentationPropagate2x2<false, true>(parent0, parent1, width - A, difference0, difference1,
                        child0, child1, childCol, index, invalid, empty, threshold, childDst0, childDst1);
                    Flush(childDst0, childDst1);
                }
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
#endif// SIMD_VMX_ENABLE
}
