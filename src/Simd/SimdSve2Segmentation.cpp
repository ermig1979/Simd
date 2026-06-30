/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void FillSingleHoles(uint8_t* mask, ptrdiff_t stride, size_t offset, size_t width, const svuint8_t& index)
        {
            const svbool_t tail = svwhilelt_b8(offset, width);
            const svuint8_t up = svld1_u8(tail, mask - stride + offset);
            const svuint8_t left = svld1_u8(tail, mask + offset - 1);
            const svuint8_t right = svld1_u8(tail, mask + offset + 1);
            const svuint8_t down = svld1_u8(tail, mask + stride + offset);
            const svuint8_t current = svld1_u8(tail, mask + offset);
            svbool_t fill = svcmpeq_u8(tail, up, index);
            fill = svand_b_z(tail, fill, svcmpeq_u8(tail, left, index));
            fill = svand_b_z(tail, fill, svcmpeq_u8(tail, right, index));
            fill = svand_b_z(tail, fill, svcmpeq_u8(tail, down, index));
            svst1_u8(tail, mask + offset, svsel_u8(fill, index, current));
        }

        void SegmentationFillSingleHoles(uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            const size_t A = svcntb();
            assert(width > A + 2 && height > 2);

            const svuint8_t _index = svdup_n_u8(index);
            const size_t bodyEnd = width - 1;
            for (size_t row = 1; row < height - 1; ++row)
            {
                uint8_t* current = mask + row * stride;
                for (size_t col = 1; col < bodyEnd; col += A)
                    FillSingleHoles(current, (ptrdiff_t)stride, col, bodyEnd, _index);
            }
        }

        SIMD_INLINE void ChangeIndex(uint8_t* mask, const svuint8_t& oldIndex, const svuint8_t& newIndex, const svbool_t& tail)
        {
            svuint8_t _mask = svld1_u8(tail, mask);
            svst1_u8(tail, mask, svsel_u8(svcmpeq_u8(tail, _mask, oldIndex), newIndex, _mask));
        }

        void SegmentationChangeIndex(uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            const size_t A = svcntb();
            const svuint8_t _oldIndex = svdup_n_u8(oldIndex);
            const svuint8_t _newIndex = svdup_n_u8(newIndex);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; col += A)
                    ChangeIndex(mask + col, _oldIndex, _newIndex, svwhilelt_b8(col, width));
                mask += stride;
            }
        }

        SIMD_INLINE bool RowHasIndex(const uint8_t* mask, size_t width, const svuint8_t& index)
        {
            const size_t A = svcntb();
            const svbool_t full = svptrue_b8();
            size_t col = 0;
            for (; col + A <= width; col += A)
            {
                if (svptest_any(full, svcmpeq_u8(full, svld1_u8(full, mask + col), index)))
                    return true;
            }
            if (col < width)
            {
                const svbool_t tail = svwhilelt_b8(col, width);
                if (svptest_any(tail, svcmpeq_u8(tail, svld1_u8(tail, mask + col), index)))
                    return true;
            }
            return false;
        }

        SIMD_INLINE bool ColsHasIndex(const uint8_t* mask, size_t stride, size_t height, const svuint8_t& index, size_t width)
        {
            const svbool_t tail = svwhilelt_b8((size_t)0, width);
            svbool_t cols = svpfalse_b();
            for (size_t row = 0; row < height; ++row)
            {
                cols = svorr_b_z(tail, cols, svcmpeq_u8(tail, svld1_u8(tail, mask), index));
                mask += stride;
            }
            return svptest_any(tail, cols);
        }

        void SegmentationShrinkRegion(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t* left, ptrdiff_t* top, ptrdiff_t* right, ptrdiff_t* bottom)
        {
            const size_t A = svcntb();
            assert(*right - *left >= (ptrdiff_t)A && *bottom > *top);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            const svuint8_t _index = svdup_n_u8(index);
            bool search = true;
            for (ptrdiff_t row = *top; search && row < *bottom; ++row)
            {
                if (RowHasIndex(mask + row * stride + *left, *right - *left, _index))
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
                if (RowHasIndex(mask + row * stride + *left, *right - *left, _index))
                {
                    search = false;
                    *bottom = row + 1;
                }
            }

            search = true;
            for (ptrdiff_t col = *left; search && col < *right; col += A)
            {
                const size_t block = (size_t)(*right - col >= (ptrdiff_t)A ? A : *right - col);
                if (ColsHasIndex(mask + (*top) * stride + col, stride, *bottom - *top, _index, block))
                {
                    for (size_t i = 0; i < block; ++i)
                    {
                        const uint8_t* current = mask + (*top) * stride + col + i;
                        for (ptrdiff_t row = *top; row < *bottom; ++row)
                        {
                            if (*current == index)
                            {
                                *left = col + i;
                                search = false;
                                break;
                            }
                            current += stride;
                        }
                        if (!search)
                            break;
                    }
                }
            }

            search = true;
            for (ptrdiff_t col = *right; search && col > *left;)
            {
                const ptrdiff_t start = col - (ptrdiff_t)A > *left ? col - (ptrdiff_t)A : *left;
                const size_t block = (size_t)(col - start);
                if (ColsHasIndex(mask + (*top) * stride + start, stride, *bottom - *top, _index, block))
                {
                    for (ptrdiff_t i = (ptrdiff_t)block - 1; i >= 0; --i)
                    {
                        const uint8_t* current = mask + (*top) * stride + start + i;
                        for (ptrdiff_t row = *top; row < *bottom; ++row)
                        {
                            if (*current == index)
                            {
                                *right = start + i + 1;
                                search = false;
                                break;
                            }
                            current += stride;
                        }
                        if (!search)
                            break;
                    }
                }
                col = start;
            }
        }

        SIMD_INLINE void SegmentationPropagate2x2(const svbool_t& parentOne, const svbool_t& parentAll,
            const uint8_t* difference0, const uint8_t* difference1, uint8_t* child0, uint8_t* child1, size_t childCol,
            uint8_t invalidIndex, const svuint8_t& current, const svuint8_t& empty, uint8_t differenceThreshold, size_t childLimit)
        {
            const svbool_t tail0 = svwhilelt_b8(childCol, childLimit);
            const svbool_t tail1 = svwhilelt_b8(childCol + svcntb(), childLimit);

            const svbool_t parentOne0 = svzip1_b8(parentOne, parentOne);
            const svbool_t parentOne1 = svzip2_b8(parentOne, parentOne);
            const svbool_t parentAll0 = svzip1_b8(parentAll, parentAll);
            const svbool_t parentAll1 = svzip2_b8(parentAll, parentAll);

            const svuint8_t difference00 = svld1_u8(tail0, difference0 + childCol);
            const svuint8_t difference10 = svld1_u8(tail0, difference1 + childCol);
            const svuint8_t difference01 = svld1_u8(tail1, difference0 + childCol + svcntb());
            const svuint8_t difference11 = svld1_u8(tail1, difference1 + childCol + svcntb());
            const svuint8_t _child00 = svld1_u8(tail0, child0 + childCol);
            const svuint8_t _child10 = svld1_u8(tail0, child1 + childCol);
            const svuint8_t _child01 = svld1_u8(tail1, child0 + childCol + svcntb());
            const svuint8_t _child11 = svld1_u8(tail1, child1 + childCol + svcntb());

            const svbool_t differenceMask00 = svcmpgt_n_u8(tail0, difference00, differenceThreshold);
            const svbool_t differenceMask10 = svcmpgt_n_u8(tail0, difference10, differenceThreshold);
            const svbool_t differenceMask01 = svcmpgt_n_u8(tail1, difference01, differenceThreshold);
            const svbool_t differenceMask11 = svcmpgt_n_u8(tail1, difference11, differenceThreshold);

            const svbool_t condition00 = svorr_b_z(tail0, parentAll0, svand_b_z(tail0, parentOne0, differenceMask00));
            const svbool_t condition10 = svorr_b_z(tail0, parentAll0, svand_b_z(tail0, parentOne0, differenceMask10));
            const svbool_t condition01 = svorr_b_z(tail1, parentAll1, svand_b_z(tail1, parentOne1, differenceMask01));
            const svbool_t condition11 = svorr_b_z(tail1, parentAll1, svand_b_z(tail1, parentOne1, differenceMask11));

            const svbool_t update00 = svcmplt_n_u8(tail0, _child00, invalidIndex);
            const svbool_t update10 = svcmplt_n_u8(tail0, _child10, invalidIndex);
            const svbool_t update01 = svcmplt_n_u8(tail1, _child01, invalidIndex);
            const svbool_t update11 = svcmplt_n_u8(tail1, _child11, invalidIndex);

            const svuint8_t propagated00 = svsel_u8(condition00, current, empty);
            const svuint8_t propagated10 = svsel_u8(condition10, current, empty);
            const svuint8_t propagated01 = svsel_u8(condition01, current, empty);
            const svuint8_t propagated11 = svsel_u8(condition11, current, empty);

            svst1_u8(tail0, child0 + childCol, svsel_u8(update00, propagated00, _child00));
            svst1_u8(tail0, child1 + childCol, svsel_u8(update10, propagated10, _child10));
            svst1_u8(tail1, child0 + childCol + svcntb(), svsel_u8(update01, propagated01, _child01));
            svst1_u8(tail1, child1 + childCol + svcntb(), svsel_u8(update11, propagated11, _child11));
        }

        void SegmentationPropagate2x2(const uint8_t* parent, size_t parentStride, size_t width, size_t height,
            uint8_t* child, size_t childStride, const uint8_t* difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            assert(width >= 2 && height >= 2);

            const size_t A = svcntb();
            const svuint8_t index = svdup_n_u8(currentIndex);
            const svuint8_t empty = svdup_n_u8(emptyIndex);
            const size_t parentWidth = width - 1;
            const size_t parentHeight = height - 1;
            const size_t childLimit = parentWidth * 2 + 1;
            for (size_t parentRow = 0, childRow = 1; parentRow < parentHeight; ++parentRow, childRow += 2)
            {
                const uint8_t* parent0 = parent + parentRow * parentStride;
                const uint8_t* parent1 = parent0 + parentStride;
                const uint8_t* difference0 = difference + childRow * differenceStride;
                const uint8_t* difference1 = difference0 + differenceStride;
                uint8_t* child0 = child + childRow * childStride;
                uint8_t* child1 = child0 + childStride;
                for (size_t parentCol = 0, childCol = 1; parentCol < parentWidth; parentCol += A, childCol += 2 * A)
                {
                    const svbool_t tail = svwhilelt_b8(parentCol, parentWidth);
                    const svbool_t parent00 = svcmpeq_u8(tail, svld1_u8(tail, parent0 + parentCol), index);
                    const svbool_t parent01 = svcmpeq_u8(tail, svld1_u8(tail, parent0 + parentCol + 1), index);
                    const svbool_t parent10 = svcmpeq_u8(tail, svld1_u8(tail, parent1 + parentCol), index);
                    const svbool_t parent11 = svcmpeq_u8(tail, svld1_u8(tail, parent1 + parentCol + 1), index);
                    const svbool_t parentOne = svorr_b_z(tail, svorr_b_z(tail, parent00, parent01), svorr_b_z(tail, parent10, parent11));
                    const svbool_t parentAll = svand_b_z(tail, svand_b_z(tail, parent00, parent01), svand_b_z(tail, parent10, parent11));
                    SegmentationPropagate2x2(parentOne, parentAll, difference0, difference1, child0, child1, childCol,
                        invalidIndex, index, empty, differenceThreshold, childLimit);
                }
            }
        }
    }
#endif//SIMD_SVE2_ENABLE
}
