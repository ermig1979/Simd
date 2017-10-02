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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void FillSingleHole(uint8_t * mask, ptrdiff_t stride, uint8_t index)
        {
            if (mask[-stride] == index && mask[stride] == index && mask[-1] == index && mask[1] == index)
                mask[0] = index;
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > 2 && height > 2);

            mask += stride + 1;
            height -= 2;
            width -= 2;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    FillSingleHole(mask + col, stride, index);
                }
                mask += stride;
            }
        }

        void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (mask[col] == oldIndex)
                        mask[col] = newIndex;
                }
                mask += stride;
            }
        }

        void SegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height,
            uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            assert(width >= 2 && height >= 2);

            width--;
            height--;
            for (size_t parentRow = 0, childRow = 1; parentRow < height; ++parentRow, childRow += 2)
            {
                const uint8_t * parent0 = parent + parentRow*parentStride;
                const uint8_t * parent1 = parent0 + parentStride;
                const uint8_t * difference0 = difference + childRow*differenceStride;
                const uint8_t * difference1 = difference0 + differenceStride;
                uint8_t * child0 = child + childRow*childStride;
                uint8_t * child1 = child0 + childStride;
                for (size_t parentCol = 0, childCol = 1; parentCol < width; ++parentCol, childCol += 2)
                {
                    const bool parent00 = parent0[parentCol] == currentIndex;
                    const bool parent01 = parent0[parentCol + 1] == currentIndex;
                    const bool parent10 = parent1[parentCol] == currentIndex;
                    const bool parent11 = parent1[parentCol + 1] == currentIndex;

                    const bool parentOne = parent00 || parent01 || parent10 || parent11;
                    const bool parentAll = parent00 && parent01 && parent10 && parent11;

                    const bool difference00 = difference0[childCol] > differenceThreshold;
                    const bool difference01 = difference0[childCol + 1] > differenceThreshold;
                    const bool difference10 = difference1[childCol] > differenceThreshold;
                    const bool difference11 = difference1[childCol + 1] > differenceThreshold;

                    uint8_t & child00 = child0[childCol];
                    uint8_t & child01 = child0[childCol + 1];
                    uint8_t & child10 = child1[childCol];
                    uint8_t & child11 = child1[childCol + 1];

                    if (child00 < invalidIndex)
                        child00 = parentAll || (parentOne && difference00) ? currentIndex : emptyIndex;

                    if (child01 < invalidIndex)
                        child01 = parentAll || (parentOne && difference01) ? currentIndex : emptyIndex;

                    if (child10 < invalidIndex)
                        child10 = parentAll || (parentOne && difference10) ? currentIndex : emptyIndex;

                    if (child11 < invalidIndex)
                        child11 = parentAll || (parentOne && difference11) ? currentIndex : emptyIndex;
                }
            }
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*left < *right && *top < *bottom);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            bool search = true;
            for (ptrdiff_t row = *top; search && row < *bottom; ++row)
            {
                const uint8_t * _mask = mask + row*stride;
                for (ptrdiff_t col = *left; col < *right; ++col)
                {
                    if (_mask[col] == index)
                    {
                        search = false;
                        *top = row;
                        break;
                    }
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
                const uint8_t * _mask = mask + row*stride;
                for (ptrdiff_t col = *left; col < *right; ++col)
                {
                    if (_mask[col] == index)
                    {
                        search = false;
                        *bottom = row + 1;
                        break;
                    }
                }
            }

            search = true;
            for (ptrdiff_t col = *left; search && col < *right; ++col)
            {
                const uint8_t * _mask = mask + (*top)*stride + col;
                for (ptrdiff_t row = *top; row < *bottom; ++row)
                {
                    if (*_mask == index)
                    {
                        search = false;
                        *left = col;
                        break;
                    }
                    _mask += stride;
                }
            }

            search = true;
            for (ptrdiff_t col = *right - 1; search && col >= *left; --col)
            {
                const uint8_t * _mask = mask + (*top)*stride + col;
                for (ptrdiff_t row = *top; row < *bottom; ++row)
                {
                    if (*_mask == index)
                    {
                        search = false;
                        *right = col + 1;
                        break;
                    }
                    _mask += stride;
                }
            }
        }
    }
}
