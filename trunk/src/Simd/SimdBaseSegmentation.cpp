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
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void FillSingleHole(uint8_t * mask, ptrdiff_t stride, uint8_t index)
        {
            if(mask[-stride] == index && mask[stride] == index && mask[-1] == index && mask[1] == index)
                mask[0] = index;
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > 2 && height > 2);

            mask += stride + 1;
            height -= 2;
            width -= 2;
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    FillSingleHole(mask + col, stride, index);
                }
                mask += stride;
            }
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*left < *right && *top < *bottom);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)width);

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

            if(search)
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