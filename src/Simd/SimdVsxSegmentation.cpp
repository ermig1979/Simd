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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, v128_u8 index)
        {
            for (size_t col = 0; col < alignedSize; col += A)
            {
                if(vec_any_eq(Load<false>(mask + col), index))
                    return true;
            }
            if(alignedSize != fullSize)
            {
                if(vec_any_eq(Load<false>(mask + fullSize - A), index))
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
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)width);

            size_t fullWidth = *right - *left;
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            v128_u8 _index = SetU8(index);
            bool search = true;
            for (ptrdiff_t row = *top; search && row < *bottom; ++row)
            {
                if(RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index))
                {
                    search = false;
                    *top = row;
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
                if(RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index))
                {
                    search = false;
                    *bottom = row + 1;
                }
            }

            search = true;
            for (ptrdiff_t col = *left; search && col < *left + alignedWidth; col += A)
            {
                SIMD_ALIGNED(16) uint8_t cols[A];
                if(ColsHasIndex(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols))
                {
                    for(size_t i = 0; i < A; i++)
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
                if(ColsHasIndex(mask + (*top)*stride + col - A, stride, *bottom - *top, _index, cols))
                {
                    for(ptrdiff_t i = A - 1; i >= 0; i--)
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
#endif// SIMD_VSX_ENABLE
}