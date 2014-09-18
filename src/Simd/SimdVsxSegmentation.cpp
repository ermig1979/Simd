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
            for(size_t row = 1; row < height; ++row)
            {
                mask += stride;

                Store<false>(mask + 1, FillSingleHoles<false>(mask + 1, stride, _index));

                if(width > DA + 2)
                {
                    Storer<align> _dst(mask + A);
                    _dst.First(FillSingleHoles<align>(mask + A, stride, _index));
                    for(size_t col = DA; col < alignedWidth; col += A)
                        _dst.Next(FillSingleHoles<align>(mask + col, stride, _index));
                    _dst.Flush();
                }

                if(alignedWidth != width)
                    Store<false>(mask + width - A, FillSingleHoles<false>(mask + width - A, stride, _index));
            }
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            if(Aligned(mask) && Aligned(stride))
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
            for(size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(mask);
                _dst.First(ChangeIndex(Load<align>(mask), _oldIndex, _newIndex));
                for(size_t col = A; col < alignedWidth; col += A)
                    _dst.Next(ChangeIndex(Load<align>(mask + col), _oldIndex, _newIndex));
                _dst.Flush();

                if(alignedWidth != width)
                    Store<false>(mask + width - A, ChangeIndex(Load<false>(mask + width - A), _oldIndex, _newIndex));
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
    }
#endif// SIMD_VSX_ENABLE
}