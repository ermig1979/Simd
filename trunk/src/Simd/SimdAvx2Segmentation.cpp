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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, __m256i index)
        {
            for (size_t col = 0; col < alignedSize; col += A)
            {
                if(!_mm256_testz_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)(mask + col)), index), K_INV_ZERO))
                    return true;
            }
            if(alignedSize != fullSize)
            {
                if(!_mm256_testz_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i*)(mask + fullSize - A)), index), K_INV_ZERO))
                    return true;
            }
            return false;
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*right - *left >= A && *bottom > *top);
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)width);

            size_t fullWidth = *right - *left;          
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            __m256i _index = _mm256_set1_epi8(index);
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
#endif//SIMD_AVX2_ENABLE
}