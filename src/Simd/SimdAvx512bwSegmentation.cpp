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
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        template<bool masked> SIMD_INLINE void ChangeIndex(uint8_t * mask, __m512i oldIndex, __m512i newIndex, __mmask64 tail = -1)
        {
            _mm512_mask_storeu_epi8(mask, _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(tail, mask) : _mm512_loadu_si512(mask))), oldIndex)&tail, newIndex);
        }

        SIMD_INLINE void ChangeIndex4(uint8_t * mask, __m512i oldIndex, __m512i newIndex)
        {
            _mm512_mask_storeu_epi8(mask + 0 * A, _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(mask + 0 * A), oldIndex), newIndex);
            _mm512_mask_storeu_epi8(mask + 1 * A, _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(mask + 1 * A), oldIndex), newIndex);
            _mm512_mask_storeu_epi8(mask + 2 * A, _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(mask + 2 * A), oldIndex), newIndex);
            _mm512_mask_storeu_epi8(mask + 3 * A, _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(mask + 3 * A), oldIndex), newIndex);
        }

        void SegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
        {
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _oldIndex = _mm512_set1_epi8((char)oldIndex);
            __m512i _newIndex = _mm512_set1_epi8((char)newIndex);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    ChangeIndex4(mask + col, _oldIndex, _newIndex);
                for (; col < alignedWidth; col += A)
                    ChangeIndex<false>(mask + col, _oldIndex, _newIndex);
                if (col < width)
                    ChangeIndex<true>(mask + col, _oldIndex, _newIndex, tailMask);
                mask += stride;
            }
        }

        template<bool masked> SIMD_INLINE void FillSingleHoles(uint8_t * mask, ptrdiff_t stride, __m512i index, __mmask64 edge = -1)
        {
            __mmask64 up = _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(edge, mask - stride) : _mm512_loadu_si512(mask - stride))), index);
            __mmask64 left = _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(edge, mask - 1) : _mm512_loadu_si512(mask - 1))), index);
            __mmask64 right = _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(edge, mask + 1) : _mm512_loadu_si512(mask + 1))), index);
            __mmask64 down = _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(edge, mask + stride) : _mm512_loadu_si512(mask + stride))), index);
            _mm512_mask_storeu_epi8(mask, up & left & right & down & edge, index);
        }

        void SegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
        {
            assert(width > 2 && height > 2);

            __m512i _index = _mm512_set1_epi8((char)index);
            size_t alignedWidth = Simd::AlignLo(width - 1, A);
            __mmask64 noseMask = NoseMask64(A - 1);
            __mmask64 tailMask = TailMask64(width - 1 - alignedWidth);
            if (alignedWidth < A)
                noseMask = noseMask&tailMask;

            for (size_t row = 2; row < height; ++row)
            {
                mask += stride;
                size_t col = A;
                FillSingleHoles<true>(mask, stride, _index, noseMask);
                for (; col < alignedWidth; col += A)
                    FillSingleHoles<false>(mask + col, stride, _index);
                if (col < width)
                    FillSingleHoles<true>(mask + col, stride, _index, tailMask);
            }
        }

        template<bool mask> SIMD_INLINE void SegmentationPropagate2x2(__mmask32 parentOne, __mmask32 parentAll,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m512i & index, const __m512i & invalid, const __m512i & empty, const __m512i & threshold, __mmask32 tail)
        {
            __m512i _difference0 = _mm512_mask_set1_epi16(_mm512_maskz_loadu_epi16(tail&parentOne, (uint16_t*)(difference0 + childCol)), parentAll, -1);
            __m512i _difference1 = _mm512_mask_set1_epi16(_mm512_maskz_loadu_epi16(tail&parentOne, (uint16_t*)(difference1 + childCol)), parentAll, -1);
            __m512i _child0 = (mask ? _mm512_maskz_loadu_epi16(tail, (uint16_t*)(child0 + childCol)) : _mm512_loadu_si512((uint16_t*)(child0 + childCol)));
            __m512i _child1 = (mask ? _mm512_maskz_loadu_epi16(tail, (uint16_t*)(child1 + childCol)) : _mm512_loadu_si512((uint16_t*)(child1 + childCol)));
            __mmask64 condition0 = _mm512_cmpgt_epu8_mask(_difference0, threshold);
            __mmask64 condition1 = _mm512_cmpgt_epu8_mask(_difference1, threshold);
            if (mask)
                _mm512_mask_storeu_epi16((uint16_t*)(child0 + childCol), tail, _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(_child0, invalid), _child0, _mm512_mask_blend_epi8(condition0, empty, index)));
            else
                _mm512_storeu_si512((uint16_t*)(child0 + childCol), _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(_child0, invalid), _child0, _mm512_mask_blend_epi8(condition0, empty, index)));
            if (mask)
                _mm512_mask_storeu_epi16((uint16_t*)(child1 + childCol), tail, _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(_child1, invalid), _child1, _mm512_mask_blend_epi8(condition1, empty, index)));
            else
                _mm512_storeu_si512((uint16_t*)(child1 + childCol), _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(_child1, invalid), _child1, _mm512_mask_blend_epi8(condition1, empty, index)));
        }

        template<bool mask> SIMD_INLINE void SegmentationPropagate2x2(const uint8_t * parent0, const uint8_t * parent1, size_t parentCol,
            const uint8_t * difference0, const uint8_t * difference1, uint8_t * child0, uint8_t * child1, size_t childCol,
            const __m512i & index, const __m512i & invalid, const __m512i & empty, const __m512i & threshold, __mmask64 tail = -1)
        {
            __mmask64 parent00 = _mm512_cmpeq_epi8_mask(((mask ? _mm512_maskz_loadu_epi8(tail, parent0 + parentCol) : _mm512_loadu_si512(parent0 + parentCol))), index);
            __mmask64 parent01 = _mm512_cmpeq_epi8_mask(((mask ? _mm512_maskz_loadu_epi8(tail, parent0 + parentCol + 1) : _mm512_loadu_si512(parent0 + parentCol + 1))), index);
            __mmask64 parent10 = _mm512_cmpeq_epi8_mask(((mask ? _mm512_maskz_loadu_epi8(tail, parent1 + parentCol) : _mm512_loadu_si512(parent1 + parentCol))), index);
            __mmask64 parent11 = _mm512_cmpeq_epi8_mask(((mask ? _mm512_maskz_loadu_epi8(tail, parent1 + parentCol + 1) : _mm512_loadu_si512(parent1 + parentCol + 1))), index);
            __mmask64 one = parent00 | parent01 | parent10 | parent11;
            __mmask64 all = parent00 & parent01 & parent10 & parent11;
            SegmentationPropagate2x2<mask>(__mmask32(one >> 00), __mmask32(all >> 00), difference0, difference1, child0, child1, childCol + 0, index, invalid, empty, threshold, __mmask32(tail >> 00));
            SegmentationPropagate2x2<mask>(__mmask32(one >> 32), __mmask32(all >> 32), difference0, difference1, child0, child1, childCol + A, index, invalid, empty, threshold, __mmask32(tail >> 32));
        }

        void SegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height,
            uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
        {
            assert(width >= 2 && height >= 2);
            height--;
            width--;

            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i index = _mm512_set1_epi8((char)currentIndex);
            __m512i invalid = _mm512_set1_epi8((char)invalidIndex);
            __m512i empty = _mm512_set1_epi8((char)emptyIndex);
            __m512i threshold = _mm512_set1_epi8((char)differenceThreshold);

            for (size_t parentRow = 0, childRow = 1; parentRow < height; ++parentRow, childRow += 2)
            {
                const uint8_t * parent0 = parent + parentRow*parentStride;
                const uint8_t * parent1 = parent0 + parentStride;
                const uint8_t * difference0 = difference + childRow*differenceStride;
                const uint8_t * difference1 = difference0 + differenceStride;
                uint8_t * child0 = child + childRow*childStride;
                uint8_t * child1 = child0 + childStride;

                size_t parentCol = 0, childCol = 1;
                for (; parentCol < alignedWidth; parentCol += A, childCol += DA)
                    SegmentationPropagate2x2<false>(parent0, parent1, parentCol, difference0, difference1,
                        child0, child1, childCol, index, invalid, empty, threshold);
                if (parentCol < width)
                    SegmentationPropagate2x2<true>(parent0, parent1, parentCol, difference0, difference1,
                        child0, child1, childCol, index, invalid, empty, threshold, tailMask);
            }
        }

        SIMD_INLINE bool RowHasIndex(const uint8_t * mask, size_t alignedSize, size_t fullSize, __m512i index, __mmask64 tail)
        {
            size_t col = 0;
            for (; col < alignedSize; col += A)
            {
                if (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512(mask + col), index))
                    return true;
            }
            if (col < fullSize)
            {
                if (_mm512_cmpeq_epi8_mask(_mm512_maskz_loadu_epi8(tail, mask + col), index))
                    return true;
            }
            return false;
        }

        template<bool masked> SIMD_INLINE void ColsHasIndex(const uint8_t * mask, size_t stride, size_t size, __m512i index, __mmask64 & cols, __mmask64 tail = -1)
        {
            for (size_t row = 0; row < size; ++row)
            {
                cols = cols | _mm512_cmpeq_epi8_mask(((masked ? _mm512_maskz_loadu_epi8(tail, mask) : _mm512_loadu_si512(mask))), index);
                mask += stride;
            }
        }

        void SegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
        {
            assert(*left >= 0 && *right <= (ptrdiff_t)width && *top >= 0 && *bottom <= (ptrdiff_t)height);

            size_t fullWidth = *right - *left;
            ptrdiff_t alignedWidth = Simd::AlignLo(fullWidth, A);
            ptrdiff_t alignedRight = *left + alignedWidth;
            __mmask64 tailMask = TailMask64(fullWidth - alignedWidth);
            ptrdiff_t alignedLeft = *right - alignedWidth;
            __mmask64 noseMask = NoseMask64(fullWidth - alignedWidth);

            __m512i _index = _mm512_set1_epi8(index);
            bool search = true;
            for (ptrdiff_t row = *top; search && row < *bottom; ++row)
            {
                if (RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index, tailMask))
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

            for (ptrdiff_t row = *bottom - 1; row >= *top; --row)
            {
                if (RowHasIndex(mask + row*stride + *left, alignedWidth, fullWidth, _index, tailMask))
                {
                    *bottom = row + 1;
                    break;
                }
            }

            for (ptrdiff_t col = *left; col < *right; col += A)
            {
                __mmask64 cols = 0;
                if (col < alignedRight)
                    ColsHasIndex<false>(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols);
                else
                    ColsHasIndex<true>(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols, tailMask);
                if (cols)
                {
                    *left = col + FirstNotZero64(cols);
                    break;
                }
            }

            for (ptrdiff_t col = *right - A; col >= *left; col -= A)
            {
                __mmask64 cols = 0;
                if (col >= alignedLeft)
                    ColsHasIndex<false>(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols);
                else
                    ColsHasIndex<true>(mask + (*top)*stride + col, stride, *bottom - *top, _index, cols, noseMask);
                if (cols)
                {
                    *right = col + LastNotZero64(cols);
                    break;
                }
            }
        }
    }
#endif//SIMD_AVX512BW_ENABLE
}
