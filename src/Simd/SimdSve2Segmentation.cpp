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
    }
#endif//SIMD_SVE2_ENABLE
}
