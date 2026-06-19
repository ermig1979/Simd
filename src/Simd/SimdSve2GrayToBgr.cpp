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
        SIMD_INLINE bool InitGrayToBgrIndex(uint8_t index[3][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t k = 0; k < 3; ++k)
            {
                size_t offset = k * A;
                for (size_t i = 0; i < A; ++i)
                    index[k][i] = (uint8_t)((offset + i) / 3);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t GRAY_TO_BGR_INDEX[3][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool GRAY_TO_BGR_INDEX_INITED = InitGrayToBgrIndex(GRAY_TO_BGR_INDEX);

        SIMD_INLINE void GrayToBgr(const uint8_t* gray, uint8_t* bgr, size_t A, const svuint8_t& index0,
            const svuint8_t& index1, const svuint8_t& index2, const svbool_t& mask)
        {
            svuint8_t _gray = svld1_u8(mask, gray);
            svst1_u8(mask, bgr + 0 * A, svtbl_u8(_gray, index0));
            svst1_u8(mask, bgr + 1 * A, svtbl_u8(_gray, index1));
            svst1_u8(mask, bgr + 2 * A, svtbl_u8(_gray, index2));
        }

        SIMD_INLINE void GrayToBgrTail(const uint8_t* gray, uint8_t* bgr, const svbool_t& mask)
        {
            svuint8_t _gray = svld1_u8(mask, gray);
            svst3_u8(mask, bgr, svcreate3_u8(_gray, _gray, _gray));
        }

        void GrayToBgr(const uint8_t* gray, size_t width, size_t height, size_t grayStride, uint8_t* bgr, size_t bgrStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t index0 = svld1_u8(body, GRAY_TO_BGR_INDEX[0]);
            const svuint8_t index1 = svld1_u8(body, GRAY_TO_BGR_INDEX[1]);
            const svuint8_t index2 = svld1_u8(body, GRAY_TO_BGR_INDEX[2]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    GrayToBgr(gray + col, bgr + offset, A, index0, index1, index2, body);
                if (widthA < width)
                    GrayToBgrTail(gray + col, bgr + offset, tail);
                gray += grayStride;
                bgr += bgrStride;
            }
        }
    }
#endif
}
