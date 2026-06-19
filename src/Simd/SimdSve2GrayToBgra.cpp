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
        SIMD_INLINE bool InitGrayToBgraIndex(uint8_t index[4][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t k = 0; k < 4; ++k)
            {
                size_t offset = k * A;
                for (size_t i = 0; i < A; ++i)
                    index[k][i] = (uint8_t)((offset + i) / 4);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t GRAY_TO_BGRA_INDEX[4][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool GRAY_TO_BGRA_INDEX_INITED = InitGrayToBgraIndex(GRAY_TO_BGRA_INDEX);

        SIMD_INLINE void GrayToBgra(const uint8_t* gray, uint8_t* bgra, size_t A, const svuint8_t& alpha,
            const svuint8_t& index0, const svuint8_t& index1, const svuint8_t& index2, const svuint8_t& index3,
            const svbool_t& alphaMask, const svbool_t& mask)
        {
            svuint8_t _gray = svld1_u8(mask, gray);
            svst1_u8(mask, bgra + 0 * A, svsel_u8(alphaMask, alpha, svtbl_u8(_gray, index0)));
            svst1_u8(mask, bgra + 1 * A, svsel_u8(alphaMask, alpha, svtbl_u8(_gray, index1)));
            svst1_u8(mask, bgra + 2 * A, svsel_u8(alphaMask, alpha, svtbl_u8(_gray, index2)));
            svst1_u8(mask, bgra + 3 * A, svsel_u8(alphaMask, alpha, svtbl_u8(_gray, index3)));
        }

        SIMD_INLINE void GrayToBgraTail(const uint8_t* gray, uint8_t* bgra, const svuint8_t& alpha, const svbool_t& mask)
        {
            svuint8_t _gray = svld1_u8(mask, gray);
            svst4_u8(mask, bgra, svcreate4_u8(_gray, _gray, _gray, alpha));
        }

        void GrayToBgra(const uint8_t* gray, size_t width, size_t height, size_t grayStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            assert((A & 3) == 0);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _alpha = svdup_n_u8(alpha);
            const svbool_t alphaMask = svcmpeq_n_u8(body, svand_n_u8_x(body, svindex_u8(0, 1), 3), 3);
            const svuint8_t index0 = svld1_u8(body, GRAY_TO_BGRA_INDEX[0]);
            const svuint8_t index1 = svld1_u8(body, GRAY_TO_BGRA_INDEX[1]);
            const svuint8_t index2 = svld1_u8(body, GRAY_TO_BGRA_INDEX[2]);
            const svuint8_t index3 = svld1_u8(body, GRAY_TO_BGRA_INDEX[3]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    GrayToBgra(gray + col, bgra + offset, A, _alpha, index0, index1, index2, index3, alphaMask, body);
                if (widthA < width)
                    GrayToBgraTail(gray + col, bgra + offset, _alpha, tail);
                gray += grayStride;
                bgra += bgraStride;
            }
        }
    }
#endif
}
