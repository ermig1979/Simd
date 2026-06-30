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
        SIMD_INLINE void SquaredDifferenceSum(const uint8_t* a, const uint8_t* b, const svbool_t& mask, const svuint8_t& zero, svuint32_t& sum)
        {
            svuint8_t diff = svabd_u8_x(mask, svld1_u8(mask, a), svld1_u8(mask, b));
            svuint8_t masked = svsel_u8(mask, diff, zero);
            sum = svdot_u32(sum, masked, masked);
        }

        SIMD_INLINE void SquaredDifferenceSumMasked(const uint8_t* a, const uint8_t* b, const uint8_t* m, const svbool_t& pg, const svuint8_t& index, const svuint8_t& zero, svuint32_t& sum)
        {
            svuint8_t diff = svabd_u8_x(pg, svld1_u8(pg, a), svld1_u8(pg, b));
            svuint8_t mask = svld1_u8(pg, m);
            svbool_t eq = svcmpeq_u8(pg, mask, index);
            svuint8_t selected = svsel_u8(eq, diff, zero);
            sum = svdot_u32(sum, selected, selected);
        }

        void SquaredDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            assert(width < 0x10000);

            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t zero = svdup_n_u8(0);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t rowSum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    SquaredDifferenceSum(a + col, b + col, body, zero, rowSum);
                if (widthA < width)
                    SquaredDifferenceSum(a + col, b + col, tail, zero, rowSum);
                *sum += svaddv_u32(svptrue_b32(), rowSum);
                a += aStride;
                b += bStride;
            }
        }

        void SquaredDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            const uint8_t* mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t* sum)
        {
            assert(width < 0x10000);

            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _index = svdup_n_u8(index);
            const svuint8_t zero = svdup_n_u8(0);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t rowSum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    SquaredDifferenceSumMasked(a + col, b + col, mask + col, body, _index, zero, rowSum);
                if (widthA < width)
                    SquaredDifferenceSumMasked(a + col, b + col, mask + col, tail, _index, zero, rowSum);
                *sum += svaddv_u32(svptrue_b32(), rowSum);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }
    }
#endif
}
