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
        SIMD_INLINE void AbsDifferenceSum(const uint8_t* a, const uint8_t* b, const svbool_t& mask, const svuint8_t& _1, const svuint8_t& zero, svuint32_t& sum)
        {
            svuint8_t diff = svabd_u8_x(mask, svld1_u8(mask, a), svld1_u8(mask, b));
            sum = svdot_u32(sum, svsel_u8(mask, diff, zero), _1);
        }

        void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum)
        {
            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _1 = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t rowSum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    AbsDifferenceSum(a + col, b + col, body, _1, zero, rowSum);
                if (widthA < width)
                    AbsDifferenceSum(a + col, b + col, tail, _1, zero, rowSum);
                *sum += svaddv_u32(svptrue_b32(), rowSum);
                a += aStride;
                b += bStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void AbsDifferenceSumMasked(const uint8_t* a, const uint8_t* b, const uint8_t* m, const svbool_t& mask,
            const svuint8_t& index, const svuint8_t& _1, const svuint8_t& zero, svuint32_t& sum)
        {
            svuint8_t diff = svabd_u8_x(mask, svld1_u8(mask, a), svld1_u8(mask, b));
            svbool_t equal = svcmpeq_u8(mask, svld1_u8(mask, m), index);
            sum = svdot_u32(sum, svsel_u8(equal, diff, zero), _1);
        }

        void AbsDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            const uint8_t* mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t* sum)
        {
            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _index = svdup_n_u8(index);
            const svuint8_t _1 = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t rowSum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    AbsDifferenceSumMasked(a + col, b + col, mask + col, body, _index, _1, zero, rowSum);
                if (widthA < width)
                    AbsDifferenceSumMasked(a + col, b + col, mask + col, tail, _index, _1, zero, rowSum);
                *sum += svaddv_u32(svptrue_b32(), rowSum);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void ClearSums(svuint32_t& sum0, svuint32_t& sum1, svuint32_t& sum2)
        {
            sum0 = svdup_n_u32(0);
            sum1 = svdup_n_u32(0);
            sum2 = svdup_n_u32(0);
        }

        SIMD_INLINE void AddSums(const svuint32_t& sum0, const svuint32_t& sum1, const svuint32_t& sum2, uint64_t* sums)
        {
            sums[0] += svaddv_u32(svptrue_b32(), sum0);
            sums[1] += svaddv_u32(svptrue_b32(), sum1);
            sums[2] += svaddv_u32(svptrue_b32(), sum2);
        }

        SIMD_INLINE void AbsDifferenceSums3(const svuint8_t& current, const uint8_t* background, const svbool_t& mask, const svuint8_t& _1,
            const svuint8_t& zero, svuint32_t& sum0, svuint32_t& sum1, svuint32_t& sum2)
        {
            svuint8_t diff0 = svabd_u8_x(mask, current, svld1_u8(mask, background - 1));
            svuint8_t diff1 = svabd_u8_x(mask, current, svld1_u8(mask, background));
            svuint8_t diff2 = svabd_u8_x(mask, current, svld1_u8(mask, background + 1));
            sum0 = svdot_u32(sum0, svsel_u8(mask, diff0, zero), _1);
            sum1 = svdot_u32(sum1, svsel_u8(mask, diff1, zero), _1);
            sum2 = svdot_u32(sum2, svsel_u8(mask, diff2, zero), _1);
        }

        SIMD_INLINE void AbsDifferenceSums3x3(const uint8_t* current, const uint8_t* background, size_t stride, const svbool_t& mask,
            const svuint8_t& _1, const svuint8_t& zero, svuint32_t& s0, svuint32_t& s1, svuint32_t& s2,
            svuint32_t& s3, svuint32_t& s4, svuint32_t& s5, svuint32_t& s6, svuint32_t& s7, svuint32_t& s8)
        {
            svuint8_t _current = svld1_u8(mask, current);
            AbsDifferenceSums3(_current, background - stride, mask, _1, zero, s0, s1, s2);
            AbsDifferenceSums3(_current, background, mask, _1, zero, s3, s4, s5);
            AbsDifferenceSums3(_current, background + stride, mask, _1, zero, s6, s7, s8);
        }

        SIMD_INLINE void AbsDifferenceSums3x2(const svuint8_t& c0, const svuint8_t& c1, const uint8_t* b, const svbool_t& mask,
            const svuint8_t& _1, const svuint8_t& zero, svuint32_t& s0, svuint32_t& s1, svuint32_t& s2,
            svuint32_t& s3, svuint32_t& s4, svuint32_t& s5)
        {
            svuint8_t b0 = svld1_u8(mask, b - 1);
            svuint8_t d00 = svabd_u8_x(mask, c0, b0);
            svuint8_t d10 = svabd_u8_x(mask, c1, b0);
            s0 = svdot_u32(s0, svsel_u8(mask, d00, zero), _1);
            s3 = svdot_u32(s3, svsel_u8(mask, d10, zero), _1);
            svuint8_t b1 = svld1_u8(mask, b);
            svuint8_t d01 = svabd_u8_x(mask, c0, b1);
            svuint8_t d11 = svabd_u8_x(mask, c1, b1);
            s1 = svdot_u32(s1, svsel_u8(mask, d01, zero), _1);
            s4 = svdot_u32(s4, svsel_u8(mask, d11, zero), _1);
            svuint8_t b2 = svld1_u8(mask, b + 1);
            svuint8_t d02 = svabd_u8_x(mask, c0, b2);
            svuint8_t d12 = svabd_u8_x(mask, c1, b2);
            s2 = svdot_u32(s2, svsel_u8(mask, d02, zero), _1);
            s5 = svdot_u32(s5, svsel_u8(mask, d12, zero), _1);
        }

        SIMD_INLINE void AbsDifferenceSums3x3x2(const uint8_t* c, size_t cStride, const uint8_t* b, size_t bStride, const svbool_t& mask,
            const svuint8_t& _1, const svuint8_t& zero, svuint32_t& s0, svuint32_t& s1, svuint32_t& s2,
            svuint32_t& s3, svuint32_t& s4, svuint32_t& s5, svuint32_t& s6, svuint32_t& s7, svuint32_t& s8)
        {
            svuint8_t c0 = svld1_u8(mask, c), c1 = svld1_u8(mask, c + cStride);
            AbsDifferenceSums3(c0, b - bStride, mask, _1, zero, s0, s1, s2);
            AbsDifferenceSums3x2(c1, c0, b, mask, _1, zero, s0, s1, s2, s3, s4, s5);
            AbsDifferenceSums3x2(c1, c0, b + bStride, mask, _1, zero, s3, s4, s5, s6, s7, s8);
            AbsDifferenceSums3(c1, b + 2 * bStride, mask, _1, zero, s6, s7, s8);
        }

        void AbsDifferenceSums3x3(const uint8_t* current, size_t currentStride, const uint8_t* background, size_t backgroundStride,
            size_t width, size_t height, uint64_t* sums)
        {
            assert(height > 2 && width > 2);

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const size_t height2 = AlignLo(height, 2);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _1 = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);

            for (size_t i = 0; i < 9; ++i)
                sums[i] = 0;
            svuint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8;
            size_t row = 0;
            for (; row < height2; row += 2)
            {
                ClearSums(s0, s1, s2);
                ClearSums(s3, s4, s5);
                ClearSums(s6, s7, s8);
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsDifferenceSums3x3x2(current + col, currentStride, background + col, backgroundStride, body, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                if (widthA < width)
                    AbsDifferenceSums3x3x2(current + col, currentStride, background + col, backgroundStride, tail, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                AddSums(s0, s1, s2, sums + 0);
                AddSums(s3, s4, s5, sums + 3);
                AddSums(s6, s7, s8, sums + 6);
                current += 2 * currentStride;
                background += 2 * backgroundStride;
            }
            for (; row < height; ++row)
            {
                ClearSums(s0, s1, s2);
                ClearSums(s3, s4, s5);
                ClearSums(s6, s7, s8);
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsDifferenceSums3x3(current + col, background + col, backgroundStride, body, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                if (widthA < width)
                    AbsDifferenceSums3x3(current + col, background + col, backgroundStride, tail, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                AddSums(s0, s1, s2, sums + 0);
                AddSums(s3, s4, s5, sums + 3);
                AddSums(s6, s7, s8, sums + 6);
                current += currentStride;
                background += backgroundStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void AbsDifferenceSums3Masked(const svuint8_t& current, const uint8_t* background, const svbool_t& mask, const svuint8_t& _1,
            const svuint8_t& zero, svuint32_t& sum0, svuint32_t& sum1, svuint32_t& sum2)
        {
            svuint8_t diff0 = svabd_u8_x(mask, current, svld1_u8(mask, background - 1));
            svuint8_t diff1 = svabd_u8_x(mask, current, svld1_u8(mask, background));
            svuint8_t diff2 = svabd_u8_x(mask, current, svld1_u8(mask, background + 1));
            sum0 = svdot_u32(sum0, svsel_u8(mask, diff0, zero), _1);
            sum1 = svdot_u32(sum1, svsel_u8(mask, diff1, zero), _1);
            sum2 = svdot_u32(sum2, svsel_u8(mask, diff2, zero), _1);
        }

        SIMD_INLINE void AbsDifferenceSums3x3Masked(const uint8_t* current, const uint8_t* background, size_t stride, const uint8_t* mask,
            const svbool_t& tail, const svuint8_t& index, const svuint8_t& _1, const svuint8_t& zero,
            svuint32_t& s0, svuint32_t& s1, svuint32_t& s2, svuint32_t& s3, svuint32_t& s4, svuint32_t& s5, svuint32_t& s6, svuint32_t& s7, svuint32_t& s8)
        {
            svbool_t equal = svcmpeq_u8(tail, svld1_u8(tail, mask), index);
            svuint8_t _current = svld1_u8(equal, current);
            AbsDifferenceSums3Masked(_current, background - stride, equal, _1, zero, s0, s1, s2);
            AbsDifferenceSums3Masked(_current, background, equal, _1, zero, s3, s4, s5);
            AbsDifferenceSums3Masked(_current, background + stride, equal, _1, zero, s6, s7, s8);
        }

        void AbsDifferenceSums3x3Masked(const uint8_t* current, size_t currentStride, const uint8_t* background, size_t backgroundStride,
            const uint8_t* mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t* sums)
        {
            assert(height > 2 && width > 2);

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;
            mask += 1 + maskStride;

            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _index = svdup_n_u8(index);
            const svuint8_t _1 = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);

            for (size_t i = 0; i < 9; ++i)
                sums[i] = 0;
            svuint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8;
            for (size_t row = 0; row < height; ++row)
            {
                ClearSums(s0, s1, s2);
                ClearSums(s3, s4, s5);
                ClearSums(s6, s7, s8);
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsDifferenceSums3x3Masked(current + col, background + col, backgroundStride, mask + col, body, _index, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                if (widthA < width)
                    AbsDifferenceSums3x3Masked(current + col, background + col, backgroundStride, mask + col, tail, _index, _1, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                AddSums(s0, s1, s2, sums + 0);
                AddSums(s3, s4, s5, sums + 3);
                AddSums(s6, s7, s8, sums + 6);
                current += currentStride;
                background += backgroundStride;
                mask += maskStride;
            }
        }
    }
#endif
}
