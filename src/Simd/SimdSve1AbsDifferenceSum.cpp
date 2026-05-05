/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2022-2022 Fabien Spindler,
*               2022-2022 Souriya Trinh.
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
#include "Simd/SimdSve1.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE
    namespace Sve
    {
        SIMD_INLINE void AbsDifferenceSum(const uint8_t* a, const uint8_t* b, const svuint8_t& _1, const svbool_t & mask, svuint32_t & sum)
        {
            svuint8_t _a = svld1_u8(mask, a);
            svuint8_t _b = svld1_u8(mask, b);
            svuint8_t abd = svabd_x(mask, _a, _b);
            sum = svdot_u32(sum, abd, _1);
        }

        void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    AbsDifferenceSum(a + col, b + col, _1, body, _sum);
                if (widthA < width)
                    AbsDifferenceSum(a + col, b + col, _1, tail, _sum);
                *sum += svaddv_u32(svptrue_b32(), _sum);
                a += aStride;
                b += bStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void AbsDifferenceSumMasked(const uint8_t* a, const uint8_t* b, const uint8_t* m, const svuint8_t& _1, const svuint8_t& index, const svbool_t& mask, svuint32_t& sum)
        {
            svuint8_t _a = svld1_u8(mask, a);
            svuint8_t _b = svld1_u8(mask, b);
            svuint8_t _m = svld1_u8(mask, m);
            svbool_t _mask = svcmpeq_u8(mask, _m, index);
            svuint8_t abd = svabd_z(_mask, _a, _b);
            sum = svdot_u32(sum, abd, _1);
        }

        void AbsDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            const uint8_t* mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t* sum)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _index = svdup_n_u8(index), _1 = svdup_n_u8(1);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    AbsDifferenceSumMasked(a + col, b + col, mask + col, _1, _index, body, _sum);
                if (widthA < width)
                    AbsDifferenceSumMasked(a + col, b + col, mask + col, _1, _index, tail, _sum);
                *sum += svaddv_u32(svptrue_b32(), _sum);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void AbsDifferenceSums3(const svuint8_t& current, const uint8_t* background, const svuint8_t& _1, const svbool_t& mask, 
            svuint32_t &sum0, svuint32_t& sum1, svuint32_t& sum2)
        {
            sum0 = svdot_u32(sum0, svabd_x(mask, current, svld1_u8(mask, background - 1)), _1);
            sum1 = svdot_u32(sum1, svabd_x(mask, current, svld1_u8(mask, background)), _1);
            sum2 = svdot_u32(sum2, svabd_x(mask, current, svld1_u8(mask, background + 1)), _1);
        }

        SIMD_INLINE void AbsDifferenceSums3x3(const uint8_t* current, const uint8_t* background, size_t stride, const svuint8_t& _1, const svbool_t& mask, 
            svuint32_t& sum0, svuint32_t& sum1, svuint32_t& sum2, svuint32_t& sum3, svuint32_t& sum4, svuint32_t& sum5, svuint32_t& sum6, svuint32_t& sum7, svuint32_t& sum8)
        {
            svuint8_t _current = svld1_u8(mask, current);
            AbsDifferenceSums3(_current, background - stride, _1, mask, sum0, sum1, sum2);
            AbsDifferenceSums3(_current, background, _1, mask, sum3, sum4, sum5);
            AbsDifferenceSums3(_current, background + stride, _1, mask, sum6, sum7, sum8);
        }

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

        void AbsDifferenceSums3x3(const uint8_t* current, size_t currentStride, const uint8_t* background, size_t backgroundStride, size_t width, size_t height, uint64_t* sums)
        {
            assert(height > 2 && width > 2);

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);

            for (size_t i = 0; i < 9; ++i)
                sums[i] = 0;
            svuint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8;
            for (size_t row = 0; row < height; ++row)
            {
                ClearSums(s0, s1, s3);
                ClearSums(s3, s4, s5);
                ClearSums(s6, s7, s8);
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsDifferenceSums3x3(current + col, background + col, backgroundStride, _1, body, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                if (widthA < width)
                    AbsDifferenceSums3x3(current + col, background + col, backgroundStride, _1, tail, s0, s1, s2, s3, s4, s5, s6, s7, s8);
                AddSums(s0, s1, s2, sums + 0);
                AddSums(s3, s4, s5, sums + 3);
                AddSums(s6, s7, s8, sums + 6);
                current += currentStride;
                background += backgroundStride;
            }
        }
    }
#endif
}
