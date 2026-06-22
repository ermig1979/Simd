/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2018-2018 Radchenko Andrey.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * width + sizeof(uint32_t) * width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* sums16;
                uint32_t* sums32;
            private:
                void* _p;
            };
        }

        SIMD_INLINE void GetAbsDxColSums(const uint8_t* src, const svbool_t& mask8, const svbool_t& lo16, const svbool_t& hi16, size_t half, uint16_t* sums)
        {
            svuint8_t diff = svabd_u8_x(mask8, svld1_u8(mask8, src), svld1_u8(mask8, src + 1));
            svst1_u16(lo16, sums, svadd_u16_x(lo16, svld1_u16(lo16, sums), svunpklo_u16(diff)));
            svst1_u16(hi16, sums + half, svadd_u16_x(hi16, svld1_u16(hi16, sums + half), svunpkhi_u16(diff)));
        }

        SIMD_INLINE void AddSums16To32(const uint16_t* src, uint32_t* dst, size_t col, size_t width, size_t half)
        {
            svbool_t mask16 = svwhilelt_b16(col, width);
            svbool_t lo32 = svwhilelt_b32(col, width);
            svbool_t hi32 = svwhilelt_b32(col + half, width);
            svuint16_t sums16 = svld1_u16(mask16, src + col);
            svst1_u32(lo32, dst + col, svadd_u32_x(lo32, svld1_u32(lo32, dst + col), svunpklo_u32(sums16)));
            svst1_u32(hi32, dst + col + half, svadd_u32_x(hi32, svld1_u32(hi32, dst + col + half), svunpkhi_u32(sums16)));
        }

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (width < 2)
            {
                if (width)
                    sums[0] = 0;
                return;
            }

            width--;
            const size_t A = svlen(svuint8_t()), HA = svlen(svuint16_t());
            const size_t widthA = AlignLo(width, A);
            const size_t widthB = AlignHi(width, A);
            const svbool_t body8 = svptrue_b8();
            const svbool_t body16 = svptrue_b16();
            const size_t stepSize = SCHAR_MAX + 1;
            const size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(widthB);
            memset(buffer.sums32, 0, sizeof(uint32_t) * widthB);
            for (size_t step = 0; step < stepCount; ++step)
            {
                const size_t rowStart = step * stepSize;
                const size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t) * widthB);
                const uint8_t* rowSrc = src + rowStart * stride;
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < widthA; col += A)
                        GetAbsDxColSums(rowSrc + col, body8, body16, body16, HA, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSums(rowSrc + col, svwhilelt_b8(col, width), svwhilelt_b16(col, width), svwhilelt_b16(col + HA, width), HA, buffer.sums16 + col);
                    rowSrc += stride;
                }

                for (size_t col = 0; col < width; col += svcntw() * 2)
                    AddSums16To32(buffer.sums16, buffer.sums32, col, width, svcntw());
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t) * width);
            sums[width] = 0;
        }

        SIMD_INLINE void GetAbsDyRowSums(const uint8_t* src0, const uint8_t* src1, svbool_t mask, svuint8_t _1, svuint8_t zero, svuint32_t& sum)
        {
            svuint8_t diff = svabd_u8_x(mask, svld1_u8(mask, src0), svld1_u8(mask, src1));
            sum = svdot_u32(sum, svsel_u8(mask, diff, zero), _1);
        }

        void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            const size_t A = svlen(svuint8_t());
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t _1 = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);

            const uint8_t* src0 = src;
            const uint8_t* src1 = src + stride;
            height--;
            sums[height] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    GetAbsDyRowSums(src0 + col, src1 + col, body, _1, zero, sum);
                if (col < width)
                    GetAbsDyRowSums(src0 + col, src1 + col, tail, _1, zero, sum);
                sums[row] = svaddv_u32(svptrue_b32(), sum);
                src0 += stride;
                src1 += stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void ValueSum(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint32_t& sum)
        {
            svuint8_t val = svld1_u8(mask, src);
            sum = svdot_u32(sum, val, _1);
        }

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    ValueSum(src + col, body, _1, _sum);
                if (widthA < width)
                    ValueSum(src + col, tail, _1, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                src += stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SquareSum(const uint8_t* src, svbool_t mask, svuint32_t& sum)
        {
            svuint8_t val = svld1_u8(mask, src);
            sum = svdot_u32(sum, val, val);
        }

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width <= 256 * 256);

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    SquareSum(src + col, body, _sum);
                if (widthA < width)
                    SquareSum(src + col, tail, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                src += stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void ValueSquareSum(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint32_t& valueSum, svuint32_t& squareSum)
        {
            svuint8_t val = svld1_u8(mask, src);
            valueSum = svdot_u32(valueSum, val, _1);
            squareSum = svdot_u32(squareSum, val, val);
        }

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            valueSum[0] = 0;
            squareSum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _valueSum = svdup_n_u32(0);
                svuint32_t _squareSum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    ValueSquareSum(src + col, body, _1, _valueSum, _squareSum);
                if (widthA < width)
                    ValueSquareSum(src + col, tail, _1, _valueSum, _squareSum);
                valueSum[0] += svaddv_u32(svptrue_b32(), _valueSum);
                squareSum[0] += svaddv_u32(svptrue_b32(), _squareSum);
                src += stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void ValueSquareSums2(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint32_t& valueSum0, svuint32_t& squareSum0,
            svuint32_t& valueSum1, svuint32_t& squareSum1)
        {
            svuint8x2_t val = svld2_u8(mask, src);
            svuint8_t val0 = svget2(val, 0);
            valueSum0 = svdot_u32(valueSum0, val0, _1);
            squareSum0 = svdot_u32(squareSum0, val0, val0);
            svuint8_t val1 = svget2(val, 1);
            valueSum1 = svdot_u32(valueSum1, val1, _1);
            squareSum1 = svdot_u32(squareSum1, val1, val1);
        }

        void ValueSquareSums2(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t A = svlen(svuint8_t()), A2 = A * 2;
            size_t widthA = AlignLo(width, A), size = width * 2, sizeA = widthA * 2;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            valueSums[0] = 0;
            squareSums[0] = 0;
            valueSums[1] = 0;
            squareSums[1] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                svuint32_t _valueSum0 = svdup_n_u32(0);
                svuint32_t _squareSum0 = svdup_n_u32(0);
                svuint32_t _valueSum1 = svdup_n_u32(0);
                svuint32_t _squareSum1 = svdup_n_u32(0);
                for (; offset < sizeA; offset += A2)
                    ValueSquareSums2(src + offset, body, _1, _valueSum0, _squareSum0, _valueSum1, _squareSum1);
                if (sizeA < size)
                    ValueSquareSums2(src + offset, tail, _1, _valueSum0, _squareSum0, _valueSum1, _squareSum1);
                valueSums[0] += svaddv_u32(svptrue_b32(), _valueSum0);
                squareSums[0] += svaddv_u32(svptrue_b32(), _squareSum0);
                valueSums[1] += svaddv_u32(svptrue_b32(), _valueSum1);
                squareSums[1] += svaddv_u32(svptrue_b32(), _squareSum1);
                src += stride;
            }
        }

        SIMD_INLINE void ValueSquareSums3(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint32_t& valueSum0, svuint32_t& squareSum0,
            svuint32_t& valueSum1, svuint32_t& squareSum1, svuint32_t& valueSum2, svuint32_t& squareSum2)
        {
            svuint8x3_t val = svld3_u8(mask, src);
            svuint8_t val0 = svget3(val, 0);
            valueSum0 = svdot_u32(valueSum0, val0, _1);
            squareSum0 = svdot_u32(squareSum0, val0, val0);
            svuint8_t val1 = svget3(val, 1);
            valueSum1 = svdot_u32(valueSum1, val1, _1);
            squareSum1 = svdot_u32(squareSum1, val1, val1);
            svuint8_t val2 = svget3(val, 2);
            valueSum2 = svdot_u32(valueSum2, val2, _1);
            squareSum2 = svdot_u32(squareSum2, val2, val2);
        }

        void ValueSquareSums3(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A), size = width * 3, sizeA = widthA * 3;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            valueSums[0] = 0;
            squareSums[0] = 0;
            valueSums[1] = 0;
            squareSums[1] = 0;
            valueSums[2] = 0;
            squareSums[2] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                svuint32_t _valueSum0 = svdup_n_u32(0);
                svuint32_t _squareSum0 = svdup_n_u32(0);
                svuint32_t _valueSum1 = svdup_n_u32(0);
                svuint32_t _squareSum1 = svdup_n_u32(0);
                svuint32_t _valueSum2 = svdup_n_u32(0);
                svuint32_t _squareSum2 = svdup_n_u32(0);
                for (; offset < sizeA; offset += A3)
                    ValueSquareSums3(src + offset, body, _1, _valueSum0, _squareSum0, _valueSum1, _squareSum1, _valueSum2, _squareSum2);
                if (sizeA < size)
                    ValueSquareSums3(src + offset, tail, _1, _valueSum0, _squareSum0, _valueSum1, _squareSum1, _valueSum2, _squareSum2);
                valueSums[0] += svaddv_u32(svptrue_b32(), _valueSum0);
                squareSums[0] += svaddv_u32(svptrue_b32(), _squareSum0);
                valueSums[1] += svaddv_u32(svptrue_b32(), _valueSum1);
                squareSums[1] += svaddv_u32(svptrue_b32(), _squareSum1);
                valueSums[2] += svaddv_u32(svptrue_b32(), _valueSum2);
                squareSums[2] += svaddv_u32(svptrue_b32(), _squareSum2);
                src += stride;
            }
        }

        SIMD_INLINE void ValueSquareSums4(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint32_t& valueSum0, svuint32_t& squareSum0,
            svuint32_t& valueSum1, svuint32_t& squareSum1, svuint32_t& valueSum2, svuint32_t& squareSum2, svuint32_t& valueSum3, svuint32_t& squareSum3)
        {
            svuint8x4_t val = svld4_u8(mask, src);
            svuint8_t val0 = svget4(val, 0);
            valueSum0 = svdot_u32(valueSum0, val0, _1);
            squareSum0 = svdot_u32(squareSum0, val0, val0);
            svuint8_t val1 = svget4(val, 1);
            valueSum1 = svdot_u32(valueSum1, val1, _1);
            squareSum1 = svdot_u32(squareSum1, val1, val1);
            svuint8_t val2 = svget4(val, 2);
            valueSum2 = svdot_u32(valueSum2, val2, _1);
            squareSum2 = svdot_u32(squareSum2, val2, val2);
            svuint8_t val3 = svget4(val, 3);
            valueSum3 = svdot_u32(valueSum3, val3, _1);
            squareSum3 = svdot_u32(squareSum3, val3, val3);
        }

        void ValueSquareSums4(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A), size = width * 4, sizeA = widthA * 4;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            valueSums[0] = 0;
            squareSums[0] = 0;
            valueSums[1] = 0;
            squareSums[1] = 0;
            valueSums[2] = 0;
            squareSums[2] = 0;
            valueSums[3] = 0;
            squareSums[3] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                svuint32_t _valueSum0 = svdup_n_u32(0);
                svuint32_t _squareSum0 = svdup_n_u32(0);
                svuint32_t _valueSum1 = svdup_n_u32(0);
                svuint32_t _squareSum1 = svdup_n_u32(0);
                svuint32_t _valueSum2 = svdup_n_u32(0);
                svuint32_t _squareSum2 = svdup_n_u32(0);
                svuint32_t _valueSum3 = svdup_n_u32(0);
                svuint32_t _squareSum3 = svdup_n_u32(0);
                for (; offset < sizeA; offset += A4)
                    ValueSquareSums4(src + offset, body, _1, _valueSum0, _squareSum0, 
                        _valueSum1, _squareSum1, _valueSum2, _squareSum2, _valueSum3, _squareSum3);
                if (sizeA < size)
                    ValueSquareSums4(src + offset, tail, _1, _valueSum0, _squareSum0, 
                        _valueSum1, _squareSum1, _valueSum2, _squareSum2, _valueSum3, _squareSum3);
                valueSums[0] += svaddv_u32(svptrue_b32(), _valueSum0);
                squareSums[0] += svaddv_u32(svptrue_b32(), _squareSum0);
                valueSums[1] += svaddv_u32(svptrue_b32(), _valueSum1);
                squareSums[1] += svaddv_u32(svptrue_b32(), _squareSum1);
                valueSums[2] += svaddv_u32(svptrue_b32(), _valueSum2);
                squareSums[2] += svaddv_u32(svptrue_b32(), _squareSum2);
                valueSums[3] += svaddv_u32(svptrue_b32(), _valueSum3);
                squareSums[3] += svaddv_u32(svptrue_b32(), _squareSum3);
                src += stride;
            }
        }

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            switch (channels)
            {
            case 1: ValueSquareSum(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums2(src, stride, width, height, valueSums, squareSums); break;
            case 3: ValueSquareSums3(src, stride, width, height, valueSums, squareSums); break;
            case 4: ValueSquareSums4(src, stride, width, height, valueSums, squareSums); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void CorrelationSum(const uint8_t* a, const uint8_t* b, svbool_t mask, svuint32_t& sum)
        {
            svuint8_t _a = svld1_u8(mask, a);
            svuint8_t _b = svld1_u8(mask, b);
            sum = svdot_u32(sum, _a, _b);
        }

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width <= 256 * 256);

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    CorrelationSum(a + col, b + col, body, _sum);
                if (widthA < width)
                    CorrelationSum(a + col, b + col, tail, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                a += aStride;
                b += bStride;
            }
        }
    }
#endif
}
