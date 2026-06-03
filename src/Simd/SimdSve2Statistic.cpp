/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar,
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

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            switch (channels)
            {
            case 1: ValueSquareSum(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums2(src, stride, width, height, valueSums, squareSums); break;
            //case 3: ValueSquareSums3(src, stride, width, height, valueSums, squareSums); break;
            //case 4: ValueSquareSums4(src, stride, width, height, valueSums, squareSums); break;
            default:
                Neon::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
                //assert(0);
            }
        }
    }
#endif
}
