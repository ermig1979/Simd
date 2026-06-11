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
        SIMD_INLINE svuint8_t FeatureDifference(const svuint8_t& value, const svuint8_t& lo, const svuint8_t& hi)
        {
            return svmax_u8_x(svptrue_b8(), svqsub_u8(value, hi), svqsub_u8(lo, value));
        }

        SIMD_INLINE svuint8_t ShiftedWeightedSquare(const svuint8_t& difference, const svuint16_t& weight)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t lo = svmovlb_u16(difference);
            svuint16_t hi = svmovlt_u16(difference);
            lo = svmulh_u16_x(mask, svmul_u16_x(mask, lo, lo), weight);
            hi = svmulh_u16_x(mask, svmul_u16_x(mask, hi, hi), weight);
            return svqxtnt_u16(svqxtnb_u16(lo), hi);
        }

        SIMD_INLINE void AddFeatureDifference(const uint8_t* value, const uint8_t* lo, const uint8_t* hi,
            uint8_t* difference, const svuint16_t& weight, const svbool_t& mask)
        {
            const svuint8_t _value = svld1_u8(mask, value);
            const svuint8_t _lo = svld1_u8(mask, lo);
            const svuint8_t _hi = svld1_u8(mask, hi);
            const svuint8_t _difference = svld1_u8(mask, difference);

            const svuint8_t featureDifference = FeatureDifference(_value, _lo, _hi);
            const svuint8_t inc = ShiftedWeightedSquare(featureDifference, weight);
            svst1_u8(mask, difference, svqadd_u8(_difference, inc));
        }

        void AddFeatureDifference(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride,
            uint16_t weight, uint8_t* difference, size_t differenceStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint16_t _weight = svdup_n_u16(weight);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    AddFeatureDifference(value + col, lo + col, hi + col, difference + col, _weight, body);
                if (widthA < width)
                    AddFeatureDifference(value + col, lo + col, hi + col, difference + col, _weight, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                difference += differenceStride;
            }
        }
    }
#endif
}
