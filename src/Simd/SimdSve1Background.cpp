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

namespace Simd
{
#ifdef SIMD_SVE_ENABLE    
    namespace Sve
    {
        SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, const svuint8_t& _1, const svbool_t & mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _lo = svld1_u8(mask, lo);
            svuint8_t _hi = svld1_u8(mask, hi);

            svbool_t inc = svcmpgt_u8(mask, _value, _hi);
            svbool_t dec = svcmplt_u8(mask, _value, _lo);

            svst1_u8(mask, lo, svqsub_u8(_lo, svand_u8_z(dec, _1, _1)));
            svst1_u8(mask, hi, svqadd_u8(_hi, svand_u8_z(inc, _1, _1)));
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundGrowRangeSlow(value + col, lo + col, hi + col, _1, body);
                if (widthA < width)
                    BackgroundGrowRangeSlow(value + col, lo + col, hi + col, _1, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t* value, uint8_t* lo, uint8_t* hi, const svbool_t& mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _lo = svld1_u8(mask, lo);
            svuint8_t _hi = svld1_u8(mask, hi);

            svst1_u8(mask, lo, svmin_u8_x(mask, _lo, _value));
            svst1_u8(mask, hi, svmax_u8_x(mask, _hi, _value));
        }

        void BackgroundGrowRangeFast(const uint8_t* value, size_t valueStride, size_t width, size_t height, uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundGrowRangeFast(value + col, lo + col, hi + col, body);
                if (widthA < width)
                    BackgroundGrowRangeFast(value + col, lo + col, hi + col, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }
    }
#endif
}
