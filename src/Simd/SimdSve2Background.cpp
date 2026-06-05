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
        SIMD_INLINE void BackgroundIncrementCount(const uint8_t* value, const uint8_t* loValue, const uint8_t* hiValue,
            uint8_t* loCount, uint8_t* hiCount, const svuint8_t& _1, const svbool_t& mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _loValue = svld1_u8(mask, loValue);
            svuint8_t _hiValue = svld1_u8(mask, hiValue);
            svuint8_t _loCount = svld1_u8(mask, loCount);
            svuint8_t _hiCount = svld1_u8(mask, hiCount);

            svbool_t incLo = svcmplt_u8(mask, _value, _loValue);
            svbool_t incHi = svcmpgt_u8(mask, _value, _hiValue);

            svst1_u8(mask, loCount, svqadd_u8(_loCount, svand_u8_z(incLo, _1, _1)));
            svst1_u8(mask, hiCount, svqadd_u8(_hiCount, svand_u8_z(incHi, _1, _1)));
        }

        void BackgroundIncrementCount(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* loValue, size_t loValueStride, const uint8_t* hiValue, size_t hiValueStride,
            uint8_t* loCount, size_t loCountStride, uint8_t* hiCount, size_t hiCountStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundIncrementCount(value + col, loValue + col, hiValue + col, loCount + col, hiCount + col, _1, body);
                if (widthA < width)
                    BackgroundIncrementCount(value + col, loValue + col, hiValue + col, loCount + col, hiCount + col, _1, tail);
                value += valueStride;
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }
    }
#endif
}
