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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t AdjustLo(const svuint8_t& count, const svuint8_t& value, const svbool_t& mask, const svuint8_t& threshold, const svuint8_t& _1)
        {
            svbool_t dec = svcmpgt_u8(mask, count, threshold);
            svbool_t inc = svcmplt_u8(mask, count, threshold);
            return svqsub_u8(svqadd_u8(value, svand_u8_z(inc, _1, _1)), svand_u8_z(dec, _1, _1));
        }

        SIMD_INLINE svuint8_t AdjustHi(const svuint8_t& count, const svuint8_t& value, const svbool_t& mask, const svuint8_t& threshold, const svuint8_t& _1)
        {
            svbool_t inc = svcmpgt_u8(mask, count, threshold);
            svbool_t dec = svcmplt_u8(mask, count, threshold);
            return svqsub_u8(svqadd_u8(value, svand_u8_z(inc, _1, _1)), svand_u8_z(dec, _1, _1));
        }

        SIMD_INLINE void BackgroundAdjustRangeMasked(uint8_t* loCount, uint8_t* loValue, uint8_t* hiCount, uint8_t* hiValue,
            const uint8_t* mask, const svuint8_t& threshold, const svuint8_t& _1, const svuint8_t& _0, const svbool_t& tail)
        {
            svuint8_t _mask = svld1_u8(tail, mask);
            svbool_t adjust = svcmpne_u8(tail, _mask, _0);
            svuint8_t _loCount = svld1_u8(tail, loCount);
            svuint8_t _loValue = svld1_u8(tail, loValue);
            svuint8_t _hiCount = svld1_u8(tail, hiCount);
            svuint8_t _hiValue = svld1_u8(tail, hiValue);

            svst1_u8(tail, loValue, AdjustLo(_loCount, _loValue, adjust, threshold, _1));
            svst1_u8(tail, hiValue, AdjustHi(_hiCount, _hiValue, adjust, threshold, _1));
            svst1_u8(tail, loCount, _0);
            svst1_u8(tail, hiCount, _0);
        }

        void BackgroundAdjustRangeMasked(uint8_t* loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t* loValue, size_t loValueStride, uint8_t* hiCount, size_t hiCountStride,
            uint8_t* hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t* mask, size_t maskStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _threshold = svdup_n_u8(threshold), _1 = svdup_n_u8(1), _0 = svdup_n_u8(0);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundAdjustRangeMasked(loCount + col, loValue + col, hiCount + col, hiValue + col, mask + col, _threshold, _1, _0, body);
                if (widthA < width)
                    BackgroundAdjustRangeMasked(loCount + col, loValue + col, hiCount + col, hiValue + col, mask + col, _threshold, _1, _0, tail);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
                mask += maskStride;
            }
        }
    }
#endif
}
