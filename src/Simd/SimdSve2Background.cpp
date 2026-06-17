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
        SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t* value, uint8_t* lo, uint8_t* hi,
            const svuint8_t& _1, const svbool_t& mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _lo = svld1_u8(mask, lo);
            svuint8_t _hi = svld1_u8(mask, hi);

            svbool_t inc = svcmpgt_u8(mask, _value, _hi);
            svbool_t dec = svcmplt_u8(mask, _value, _lo);

            svst1_u8(mask, lo, svqsub_u8(_lo, svand_u8_z(dec, _1, _1)));
            svst1_u8(mask, hi, svqadd_u8(_hi, svand_u8_z(inc, _1, _1)));
        }

        void BackgroundGrowRangeSlow(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride)
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
                    BackgroundGrowRangeSlow(value + col, lo + col, hi + col, _1, body);
                if (widthA < width)
                    BackgroundGrowRangeSlow(value + col, lo + col, hi + col, _1, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t* value, uint8_t* lo, uint8_t* hi, const svbool_t& mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _lo = svld1_u8(mask, lo);
            svuint8_t _hi = svld1_u8(mask, hi);

            svst1_u8(mask, lo, svmin_u8_x(mask, _lo, _value));
            svst1_u8(mask, hi, svmax_u8_x(mask, _hi, _value));
        }

        void BackgroundGrowRangeFast(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
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

        //-------------------------------------------------------------------------------------------------

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
      
        SIMD_INLINE svuint8_t AdjustLo(const svbool_t& mask, const svuint8_t& count, const svuint8_t& value, const svuint8_t& threshold, const svuint8_t& _1)
        {
            svuint8_t dec = svand_u8_z(svcmpgt_u8(mask, count, threshold), _1, _1);
            svuint8_t inc = svand_u8_z(svcmplt_u8(mask, count, threshold), _1, _1);
            return svqsub_u8(svqadd_u8(value, inc), dec);
        }

        SIMD_INLINE svuint8_t AdjustHi(const svbool_t& mask, const svuint8_t& count, const svuint8_t& value, const svuint8_t& threshold, const svuint8_t& _1)
        {
            svuint8_t inc = svand_u8_z(svcmpgt_u8(mask, count, threshold), _1, _1);
            svuint8_t dec = svand_u8_z(svcmplt_u8(mask, count, threshold), _1, _1);
            return svqsub_u8(svqadd_u8(value, inc), dec);
        }

        SIMD_INLINE void BackgroundAdjustRange(uint8_t* loCount, uint8_t* loValue, uint8_t* hiCount, uint8_t* hiValue,
            const svuint8_t& threshold, const svuint8_t& _1, const svuint8_t& _0, const svbool_t& mask)
        {
            svuint8_t _loCount = svld1_u8(mask, loCount);
            svuint8_t _loValue = svld1_u8(mask, loValue);
            svuint8_t _hiCount = svld1_u8(mask, hiCount);
            svuint8_t _hiValue = svld1_u8(mask, hiValue);

            svst1_u8(mask, loValue, AdjustLo(mask, _loCount, _loValue, threshold, _1));
            svst1_u8(mask, hiValue, AdjustHi(mask, _hiCount, _hiValue, threshold, _1));
            svst1_u8(mask, loCount, _0);
            svst1_u8(mask, hiCount, _0);
        }

        void BackgroundAdjustRange(uint8_t* loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t* loValue, size_t loValueStride, uint8_t* hiCount, size_t hiCountStride,
            uint8_t* hiValue, size_t hiValueStride, uint8_t threshold)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _threshold = svdup_n_u8(threshold);
            svuint8_t _1 = svdup_n_u8(1);
            svuint8_t _0 = svdup_n_u8(0);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundAdjustRange(loCount + col, loValue + col, hiCount + col, hiValue + col, _threshold, _1, _0, body);
                if (widthA < width)
                    BackgroundAdjustRange(loCount + col, loValue + col, hiCount + col, hiValue + col, _threshold, _1, _0, tail);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }
      
        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundAdjustRangeMasked(uint8_t* loCount, uint8_t* loValue, uint8_t* hiCount, uint8_t* hiValue,
            const uint8_t* mask, const svuint8_t& threshold, const svuint8_t& _1, const svuint8_t& _0, const svbool_t& tail)
        {
            svuint8_t _mask = svld1_u8(tail, mask);
            svbool_t adjust = svcmpne_u8(tail, _mask, _0);
            svuint8_t _loCount = svld1_u8(tail, loCount);
            svuint8_t _loValue = svld1_u8(tail, loValue);
            svuint8_t _hiCount = svld1_u8(tail, hiCount);
            svuint8_t _hiValue = svld1_u8(tail, hiValue);

            svst1_u8(tail, loValue, AdjustLo(adjust, _loCount, _loValue, threshold, _1));
            svst1_u8(tail, hiValue, AdjustHi(adjust, _hiCount, _hiValue, threshold, _1));
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
      
        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundShiftRange(const uint8_t* value, uint8_t* lo, uint8_t* hi, const svbool_t& mask)
        {
            svuint8_t _value = svld1_u8(mask, value);
            svuint8_t _lo = svld1_u8(mask, lo);
            svuint8_t _hi = svld1_u8(mask, hi);

            svuint8_t add = svqsub_u8(_value, _hi);
            svuint8_t sub = svqsub_u8(_lo, _value);

            svst1_u8(mask, lo, svqsub_u8(svqadd_u8(_lo, add), sub));
            svst1_u8(mask, hi, svqsub_u8(svqadd_u8(_hi, add), sub));
        }

        void BackgroundShiftRange(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundShiftRange(value + col, lo + col, hi + col, body);
                if (widthA < width)
                    BackgroundShiftRange(value + col, lo + col, hi + col, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundShiftRangeMasked(const uint8_t* value, uint8_t* lo, uint8_t* hi,
            const uint8_t* mask, const svuint8_t& _0, const svbool_t& tail)
        {
            svuint8_t _mask = svld1_u8(tail, mask);
            svbool_t shift = svcmpne_u8(tail, _mask, _0);
            svuint8_t _value = svld1_u8(shift, value);
            svuint8_t _lo = svld1_u8(shift, lo);
            svuint8_t _hi = svld1_u8(shift, hi);

            svuint8_t add = svqsub_u8(_value, _hi);
            svuint8_t sub = svqsub_u8(_lo, _value);

            svst1_u8(shift, lo, svqsub_u8(svqadd_u8(_lo, add), sub));
            svst1_u8(shift, hi, svqsub_u8(svqadd_u8(_hi, add), sub));
        }

        void BackgroundShiftRangeMasked(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride, const uint8_t* mask, size_t maskStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _0 = svdup_n_u8(0);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundShiftRangeMasked(value + col, lo + col, hi + col, mask + col, _0, body);
                if (widthA < width)
                    BackgroundShiftRangeMasked(value + col, lo + col, hi + col, mask + col, _0, tail);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                mask += maskStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BackgroundInitMask(const uint8_t* src, uint8_t* dst, const svuint8_t& index,
            const svuint8_t& value, const svbool_t& tail)
        {
            svuint8_t _src = svld1_u8(tail, src);
            svst1_u8(svcmpeq_u8(tail, _src, index), dst, value);
        }

        void BackgroundInitMask(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _index = svdup_n_u8(index), _value = svdup_n_u8(value);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BackgroundInitMask(src + col, dst + col, _index, _value, body);
                if (widthA < width)
                    BackgroundInitMask(src + col, dst + col, _index, _value, tail);
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
