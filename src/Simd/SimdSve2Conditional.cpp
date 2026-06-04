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
#include "Simd/SimdExtract.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        template <SimdCompareType compareType> SIMD_INLINE
        void ConditionalCount8u(const uint8_t* src, const svbool_t& mask, const svuint8_t& value, svuint8_t _1, svuint32_t& count)
        {
            svuint8_t _src = svld1_u8(mask, src);
            svbool_t cond = Compare8u<compareType>(mask, _src, value);
            svuint8_t ones = svand_u8_z(cond, _1, _1);
            count = svdot_u32(count, ones, ones);
        }

        template <SimdCompareType compareType>
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1), _value = svdup_n_u8(value);
            count[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _count = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    ConditionalCount8u<compareType>(src + col, body, _value, _1, _count);
                if (widthA < width)
                    ConditionalCount8u<compareType>(src + col, tail, _value, _1, _count);
                count[0] += svaddv_u32(svptrue_b32(), _count);
                src += stride;
            }
        }

        void ConditionalCount8u(const uint8_t* src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t* count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount8u<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount8u<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount8u<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount8u<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount8u<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount8u<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        template <SimdCompareType compareType> SIMD_INLINE
            void ConditionalCount16i(const int16_t* src, const svbool_t& mask, const svint16_t& value, svint16_t _1, svuint32_t& count)
        {
            svint16_t _src = svld1_s16(mask, src);
            svbool_t cond = Compare16i<compareType>(mask, _src, value);
            svuint8_t ones = svreinterpret_u8_s16(svand_s16_z(cond, _1, _1));
            count = svdot_u32(count, ones, ones);
        }

        template <SimdCompareType compareType>
        void ConditionalCount16i(const uint8_t* src, size_t stride, size_t width, size_t height, int16_t value, uint32_t* count)
        {
            size_t A = svlen(svint16_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b16();
            const svbool_t tail = svwhilelt_b16(widthA, width);
            svint16_t _1 = svdup_n_s16(1), _value = svdup_n_s16(value);
            count[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _count = svdup_n_u32(0);
                const int16_t* s = (const int16_t*)src;
                for (; col < widthA; col += A)
                    ConditionalCount16i<compareType>(s + col, body, _value, _1, _count);
                if (widthA < width)
                    ConditionalCount16i<compareType>(s + col, tail, _value, _1, _count);
                count[0] += svaddv_u32(svptrue_b32(), _count);
                src += stride;
            }
        }

        void ConditionalCount16i(const uint8_t* src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t* count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount16i<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount16i<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount16i<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount16i<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount16i<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount16i<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        template <SimdCompareType compareType> SIMD_INLINE
            void ConditionalSum(const uint8_t* src, const uint8_t* msk, const svbool_t& mask, const svuint8_t& value, svuint8_t _1, svuint32_t& sum)
        {
            svuint8_t _msk = svld1_u8(mask, msk);
            svbool_t cond = Compare8u<compareType>(mask, _msk, value);
            svuint8_t _src = svld1_u8(cond, src);
            sum = svdot_u32(sum, _src, _1);
        }

        template <SimdCompareType compareType>
        void ConditionalSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint64_t* sum)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _1 = svdup_n_u8(1), _value = svdup_n_u8(value);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                svuint32_t _sum = svdup_n_u32(0);
                size_t col = 0;
                for (; col < widthA; col += A)
                    ConditionalSum<compareType>(src + col, mask + col, body, _value, _1, _sum);
                if (widthA < width)
                    ConditionalSum<compareType>(src + col, mask + col, tail, _value, _1, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, 
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        template <SimdCompareType compareType> SIMD_INLINE
            void ConditionalSquareSum(const uint8_t* src, const uint8_t* msk, const svbool_t& mask, const svuint8_t& value, svuint32_t& sum)
        {
            svuint8_t _msk = svld1_u8(mask, msk);
            svbool_t cond = Compare8u<compareType>(mask, _msk, value);
            svuint8_t _src = svld1_u8(cond, src);
            sum = svdot_u32(sum, _src, _src);
        }

        template <SimdCompareType compareType>
        void ConditionalSquareSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint64_t* sum)
        {
            assert(width <= 256 * 256);

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _value = svdup_n_u8(value);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                svuint32_t _sum = svdup_n_u32(0);
                size_t col = 0;
                for (; col < widthA; col += A)
                    ConditionalSquareSum<compareType>(src + col, mask + col, body, _value, _sum);
                if (widthA < width)
                    ConditionalSquareSum<compareType>(src + col, mask + col, tail, _value, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSquareSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        template <SimdCompareType compareType> SIMD_INLINE
            void ConditionalSquareGradientSum(const uint8_t* src, size_t stride, const uint8_t* msk, const svbool_t& mask, const svuint8_t& value, svuint32_t& sum)
        {
            svuint8_t _msk = svld1_u8(mask, msk);
            svbool_t cond = Compare8u<compareType>(mask, _msk, value);
            svuint8_t dx = svabd_u8_z(cond, svld1_u8(mask, src - 1), svld1_u8(mask, src + 1));
            svuint8_t dy = svabd_u8_z(cond, svld1_u8(mask, src - stride), svld1_u8(mask, src + stride));
            sum = svdot_u32(svdot_u32(sum, dx, dx), dy, dy);
        }

        template <SimdCompareType compareType>
        void ConditionalSquareGradientSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint64_t* sum)
        {
            assert(width >= 3 && height >= 3);
            assert(width <= 256 * 128);

            src += srcStride;
            mask += maskStride;
            width -= 1;
            height -= 2;

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            svbool_t nose = widthA ? svnot_b_z(svptrue_b8(), svwhilelt_b8(0, 1)) : 
                svand_b_z(svptrue_b8(), svnot_b_z(svptrue_b8(), svwhilege_b8(0, 1)), svwhilelt_b8(widthA, width));
            svbool_t body = svptrue_b8();
            svbool_t tail = svwhilelt_b8(widthA, width);
            svuint8_t _value = svdup_n_u8(value);
            sum[0] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                svuint32_t _sum = svdup_n_u32(0);
                size_t col = 0;
                ConditionalSquareGradientSum<compareType>(src + col, srcStride, mask + col, nose, _value, _sum), col += A;
                for (; col < widthA; col += A)
                    ConditionalSquareGradientSum<compareType>(src + col, srcStride, mask + col, body, _value, _sum);
                if (col < width)
                    ConditionalSquareGradientSum<compareType>(src + col, srcStride, mask + col, tail, _value, _sum);
                sum[0] += svaddv_u32(svptrue_b32(), _sum);
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSquareGradientSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareGradientSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareGradientSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareGradientSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareGradientSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareGradientSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareGradientSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }
    }
#endif
}
