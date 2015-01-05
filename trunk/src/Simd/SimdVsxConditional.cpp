/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool align, SimdCompareType compareType> 
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);

            v128_u8 _value = SIMD_VEC_SET1_EPI8(value);
            v128_u32 _count = K32_00000000;
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 mask = vec_and(Compare8u<compareType>(Load<align>(src + col), _value), K8_01);
                    _count = vec_msum(mask, K8_01, _count);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 mask = vec_and(Compare8u<compareType>(Load<false>(src + width - A), _value), tailMask);
                    _count = vec_msum(mask, K8_01, _count);
                }
                src += stride;
            }
            *count = ExtractSum(_count);
        }

        template <SimdCompareType compareType> 
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            if(Aligned(src) && Aligned(stride))
                ConditionalCount8u<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount8u<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch(compareType)
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

        template <bool align, SimdCompareType compareType> 
        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            assert(width >= HA);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            v128_u16 tailMask = ShiftLeft(K16_0001, HA - width + alignedWidth);

            v128_s16 _value = SIMD_VEC_SET1_EPI16(value);
            v128_u32 _count = K32_00000000;
            for(size_t row = 0; row < height; ++row)
            {
                const int16_t * s = (const int16_t *)src;
                for(size_t col = 0; col < alignedWidth; col += HA)
                {
                    const v128_u16 mask = vec_and((v128_u16)Compare16i<compareType>(Load<align>(s + col), _value), K16_0001);
                    _count = vec_msum(mask, K16_0001, _count);
                }
                if(alignedWidth != width)
                {
                    const v128_u16 mask = vec_and((v128_u16)Compare16i<compareType>(Load<false>(s + width - HA), _value), tailMask);
                    _count = vec_msum(mask, K16_0001, _count);
                }
                src += stride;
            }
            *count = ExtractSum(_count);
        }

        template <SimdCompareType compareType> 
        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            if(Aligned(src) && Aligned(stride))
                ConditionalCount16i<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount16i<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, 
            int16_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch(compareType)
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

        template <bool align, SimdCompareType compareType> 
        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            v128_u8 _value = SetU8(value);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
                    const v128_u8 _src = vec_and(Load<align>(src + col), _mask);
                    rowSum = vec_msum(_src, K8_01, rowSum);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 _mask = Compare8u<compareType>(Load<false>(mask + width - A), _value);
                    const v128_u8 _src = vec_and(vec_and(Load<false>(src + width - A), _mask), tailMask);
                    rowSum = vec_msum(_src, K8_01, rowSum);
                }
                *sum += ExtractSum(rowSum);
                src += srcStride;
                mask += maskStride;
            }
        }

        template <SimdCompareType compareType> 
        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch(compareType)
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

        template <bool align, SimdCompareType compareType> 
        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            v128_u8 _value = SetU8(value);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
                    const v128_u8 _src = vec_and(Load<align>(src + col), _mask);
                    rowSum = vec_msum(_src, _src, rowSum);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 _mask = Compare8u<compareType>(Load<false>(mask + width - A), _value);
                    const v128_u8 _src = vec_and(vec_and(Load<false>(src + width - A), _mask), tailMask);
                    rowSum = vec_msum(_src, _src, rowSum);
                }
                *sum += ExtractSum(rowSum);
                src += srcStride;
                mask += maskStride;
            }
        }

        template <SimdCompareType compareType> 
        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSquareSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSquareSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch(compareType)
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

        template <bool align>
        SIMD_INLINE void AddSquareDifference(const uint8_t * src, ptrdiff_t step, const v128_u8 & mask, v128_u32 & sum)
        {
            const v128_u8 a = vec_and(Load<align>(src - step), mask);
            const v128_u8 b = vec_and(Load<align>(src + step), mask);
            const v128_u8 d = AbsDifferenceU8(a, b);
            sum = vec_msum(d, d, sum);
        }

        template <bool align, SimdCompareType compareType> 
        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A + 3 && height >= 3);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            src += srcStride;
            mask += maskStride;
            height -= 2;

            size_t alignedWidth = Simd::AlignLo(width - 1, A);
            v128_u8 noseMask = ShiftRight(K8_FF, 1);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + 1 + alignedWidth);

            v128_u8 _value = SetU8(value);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                {
                    const v128_u8 _mask = vec_and(Compare8u<compareType>(Load<false>(mask + 1), _value), noseMask);
                    AddSquareDifference<false>(src + 1, 1, _mask, rowSum);
                    AddSquareDifference<false>(src + 1, srcStride, _mask, rowSum);
                }
                for(size_t col = A; col < alignedWidth; col += A)
                {
                    const v128_u8 _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
                    AddSquareDifference<false>(src + col, 1, _mask, rowSum);
                    AddSquareDifference<align>(src + col, srcStride, _mask, rowSum);
                }
                if(alignedWidth != width - 1)
                {
                    size_t offset = width - A - 1;
                    const v128_u8 _mask = vec_and(Compare8u<compareType>(Load<false>(mask + offset), _value), tailMask);
                    AddSquareDifference<false>(src + offset, 1, _mask, rowSum);
                    AddSquareDifference<false>(src + offset, srcStride, _mask, rowSum);
                }
                *sum += ExtractSum(rowSum);
                src += srcStride;
                mask += maskStride;
            }
        }

        template <SimdCompareType compareType> 
        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSquareGradientSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSquareGradientSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch(compareType)
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
#endif// SIMD_VSX_ENABLE
}