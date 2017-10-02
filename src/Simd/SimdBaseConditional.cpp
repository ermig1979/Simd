/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar,
*               2014-2016 Antonenka Mikhail.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        template <SimdCompareType compareType>
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            *count = 0;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare8u<compareType>(src[col], value))
                        (*count)++;
                }
                src += stride;
            }
        }

        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t value, SimdCompareType compareType, uint32_t * count)
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

        template <SimdCompareType compareType>
        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            *count = 0;
            for (size_t row = 0; row < height; ++row)
            {
                const int16_t * s = (const int16_t *)src;
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare16i<compareType>(s[col], value))
                        (*count)++;
                }
                src += stride;
            }
        }

        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
            int16_t value, SimdCompareType compareType, uint32_t * count)
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

        template <SimdCompareType compareType>
        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare8u<compareType>(mask[col], value))
                        rowSum += src[col];
                }
                *sum += rowSum;
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
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

        template <SimdCompareType compareType>
        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare8u<compareType>(mask[col], value))
                        rowSum += Square(src[col]);
                }
                *sum += rowSum;
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
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

        template <SimdCompareType compareType>
        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            src += srcStride + 1;
            mask += maskStride + 1;
            width -= 2;
            height -= 2;

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare8u<compareType>(mask[col], value))
                    {
                        rowSum += SquaredDifference(src[col + 1], src[col - 1]);
                        rowSum += SquaredDifference(src[col + srcStride], src[col - srcStride]);
                    }
                }
                *sum += rowSum;
                src += srcStride;
                mask += maskStride;
            }
        }

        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
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

        template <SimdCompareType compareType>
        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (Compare8u<compareType>(src[col], threshold))
                        dst[col] = value;
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalFill<SimdCompareEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareNotEqual:
                return ConditionalFill<SimdCompareNotEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreater:
                return ConditionalFill<SimdCompareGreater>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return ConditionalFill<SimdCompareGreaterOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesser:
                return ConditionalFill<SimdCompareLesser>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return ConditionalFill<SimdCompareLesserOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
}
