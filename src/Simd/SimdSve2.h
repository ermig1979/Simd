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
#ifndef __SimdSve2_h__
#define __SimdSve2_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        void BackgroundIncrementCount(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* loValue, size_t loValueStride, const uint8_t* hiValue, size_t hiValueStride,
            uint8_t* loCount, size_t loCountStride, uint8_t* hiCount, size_t hiCountStride);

        void BgraToGray(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* gray, size_t grayStride);

        void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* gray, size_t grayStride);

        void ConditionalCount8u(const uint8_t* src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t* count);

        void ConditionalCount16i(const uint8_t* src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t* count);

        void ConditionalSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void ConditionalSquareSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void ConditionalSquareGradientSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum);

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride);

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride);

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum);

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums);
    }
#endif
}
#endif
