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
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (value[col] > background[col])
                        background[col]++;
                }
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    background[col] = MaxU8(value[col], background[col]);
                }
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (value[col] > backgroundValue[col] && backgroundCount[col] < 0xFF)
                        backgroundCount[col]++;
                }
                value += valueStride;
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        SIMD_INLINE void AdjustEdge(const uint8_t & count, uint8_t & value, uint8_t threshold)
        {
            if (count < threshold)
            {
                if (value > 0)
                    value--;
            }
            else if (count > threshold)
            {
                if (value < 0xFF)
                    value++;
            }
        }

        void EdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    AdjustEdge(backgroundCount[col], backgroundValue[col], threshold);
                    backgroundCount[col] = 0;
                }
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        void EdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
            uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (mask[col])
                        AdjustEdge(backgroundCount[col], backgroundValue[col], threshold);
                    backgroundCount[col] = 0;
                }
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
                mask += maskStride;
            }
        }

        void EdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            Copy(value, valueStride, width, height, 1, background, backgroundStride);
        }

        void EdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    if (mask[col])
                        background[col] = value[col];
                }
                value += valueStride;
                background += backgroundStride;
                mask += maskStride;
            }
        }
    }
}
