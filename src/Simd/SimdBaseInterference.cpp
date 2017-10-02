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

namespace Simd
{
    namespace Base
    {
        void InterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
        {
            for (size_t row = 0; row < height; ++row)
            {
                int16_t * s = (int16_t *)statistic;
                for (size_t col = 0; col < width; ++col)
                    s[col] = Min(s[col] + increment, saturation);
                statistic += stride;
            }
        }

        void InterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            for (size_t row = 0; row < height; ++row)
            {
                int16_t * s = (int16_t *)statistic;
                for (size_t col = 0; col < width; ++col)
                    s[col] = Min(s[col] + (mask[col] == index ? increment : 0), saturation);
                statistic += statisticStride;
                mask += maskStride;
            }
        }

        void InterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation)
        {
            for (size_t row = 0; row < height; ++row)
            {
                int16_t * s = (int16_t *)statistic;
                for (size_t col = 0; col < width; ++col)
                    s[col] = Max(s[col] - decrement, saturation);
                statistic += stride;
            }
        }

        void InterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            for (size_t row = 0; row < height; ++row)
            {
                int16_t * s = (int16_t *)statistic;
                for (size_t col = 0; col < width; ++col)
                    s[col] = Max(s[col] - (mask[col] == index ? decrement : 0), saturation);
                statistic += statisticStride;
                mask += maskStride;
            }
        }
    }
}
