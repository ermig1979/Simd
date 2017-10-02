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
        const int SHIFT = 16;

        SIMD_INLINE uint32_t ShiftedWeightedSquare(int difference, int weight)
        {
            return difference*difference*weight >> SHIFT;
        }

        SIMD_INLINE int FeatureDifference(int value, int lo, int hi)
        {
            return Max(0, Max(value - hi, lo - value));
        }

        void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
            uint16_t weight, uint8_t * difference, size_t differenceStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    int featureDifference = FeatureDifference(value[col], lo[col], hi[col]);
                    int sum = difference[col] + ShiftedWeightedSquare(featureDifference, weight);
                    difference[col] = Min(sum, 0xFF);
                }
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                difference += differenceStride;
            }
        }
    }
}
