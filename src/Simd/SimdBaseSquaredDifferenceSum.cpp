/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    rowSum += SquaredDifference(a[col], b[col]);
                }
                *sum += rowSum;
                a += aStride;
                b += bStride;
            }
        }

        void SquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    if (mask[col] == index)
                        rowSum += SquaredDifference(a[col], b[col]);
                }
                *sum += rowSum;
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }

        void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float sums[4] = { 0, 0, 0, 0 };
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                sums[0] += Simd::Square(a[i + 0] - b[i + 0]);
                sums[1] += Simd::Square(a[i + 1] - b[i + 1]);
                sums[2] += Simd::Square(a[i + 2] - b[i + 2]);
                sums[3] += Simd::Square(a[i + 3] - b[i + 3]);
            }
            for (; i < size; ++i)
                sums[0] += Simd::Square(a[i] - b[i]);
            *sum = sums[0] + sums[1] + sums[2] + sums[3];
        }

        SIMD_INLINE void KahanSum(float value, float & sum, float & correction)
        {
            float term = value - correction;
            float temp = sum + term;
            correction = (temp - sum) - term;
            sum = temp;
        }

#if defined(__GNUC__) && (defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE))
#ifdef __clang__
#pragma clang optimize off
#else
#pragma GCC push_options
#pragma GCC optimize ("O1")
#endif
#elif defined(_MSC_VER) && (_MSC_VER >= 1914)
#pragma optimize ("", off)
#endif
        void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float sums[4] = { 0, 0, 0, 0 };
            float corrections[4] = { 0, 0, 0, 0 };
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                KahanSum(Simd::Square(a[i + 0] - b[i + 0]), sums[0], corrections[0]);
                KahanSum(Simd::Square(a[i + 1] - b[i + 1]), sums[1], corrections[1]);
                KahanSum(Simd::Square(a[i + 2] - b[i + 2]), sums[2], corrections[2]);
                KahanSum(Simd::Square(a[i + 3] - b[i + 3]), sums[3], corrections[3]);
            }
            for (; i < size; ++i)
                KahanSum(Simd::Square(a[i + 0] - b[i + 0]), sums[0], corrections[0]);
            *sum = sums[0] + sums[1] + sums[2] + sums[3];
        }
#if defined(__GNUC__) && (defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE))
#ifdef __clang__
#pragma clang optimize on
#else
#pragma GCC pop_options
#endif 
#elif defined(_MSC_VER) && (_MSC_VER >= 1920)
#pragma optimize ("", on)
#endif
    }
}
