/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height);

            uint64_t sum = 0;
            int min_ = UCHAR_MAX;
            int max_ = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                {
                    int value = src[col];
                    max_ = MaxU8(value, max_);
                    min_ = MinU8(value, min_);
                    rowSum += value;
                }
                sum += rowSum;
                src += stride;
            }
            *average = (uint8_t)((sum + width*height / 2) / (width*height));
            *min = min_;
            *max = max_;
        }

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t sum = 0;
                for (size_t col = 0; col < width; ++col)
                    sum += src[col];
                sums[row] = sum;
                src += stride;
            }
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            memset(sums, 0, sizeof(uint32_t)*width);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    sums[col] += src[col];
                src += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
            height--;
            sums[height] = 0;
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t sum = 0;
                for (size_t col = 0; col < width; ++col)
                    sum += AbsDifferenceU8(src0[col], src1[col]);
                sums[row] = sum;
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + 1;
            memset(sums, 0, sizeof(uint32_t)*width);
            width--;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    sums[col] += AbsDifferenceU8(src0[col], src1[col]);
                src0 += stride;
                src1 += stride;
            }
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                    rowSum += src[col];
                *sum += rowSum;
                src += stride;
            }
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                    rowSum += Square(src[col]);
                *sum += rowSum;
                src += stride;
            }
        }
		
		void ValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
        {
            assert(width < 0x10000);

            *valueSum = 0;
			*squareSum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowValueSum = 0;
				int rowSquareSum = 0;
                for (size_t col = 0; col < width; ++col)
				{
                    int value = src[col];
                    rowValueSum += value;
                    rowSquareSum += Square(value);
				}
                *valueSum += rowValueSum;
				*squareSum += rowSquareSum;
                src += stride;
            }
        }

        template<size_t channels> void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width < 0x10000);

            for (size_t c = 0; c < channels; ++c)
            {
                valueSums[c] = 0;
                squareSums[c] = 0;
            }
            for (size_t y = 0; y < height; ++y)
            {
                uint32_t rowValueSums[channels], rowSquareSums[channels];
                for (size_t c = 0; c < channels; ++c)
                {
                    rowValueSums[c] = 0;
                    rowSquareSums[c] = 0;
                }
                for (size_t x = 0; x < width; ++x)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        int value = src[c];
                        rowValueSums[c] += value;
                        rowSquareSums[c] += Square(value);
                    }
                    src += channels;
                }
                for (size_t c = 0; c < channels; ++c)
                {
                    valueSums[c] += rowValueSums[c];
                    squareSums[c] += rowSquareSums[c];
                }                
                src += stride - width*channels;
            }
        }

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            switch (channels)
            {
            case 1: ValueSquareSums<1>(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums<2>(src, stride, width, height, valueSums, squareSums); break;
            case 3: ValueSquareSums<3>(src, stride, width, height, valueSums, squareSums); break;
            case 4: ValueSquareSums<4>(src, stride, width, height, valueSums, squareSums); break;
            default:
                assert(0);
            }
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                    rowSum += a[col] * b[col];
                *sum += rowSum;
                a += aStride;
                b += bStride;
            }
        }
    }
}
