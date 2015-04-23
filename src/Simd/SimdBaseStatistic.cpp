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
            for(size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for(size_t col = 0; col < width; ++col)
                {
                    int value = src[col];
                    max_ = MaxU8(value, max_);
                    min_ = MinU8(value, min_);
                    rowSum += value;
                }
                sum += rowSum;
                src += stride;
            }
            *average = (uint8_t)((sum + width*height/2)/(width*height));
            *min = min_;
            *max = max_;
        }

        void GetMomentsSmall(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            for(size_t row = 0; row < height; ++row)
            {
                size_t rowArea = 0;
                size_t rowX = 0;
                size_t rowY = 0;
                size_t rowXX = 0;
                size_t rowXY = 0;
                size_t rowYY = 0;
                for(size_t col = 0; col < width; ++col)
                {
                    if(mask[col] == index)
                    {
                        rowArea++;
                        rowX += col;
                        rowY += row;
                        rowXX += col*col;
                        rowXY += col*row;
                        rowYY += row*row;
                    }               
                }
                *area += rowArea;
                *x += rowX;
                *y += rowY;
                *xx += rowXX;
                *xy += rowXY;
                *yy += rowYY;

                mask += stride;
            }
        }

        void GetMomentsLarge(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            for(size_t row = 0; row < height; ++row)
            {
                size_t rowArea = 0;
                size_t rowX = 0;
                size_t rowY = 0;
                for(size_t col = 0; col < width; ++col)
                {
                    if(mask[col] == index)
                    {
                        rowArea++;
                        rowX += col;
                        rowY += row;
                        *xx += col*col;
                        *xy += col*row;
                        *yy += row*row;
                    }               
                }
                *area += rowArea;
                *x += rowX;
                *y += rowY;

                mask += stride;
            }
        }

        SIMD_INLINE bool IsSmall(uint64_t width, uint64_t height)
        {
            return 
                width*width*width < 0x300000000ULL && 
                width*width*height < 0x200000000ULL && 
                width*height*height < 0x100000000ULL;
        }

        void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            *area = 0;
            *x = 0;
            *y = 0;
            *xx = 0;
            *xy = 0;
            *yy = 0;
            if(IsSmall(width, height))
                GetMomentsSmall(mask, stride, width, height, index, area, x, y, xx, xy, yy);
            else
                GetMomentsLarge(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        }

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            for(size_t row = 0; row < height; ++row)
            {
                uint32_t sum = 0;
                for(size_t col = 0; col < width; ++col)
                    sum += src[col];
                sums[row] = sum;
                src += stride;
            }
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            memset(sums, 0, sizeof(uint32_t)*width);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
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
            for(size_t row = 0; row < height; ++row)
            {
                uint32_t sum = 0;
                for(size_t col = 0; col < width; ++col)
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
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                    sums[col] += AbsDifferenceU8(src0[col], src1[col]);
                src0 += stride;
                src1 += stride;
            }
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for(size_t col = 0; col < width; ++col)
                    rowSum += src[col];
                *sum += rowSum;
                src += stride;
            }
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for(size_t col = 0; col < width; ++col)
                    rowSum += Square(src[col]);
                *sum += rowSum;
                src += stride;
            }
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);

            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for(size_t col = 0; col < width; ++col)
                    rowSum += a[col]*b[col];
                *sum += rowSum;
                a += aStride;
                b += bStride;
            }
        }
    }
}