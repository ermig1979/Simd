/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

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

        void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            *area = 0;
            *x = 0;
            *y = 0;
            *xx = 0;
            *xy = 0;
            *yy = 0;
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
    }
}