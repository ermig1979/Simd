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
#ifndef __SimdIntegral_h__
#define __SimdIntegral_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
    template <class T> struct IntegralBuffer
    {
        IntegralBuffer(size_t size)
        {
            _p = Allocate(sizeof(T)*size);
            p = (T*)_p;
        }

        ~IntegralBuffer()
        {
            Free(_p);
        }

        T * p;
    private:
        void *_p;
    };

    template <class TSum> void IntegralSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, TSum * sum, size_t sumStride)
    {
        memset(sum, 0, (width + 1) * sizeof(TSum));
        sum += sumStride + 1;

        for (size_t row = 0; row < height; row++)
        {
            TSum rowSum = 0;
            sum[-1] = 0;
            for (size_t col = 0; col < width; col++)
            {
                rowSum += src[col];
                sum[col] = rowSum + sum[col - sumStride];
            }
            src += srcStride;
            sum += sumStride;
        }
    }

    template <class TSum, class TSqsum> void IntegralSumSqsum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        TSum * sum, size_t sumStride, TSqsum * sqsum, size_t sqsumStride)
    {
        memset(sum, 0, (width + 1) * sizeof(TSum));
        sum += sumStride + 1;

        memset(sqsum, 0, (width + 1) * sizeof(TSqsum));
        sqsum += sqsumStride + 1;

        for (size_t row = 0; row < height; row++)
        {
            TSum row_sum = 0;
            TSqsum row_sqsum = 0;
            sum[-1] = 0;
            sqsum[-1] = 0;
            for (size_t col = 0; col < width; col++)
            {
                TSum value = src[col];
                row_sum += value;
                row_sqsum += value*value;
                sum[col] = row_sum + sum[col - sumStride];
                sqsum[col] = row_sqsum + sqsum[col - sqsumStride];
            }
            src += srcStride;
            sum += sumStride;
            sqsum += sqsumStride;
        }
    }

    template <class TSum, class TSqsum> void IntegralSumSqsumTilted(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height,
        TSum * sum, ptrdiff_t sumStride, TSqsum * sqsum, ptrdiff_t sqsumStride, TSum * tilted, ptrdiff_t tiltedStride)
    {
        memset(sum, 0, (width + 1) * sizeof(TSum));
        sum += sumStride + 1;

        memset(sqsum, 0, (width + 1) * sizeof(TSqsum));
        sqsum += sqsumStride + 1;

        memset(tilted, 0, (width + 1) * sizeof(TSum));
        tilted += tiltedStride + 1;

        IntegralBuffer<TSum> _buffer(width + 1);
        TSum * buffer = _buffer.p;
        TSum s = 0;
        TSqsum sq = 0;

        sum[-1] = 0;
        tilted[-1] = 0;
        sqsum[-1] = 0;

        for (size_t col = 0; col < width; col++)
        {
            TSum value = src[col];
            buffer[col] = value;
            tilted[col] = value;
            s += value;
            sq += value*value;
            sum[col] = s;
            sqsum[col] = sq;
        }

        if (width == 1)
            buffer[1] = 0;

        for (size_t row = 1; row < height; ++row)
        {
            src += srcStride;
            sum += sumStride;
            tilted += tiltedStride;
            sqsum += sqsumStride;

            TSum value = src[0];
            TSum t0 = s = value;
            TSqsum tq0 = sq = value*value;

            sum[-1] = 0;
            sqsum[-1] = 0;
            tilted[-1] = tilted[-tiltedStride];

            sum[0] = sum[-sumStride] + t0;
            sqsum[0] = sqsum[-sqsumStride] + tq0;
            tilted[0] = tilted[-tiltedStride] + t0 + buffer[1];

            size_t col;
            for (col = 1; col < width - 1; ++col)
            {
                TSum t1 = buffer[col];
                buffer[col - 1] = t1 + t0;
                t0 = value = src[col];
                tq0 = value*value;
                s += t0;
                sq += tq0;
                sum[col] = sum[col - sumStride] + s;
                sqsum[col] = sqsum[col - sqsumStride] + sq;
                t1 += buffer[col + 1] + t0 + tilted[col - tiltedStride - 1];
                tilted[col] = t1;
            }

            if (width > 1)
            {
                TSum t1 = buffer[col];
                buffer[col - 1] = t1 + t0;
                t0 = value = src[col];
                tq0 = value*value;
                s += t0;
                sq += tq0;
                sum[col] = sum[col - sumStride] + s;
                sqsum[col] = sqsum[col - sqsumStride] + sq;
                tilted[col] = t0 + t1 + tilted[col - tiltedStride - 1];
                buffer[col] = t0;
            }
        }
    }

    template <class TSum> void IntegralSumTilted(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height,
        TSum * sum, ptrdiff_t sumStride, TSum * tilted, ptrdiff_t tiltedStride)
    {
        memset(sum, 0, (width + 1) * sizeof(TSum));
        sum += sumStride + 1;

        memset(tilted, 0, (width + 1) * sizeof(TSum));
        tilted += tiltedStride + 1;

        IntegralBuffer<TSum> _buffer(width + 1);
        TSum * buffer = _buffer.p;
        TSum s = 0;

        sum[-1] = 0;
        tilted[-1] = 0;

        for (size_t col = 0; col < width; col++)
        {
            TSum value = src[col];
            buffer[col] = value;
            tilted[col] = value;
            s += value;
            sum[col] = s;
        }

        if (width == 1)
            buffer[1] = 0;

        for (size_t row = 1; row < height; ++row)
        {
            src += srcStride;
            sum += sumStride;
            tilted += tiltedStride;

            TSum value = src[0];
            TSum t0 = s = value;

            sum[-1] = 0;
            tilted[-1] = tilted[-tiltedStride];

            sum[0] = sum[-sumStride] + t0;
            tilted[0] = tilted[-tiltedStride] + t0 + buffer[1];

            size_t col;
            for (col = 1; col < width - 1; ++col)
            {
                TSum t1 = buffer[col];
                buffer[col - 1] = t1 + t0;
                t0 = value = src[col];
                s += t0;
                sum[col] = sum[col - sumStride] + s;
                t1 += buffer[col + 1] + t0 + tilted[col - tiltedStride - 1];
                tilted[col] = t1;
            }

            if (width > 1)
            {
                TSum t1 = buffer[col];
                buffer[col - 1] = t1 + t0;
                t0 = value = src[col];
                s += t0;
                sum[col] = sum[col - sumStride] + s;
                tilted[col] = t0 + t1 + tilted[col - tiltedStride - 1];
                buffer[col] = t0;
            }
        }
    }
}
#endif//__SimdIntegral_h__
