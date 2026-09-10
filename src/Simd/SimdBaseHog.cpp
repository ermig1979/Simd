/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"

namespace Simd
{
    namespace Base
    {
        namespace HogSeparableFilter_Detail
        {
            template <int add> void Set(float & dst, float value);

            template <> SIMD_INLINE void Set<0>(float & dst, float value)
            {
                dst = value;
            }

            template <> SIMD_INLINE void Set<1>(float & dst, float value)
            {
                dst += value;
            }
        }

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * psrc = src + row * srcStride;
                size_t offset = row * dstStride;
                for (size_t col = 0; col < width; ++col)
                {
                    for (size_t i = 0; i < count; ++i)
                        dst[i][offset + col] = *psrc++;
                }
            }
        }

        class HogSeparableFilter
        {
            typedef Array<float> Array32f;

            size_t _w, _h;
            Array32f _buffer;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _h = h - cs + 1;
                _buffer.Resize(_w*h);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i] * filter[i];
                        dst[col] = sum;
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }


            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i*srcStride] * filter[i];
                        HogSeparableFilter_Detail::Set<add>(dst[col], sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:

            void Run(const float * src, size_t srcStride, size_t width, size_t height,
                const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _w);

                if (add)
                    FilterCols<1>(_buffer.data, _w, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _w, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
            const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
        {
            assert(width >= rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
}
