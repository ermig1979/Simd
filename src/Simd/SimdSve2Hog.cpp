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
#include "Simd/SimdArray.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        namespace HogSeparableFilter_Detail
        {
            template <int add> SIMD_INLINE void Set(float* dst, const svfloat32_t& value, const svbool_t& mask)
            {
                svst1_f32(mask, dst, value);
            }

            template <> SIMD_INLINE void Set<1>(float* dst, const svfloat32_t& value, const svbool_t& mask)
            {
                svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), value));
            }
        }

        class HogSeparableFilter
        {
            typedef Array<float> Array32f;

            size_t _w, _h, _s;
            Array32f _buffer;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHiAny(_w, svcntw());
                _h = h - cs + 1;
                _buffer.Resize(_s * h);
            }

            SIMD_INLINE void FilterRows(const float* src, const float* filter, size_t size, float* dst, const svbool_t& mask)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t i = 0; i < size; ++i)
                    sum = svmla_f32_x(mask, sum, svld1_f32(mask, src + i), svdup_n_f32(filter[i]));
                svst1_f32(mask, dst, sum);
            }

            void FilterRows(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
            {
                size_t F = svcntw();
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; col += F)
                        FilterRows(src + col, filter, size, dst + col, svwhilelt_b32(col, width));
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <int add> SIMD_INLINE void FilterCols(const float* src, size_t stride, const float* filter, size_t size, float* dst, const svbool_t& mask)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = svmla_f32_x(mask, sum, svld1_f32(mask, src), svdup_n_f32(filter[i]));
                HogSeparableFilter_Detail::Set<add>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
            {
                size_t F = svcntw();
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; col += F)
                        FilterCols<add>(src + col, srcStride, filter, size, dst + col, svwhilelt_b32(col, width));
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:
            void Run(const float* src, size_t srcStride, size_t width, size_t height,
                const float* rowFilter, size_t rowSize, const float* colFilter, size_t colSize, float* dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _s);

                if (add)
                    FilterCols<1>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float* src, size_t srcStride, size_t width, size_t height,
            const float* rowFilter, size_t rowSize, const float* colFilter, size_t colSize, float* dst, size_t dstStride, int add)
        {
            assert(width >= rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
#endif
}
