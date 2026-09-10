/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        namespace HogSeparableFilter_Detail
        {
            template <int add, bool end> SIMD_INLINE void Set(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, vaddq_f32(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, vaddq_f32(Load<false>(dst), And(value, mask)));
            }
        }

        class HogSeparableFilter
        {
            typedef Array<float> Array32f;
            typedef Array<float32x4_t> Array128f;

            size_t _w, _h, _s;
            Array32f _buffer;
            Array128f _filter;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHi(_w, F);
                _h = h - cs + 1;
                _buffer.Resize(_s*h);
            }

            template <bool align> void FilterRows(const float * src, const float32x4_t * filter, size_t size, float * dst)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i)
                    sum = vmlaq_f32(sum, Load<false>(src + i), filter[i]);
                Store<align>(dst, sum);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

                size_t alignedWidth = AlignLo(width, F);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterRows<true>(src + col, _filter.data, size, dst + col);
                    if (alignedWidth != width)
                        FilterRows<false>(src + width - F, _filter.data, size, dst + width - F);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <int add, bool end> void FilterCols(const float * src, size_t stride, const float32x4_t * filter, size_t size, float * dst, const float32x4_t & mask)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = vmlaq_f32(sum, Load<!end>(src), filter[i]);
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                float32x4_t tailMask = RightNotZero32f(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterCols<add, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterCols<add, true>(src + width - F, srcStride, _filter.data, size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:

            void Run(const float * src, size_t srcStride, size_t width, size_t height,
                const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _s);

                if (add)
                    FilterCols<1>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
            const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
        {
            assert(width >= F + rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
#endif// SIMD_NEON_ENABLE
}
