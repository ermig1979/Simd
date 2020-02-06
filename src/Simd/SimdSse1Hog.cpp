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
#include "Simd/SimdStore.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
        {
            src += i;
            __m128 a0 = Load<false>(src + 0 * count);
            __m128 a1 = Load<false>(src + 1 * count);
            __m128 a2 = Load<false>(src + 2 * count);
            __m128 a3 = Load<false>(src + 3 * count);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpackhi_ps(a0, a2);
            __m128 b2 = _mm_unpacklo_ps(a1, a3);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            Store<false>(dst[i + 0] + offset, _mm_unpacklo_ps(b0, b2));
            Store<false>(dst[i + 1] + offset, _mm_unpackhi_ps(b0, b2));
            Store<false>(dst[i + 2] + offset, _mm_unpacklo_ps(b1, b3));
            Store<false>(dst[i + 3] + offset, _mm_unpackhi_ps(b1, b3));
        }

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            assert(width >= F && count >= F);

            size_t alignedCount = AlignLo(count, F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float * s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float * s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                src += srcStride;
            }
        }

        namespace HogSeparableFilter_Detail
        {
            template <int add, bool end> SIMD_INLINE void Set(float * dst, const __m128 & value, const __m128 & mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float * dst, const __m128 & value, const __m128 & mask)
            {
                Store<false>(dst, _mm_add_ps(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float * dst, const __m128 & value, const __m128 & mask)
            {
                Store<false>(dst, _mm_add_ps(Load<false>(dst), _mm_and_ps(value, mask)));
            }
        }

        class HogSeparableFilter
        {
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

            template <bool align> SIMD_INLINE void FilterRows(const float * src, const __m128 * filter, size_t size, float * dst)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<false>(src + i), filter[i]));
                Store<align>(dst, sum);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

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

            template <int add, bool end> SIMD_INLINE void FilterCols(const float * src, size_t stride, const __m128 * filter, size_t size, float * dst, const __m128 & mask)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<!end>(src), filter[i]));
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __m128 tailMask = RightNotZero32f(width - alignedWidth);

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
#endif// SIMD_SSE_ENABLE
}
