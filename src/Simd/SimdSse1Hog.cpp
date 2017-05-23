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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdAllocator.hpp"

#include <vector>

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
	namespace Sse
	{
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
            typedef std::vector<float, Simd::Allocator<float> > Vector32f;
            typedef std::vector<__m128, Simd::Allocator<__m128> > Vector128f;

            size_t _w, _h, _s;
            Vector32f _buffer;
            Vector128f _filter;

            void Init(size_t w, size_t h, size_t cs, size_t rs)
            {
                _w = w - cs + 1;
                _s = AlignHi(_w, F);
                _h = h - rs + 1;
                _buffer.resize(_s*h);
            }

            template <bool align> void FilterCols(const float * src, const __m128 * filter, size_t size, float * dst)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<false>(src + i), filter[i]));
                Store<align>(dst, sum);
            }

            void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterCols<true>(src + col, _filter.data(), size, dst + col);
                    if(alignedWidth != width)
                        FilterCols<false>(src + width - F, _filter.data(), size, dst + width - F);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <int add, bool end> void FilterRows(const float * src, size_t stride, const __m128 * filter, size_t size, float * dst, const __m128 & mask)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<!end>(src), filter[i]));
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __m128 tailMask = RightNotZero(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterRows<add, false>(src + col, srcStride, _filter.data(), size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterRows<add, true>(src + width - F, srcStride, _filter.data(), size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:

            void Run(const float * src, size_t srcStride, size_t width, size_t height,
                const float * colFilter, size_t colSize, const float * rowFilter, size_t rowSize, float * dst, size_t dstStride, int add)
            {
                Init(width, height, colSize, rowSize);

                FilterCols(src, srcStride, _w, height, colFilter, colSize, _buffer.data(), _s);

                if (add)
                    FilterRows<1>(_buffer.data(), _s, _w, _h, rowFilter, rowSize, dst, dstStride);
                else
                    FilterRows<0>(_buffer.data(), _s, _w, _h, rowFilter, rowSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
            const float * colFilter, size_t colSize, const float * rowFilter, size_t rowSize, float * dst, size_t dstStride, int add)
        {
            assert(width >= F + colSize - 1 && height >= rowSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, colFilter, colSize, rowFilter, rowSize, dst, dstStride, add);
        }
	}
#endif// SIMD_SSE_ENABLE
}
