/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
        {
            src += i;
            __m256 a0 = Load<false>(src + 0 * count, src + 4 * count);
            __m256 a1 = Load<false>(src + 1 * count, src + 5 * count);
            __m256 a2 = Load<false>(src + 2 * count, src + 6 * count);
            __m256 a3 = Load<false>(src + 3 * count, src + 7 * count);
            __m256 b0 = _mm256_unpacklo_ps(a0, a2);
            __m256 b1 = _mm256_unpackhi_ps(a0, a2);
            __m256 b2 = _mm256_unpacklo_ps(a1, a3);
            __m256 b3 = _mm256_unpackhi_ps(a1, a3);
            Store<false>(dst[i + 0] + offset, _mm256_unpacklo_ps(b0, b2));
            Store<false>(dst[i + 1] + offset, _mm256_unpackhi_ps(b0, b2));
            Store<false>(dst[i + 2] + offset, _mm256_unpacklo_ps(b1, b3));
            Store<false>(dst[i + 3] + offset, _mm256_unpackhi_ps(b1, b3));
        }

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            assert(width >= F && count >= Sse41::F);

            size_t alignedCount = AlignLo(count, Sse41::F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float * s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += Sse41::F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - Sse41::F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float * s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += Sse41::F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - Sse41::F);
                }
                src += srcStride;
            }
        }

        namespace HogSeparableFilter_Detail
        {
            template <int add, bool end> SIMD_INLINE void Set(float * dst, const __m256 & value, const __m256 & mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float * dst, const __m256 & value, const __m256 & mask)
            {
                Store<false>(dst, _mm256_add_ps(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float * dst, const __m256 & value, const __m256 & mask)
            {
                Store<false>(dst, _mm256_add_ps(Load<false>(dst), _mm256_and_ps(value, mask)));
            }
        }

        class HogSeparableFilter
        {
            size_t _w, _h, _s;
            Array32f _buffer;
            Array256f _filter;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHi(_w, F);
                _h = h - cs + 1;
                _buffer.Resize(_s*h);
            }

            template <bool align> SIMD_INLINE void FilterRows(const float * src, const __m256 * filter, size_t size, float * dst)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm256_fmadd_ps(Load<false>(src + i), filter[i], sum);
                Store<align>(dst, sum);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm256_set1_ps(filter[i]);

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

            template <bool align> SIMD_INLINE void FilterRows_10(const float * src, const __m256 * filter, float * dst)
            {
                __m256  src0 = Load<false>(src + 0);
                __m256  src4 = Load<false>(src + 4);
                __m256  src8 = Load<false>(src + 8);
                __m256 sum0 = _mm256_mul_ps(src0, filter[0]);
                __m256 sum1 = _mm256_mul_ps(Alignr<1>(src0, src4), filter[1]);
                sum0 = _mm256_fmadd_ps(Alignr<2>(src0, src4), filter[2], sum0);
                sum1 = _mm256_fmadd_ps(Alignr<3>(src0, src4), filter[3], sum1);
                sum0 = _mm256_fmadd_ps(src4, filter[4], sum0);
                sum1 = _mm256_fmadd_ps(Alignr<1>(src4, src8), filter[5], sum1);
                sum0 = _mm256_fmadd_ps(Alignr<2>(src4, src8), filter[6], sum0);
                sum1 = _mm256_fmadd_ps(Alignr<3>(src4, src8), filter[7], sum1);
                sum0 = _mm256_fmadd_ps(src8, filter[8], sum0);
                sum1 = _mm256_fmadd_ps(Load<false>(src + 9), filter[9], sum1);
                Store<align>(dst, _mm256_add_ps(sum0, sum1));
            }

            void FilterRows_10(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, float * dst, size_t dstStride)
            {
                __m256 _filter[10];
                for (size_t i = 0; i < 10; ++i)
                    _filter[i] = _mm256_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterRows_10<true>(src + col, _filter, dst + col);
                    if (alignedWidth != width)
                        FilterRows_10<false>(src + width - F, _filter, dst + width - F);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <int add, bool end> SIMD_INLINE void FilterCols(const float * src, size_t stride, const __m256 * filter, size_t size, float * dst, const __m256 & mask)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm256_fmadd_ps(Load<!end>(src), filter[i], sum);
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add, bool end> SIMD_INLINE void FilterCols4x(const float * src, size_t stride, const __m256 * filter, size_t size, float * dst, const __m256 & mask)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                for (size_t i = 0; i < size; ++i, src += stride)
                {
                    __m256 f = filter[i];
                    sums[0] = _mm256_fmadd_ps(Load<!end>(src + 0 * F), f, sums[0]);
                    sums[1] = _mm256_fmadd_ps(Load<!end>(src + 1 * F), f, sums[1]);
                    sums[2] = _mm256_fmadd_ps(Load<!end>(src + 2 * F), f, sums[2]);
                    sums[3] = _mm256_fmadd_ps(Load<!end>(src + 3 * F), f, sums[3]);
                }
                HogSeparableFilter_Detail::Set<add, end>(dst + 0 * F, sums[0], mask);
                HogSeparableFilter_Detail::Set<add, end>(dst + 1 * F, sums[1], mask);
                HogSeparableFilter_Detail::Set<add, end>(dst + 2 * F, sums[2], mask);
                HogSeparableFilter_Detail::Set<add, end>(dst + 3 * F, sums[3], mask);
            }

            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm256_set1_ps(filter[i]);

                size_t fullAlignedWidth = AlignLo(width, QF);
                size_t partialAlignedWidth = AlignLo(width, F);
                __m256 tailMask = RightNotZero32f(width - partialAlignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QF)
                        FilterCols4x<add, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    for (; col < partialAlignedWidth; col += F)
                        FilterCols<add, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (partialAlignedWidth != width)
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

                if (colSize == 10)
                    FilterRows_10(src, srcStride, _w, height, rowFilter, _buffer.data, _s);
                else
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
#endif
}
