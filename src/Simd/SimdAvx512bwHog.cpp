/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
        {
            src += i;
            __m512 a0 = Load<false>(src + 0x0 * count, src + 0x4 * count, src + 0x8 * count, src + 0xC * count);
            __m512 a1 = Load<false>(src + 0x1 * count, src + 0x5 * count, src + 0x9 * count, src + 0xD * count);
            __m512 a2 = Load<false>(src + 0x2 * count, src + 0x6 * count, src + 0xA * count, src + 0xE * count);
            __m512 a3 = Load<false>(src + 0x3 * count, src + 0x7 * count, src + 0xB * count, src + 0xF * count);
            __m512 b0 = _mm512_unpacklo_ps(a0, a2);
            __m512 b1 = _mm512_unpackhi_ps(a0, a2);
            __m512 b2 = _mm512_unpacklo_ps(a1, a3);
            __m512 b3 = _mm512_unpackhi_ps(a1, a3);
            Store<false>(dst[i + 0] + offset, _mm512_unpacklo_ps(b0, b2));
            Store<false>(dst[i + 1] + offset, _mm512_unpackhi_ps(b0, b2));
            Store<false>(dst[i + 2] + offset, _mm512_unpacklo_ps(b1, b3));
            Store<false>(dst[i + 3] + offset, _mm512_unpackhi_ps(b1, b3));
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
            template <int add, bool mask> SIMD_INLINE void Set(float * dst, const __m512 & value, __mmask16 tail = -1)
            {
                Store<false, mask>(dst, value, tail);
            }

            template <> SIMD_INLINE void Set<1, false>(float * dst, const __m512 & value, __mmask16 tail)
            {
                Store<false>(dst, _mm512_add_ps(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float * dst, const __m512 & value, __mmask16 tail)
            {
                Store<false, true>(dst, _mm512_add_ps((Load<false, true>(dst, tail)), value), tail);
            }
        }

        class HogSeparableFilter
        {
            size_t _w, _h, _s;
            Array32f _buffer;
            Array512f _filter;

            SIMD_INLINE void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHi(_w, F);
                _h = h - cs + 1;
                _buffer.Resize(_s*h);
            }

            template <bool align> SIMD_INLINE void FilterRows(const float * src, const __m512 * filter, size_t size, float * dst)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm512_fmadd_ps(Load<false>(src + i), filter[i], sum);
                Store<align>(dst, sum);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm512_set1_ps(filter[i]);

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

            template <bool align> SIMD_INLINE void FilterRows_10(const float * src, const __m512 * filter, float * dst)
            {
                __m512  src0 = Load<false>(src + 0);
                __m512  srcf = Load<false>(src + F);
                __m512 sum0 = _mm512_mul_ps(Alignr<0>(src0, srcf), filter[0]);
                __m512 sum1 = _mm512_mul_ps(Alignr<1>(src0, srcf), filter[1]);
                sum0 = _mm512_fmadd_ps(Alignr<2>(src0, srcf), filter[2], sum0);
                sum1 = _mm512_fmadd_ps(Alignr<3>(src0, srcf), filter[3], sum1);
                sum0 = _mm512_fmadd_ps(Alignr<4>(src0, srcf), filter[4], sum0);
                sum1 = _mm512_fmadd_ps(Alignr<5>(src0, srcf), filter[5], sum1);
                sum0 = _mm512_fmadd_ps(Alignr<6>(src0, srcf), filter[6], sum0);
                sum1 = _mm512_fmadd_ps(Alignr<7>(src0, srcf), filter[7], sum1);
                sum0 = _mm512_fmadd_ps(Alignr<8>(src0, srcf), filter[8], sum0);
                sum1 = _mm512_fmadd_ps(Alignr<9>(src0, srcf), filter[9], sum1);
                Store<align>(dst, _mm512_add_ps(sum0, sum1));
            }

            void FilterRows_10(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, float * dst, size_t dstStride)
            {
                __m512 _filter[10];
                for (size_t i = 0; i < 10; ++i)
                    _filter[i] = _mm512_set1_ps(filter[i]);

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

            template <int add, bool mask> SIMD_INLINE void FilterCols(const float * src, size_t stride, const __m512 * filter, size_t size, float * dst, __mmask16 tail = -1)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm512_fmadd_ps((Load<true, mask>(src, tail)), filter[i], sum);
                HogSeparableFilter_Detail::Set<add, mask>(dst, sum, tail);
            }

            template <int add> void SIMD_INLINE FilterCols4x(const float * src, size_t stride, const __m512 * filter, size_t size, float * dst)
            {
                __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                for (size_t i = 0; i < size; ++i, src += stride)
                {
                    __m512 f = filter[i];
                    sums[0] = _mm512_fmadd_ps(Load<true>(src + 0 * F), f, sums[0]);
                    sums[1] = _mm512_fmadd_ps(Load<true>(src + 1 * F), f, sums[1]);
                    sums[2] = _mm512_fmadd_ps(Load<true>(src + 2 * F), f, sums[2]);
                    sums[3] = _mm512_fmadd_ps(Load<true>(src + 3 * F), f, sums[3]);
                }
                HogSeparableFilter_Detail::Set<add, false>(dst + 0 * F, sums[0]);
                HogSeparableFilter_Detail::Set<add, false>(dst + 1 * F, sums[1]);
                HogSeparableFilter_Detail::Set<add, false>(dst + 2 * F, sums[2]);
                HogSeparableFilter_Detail::Set<add, false>(dst + 3 * F, sums[3]);
            }

            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm512_set1_ps(filter[i]);

                size_t fullAlignedWidth = AlignLo(width, QF);
                size_t alignedWidth = AlignLo(width, F);
                __mmask16 tailMask = TailMask16(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QF)
                        FilterCols4x<add>(src + col, srcStride, _filter.data, size, dst + col);
                    for (; col < alignedWidth; col += F)
                        FilterCols<add, false>(src + col, srcStride, _filter.data, size, dst + col);
                    if (col < width)
                        FilterCols<add, true>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
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
