/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        namespace
        {
            struct Buffer
            {
                const int size;
                __m128 * cos, *sin;
                __m128i * pos, *neg;
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    width = AlignHi(width, A / sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(__m128i) + sizeof(__m128)) * 2 * size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (__m128*)(value + width + 1);
                    sin = cos + size;
                    pos = (__m128i*)(sin + size);
                    neg = pos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = _mm_set1_ps((float)::cos(i*M_PI / size));
                        sin[i] = _mm_set1_ps((float)::sin(i*M_PI / size));
                        pos[i] = _mm_set1_epi32(i);
                        neg[i] = _mm_set1_epi32(size + i);
                    }
                }

                ~Buffer()
                {
                    Free(_p);
                }

            private:
                void *_p;
            };
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m128 & dx, const __m128 & dy, Buffer & buffer, size_t col)
        {
            __m128 bestDot = _mm_setzero_ps();
            __m128i bestIndex = _mm_setzero_si128();
            for (int i = 0; i < buffer.size; ++i)
            {
                __m128 dot = _mm_add_ps(_mm_mul_ps(dx, buffer.cos[i]), _mm_mul_ps(dy, buffer.sin[i]));
                __m128 mask = _mm_cmpgt_ps(dot, bestDot);
                bestDot = _mm_max_ps(dot, bestDot);
                bestIndex = Combine(_mm_castps_si128(mask), buffer.pos[i], bestIndex);

                dot = _mm_sub_ps(_mm_setzero_ps(), dot);
                mask = _mm_cmpgt_ps(dot, bestDot);
                bestDot = _mm_max_ps(dot, bestDot);
                bestIndex = Combine(_mm_castps_si128(mask), buffer.neg[i], bestIndex);
            }
            Store<align>((__m128i*)(buffer.index + col), bestIndex);
            Store<align>(buffer.value + col, Sse2::Sqrt<0>(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m128i & t, const __m128i & l, const __m128i & r, const __m128i & b, Buffer & buffer, size_t col)
        {
            HogDirectionHistograms<align>(
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpacklo_epi16(r, K_ZERO), _mm_unpacklo_epi16(l, K_ZERO))),
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpacklo_epi16(b, K_ZERO), _mm_unpacklo_epi16(t, K_ZERO))),
                buffer, col + 0);
            HogDirectionHistograms<align>(
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpackhi_epi16(r, K_ZERO), _mm_unpackhi_epi16(l, K_ZERO))),
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpackhi_epi16(b, K_ZERO), _mm_unpackhi_epi16(t, K_ZERO))),
                buffer, col + 4);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            __m128i t = Load<false>((__m128i*)(s - stride));
            __m128i l = Load<false>((__m128i*)(s - 1));
            __m128i r = Load<false>((__m128i*)(s + 1));
            __m128i b = Load<false>((__m128i*)(s + stride));
            HogDirectionHistograms<align>(_mm_unpacklo_epi8(t, K_ZERO), _mm_unpacklo_epi8(l, K_ZERO),
                _mm_unpacklo_epi8(r, K_ZERO), _mm_unpacklo_epi8(b, K_ZERO), buffer, col + 0);
            HogDirectionHistograms<align>(_mm_unpackhi_epi8(t, K_ZERO), _mm_unpackhi_epi8(l, K_ZERO),
                _mm_unpackhi_epi8(r, K_ZERO), _mm_unpackhi_epi8(b, K_ZERO), buffer, col + 8);
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);
            assert(width >= A + 2);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width / cellX)*(height / cellY) * sizeof(float));

            size_t alignedWidth = AlignLo(width - 2, A) + 1;

            for (size_t row = 1; row < height - 1; ++row)
            {
                const uint8_t * s = src + stride*row;
                for (size_t col = 1; col < alignedWidth; col += A)
                    HogDirectionHistograms<true>(s, stride, buffer, col);
                HogDirectionHistograms<false>(s, stride, buffer, width - 1 - A);
                Base::AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void HogDeinterleave(const float* src, size_t count, float** dst, size_t offset, size_t i)
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

        void HogDeinterleave(const float* src, size_t srcStride, size_t width, size_t height, size_t count, float** dst, size_t dstStride)
        {
            assert(width >= F && count >= F);

            size_t alignedCount = AlignLo(count, F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                src += srcStride;
            }
        }

        //---------------------------------------------------------------------

        namespace HogSeparableFilter_Detail
        {
            template <int add, bool end> SIMD_INLINE void Set(float* dst, const __m128& value, const __m128& mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float* dst, const __m128& value, const __m128& mask)
            {
                Store<false>(dst, _mm_add_ps(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float* dst, const __m128& value, const __m128& mask)
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
                _buffer.Resize(_s * h);
            }

            template <bool align> SIMD_INLINE void FilterRows(const float* src, const __m128* filter, size_t size, float* dst)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<false>(src + i), filter[i]));
                Store<align>(dst, sum);
            }

            void FilterRows(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
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

            template <int add, bool end> SIMD_INLINE void FilterCols(const float* src, size_t stride, const __m128* filter, size_t size, float* dst, const __m128& mask)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<!end>(src), filter[i]));
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
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
            assert(width >= F + rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
