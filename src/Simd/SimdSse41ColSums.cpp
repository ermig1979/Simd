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
#include "Simd/SimdStore.h"

#if defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void Sum16x1(const uint8_t* src, size_t stride, uint16_t* dst)
        {
            __m128i sum0 = _mm_loadu_si128((__m128i*)dst + 0);
            __m128i sum1 = _mm_loadu_si128((__m128i*)dst + 1);
            __m128i src0 = _mm_loadu_si128((__m128i*)src);
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            _mm_storeu_si128((__m128i*)dst + 0, sum0);
            _mm_storeu_si128((__m128i*)dst + 1, sum1);
        }

        SIMD_INLINE void Sum16x4(const uint8_t* src, size_t stride, uint16_t* dst)
        {
            __m128i sum0 = _mm_loadu_si128((__m128i*)dst + 0);
            __m128i sum1 = _mm_loadu_si128((__m128i*)dst + 1);
            __m128i src0 = _mm_loadu_si128((__m128i*)src);
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 1 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 2 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 3 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            _mm_storeu_si128((__m128i*)dst + 0, sum0);
            _mm_storeu_si128((__m128i*)dst + 1, sum1);
        }

        SIMD_INLINE void Sum16x8(const uint8_t* src, size_t stride, uint16_t* dst)
        {
            __m128i sum0 = _mm_loadu_si128((__m128i*)dst + 0);
            __m128i sum1 = _mm_loadu_si128((__m128i*)dst + 1);
            __m128i src0 = _mm_loadu_si128((__m128i*)src);
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 1 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 2 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 3 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 4 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 5 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 6 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            src0 = _mm_loadu_si128((__m128i*)(src + 7 * stride));
            sum0 = _mm_add_epi16(sum0, _mm_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm_add_epi16(sum1, _mm_unpackhi_epi8(src0, K_ZERO));
            _mm_storeu_si128((__m128i*)dst + 0, sum0);
            _mm_storeu_si128((__m128i*)dst + 1, sum1);
        }

        SIMD_INLINE void Sum32(const uint16_t* src, uint32_t* dst)
        {
            __m128i sum0 = _mm_loadu_si128((__m128i*)dst + 0);
            __m128i sum1 = _mm_loadu_si128((__m128i*)dst + 1);
            __m128i src0 = _mm_loadu_si128((__m128i*)src);
            sum0 = _mm_add_epi32(sum0, _mm_unpacklo_epi16(src0, K_ZERO));
            sum1 = _mm_add_epi32(sum1, _mm_unpackhi_epi16(src0, K_ZERO));
            _mm_storeu_si128((__m128i*)dst + 0, sum0);
            _mm_storeu_si128((__m128i*)dst + 1, sum1);
        }

        template <bool align> void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedLoWidth);
            size_t stepSize = 256;
            size_t stepCount = DivHi(height, 256);

            Array16u sums16(alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step * stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                size_t rowEnd4 = AlignLo(rowEnd, 4);
                size_t rowEnd8 = AlignLo(rowEnd, 8);

                sums16.Clear();
                size_t row = rowStart;
                for (; row < rowEnd8; row += 8)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x8(src + col, stride, sums16.data + col);
                    if (alignedLoWidth != width)
                        Sum16x8(src + width - A, stride, sums16.data + alignedLoWidth);
                    src += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x4(src + col, stride, sums16.data + col);
                    if (alignedLoWidth != width)
                        Sum16x4(src + width - A, stride, sums16.data + alignedLoWidth);
                    src += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x1(src + col, stride, sums16.data + col);
                    if (alignedLoWidth != width)
                        Sum16x1(src + width - A, stride, sums16.data + alignedLoWidth);
                    src += stride;
                }
                if(step == 0)
                    memset(sums, 0, sizeof(uint32_t) * width);
                size_t col = 0;
                for (; col < alignedLoWidth; col += HA)
                    Sum32(sums16.data + col, sums + col);
                for (size_t shift = A - (width - alignedLoWidth); col < width; col++)
                    sums[col] += sums16[shift + col];
            }
        }

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void Sum16(__m128i src8, uint16_t* sums16)
        {
            Store<align>((__m128i*)sums16 + 0, _mm_add_epi16(Load<align>((__m128i*)sums16 + 0), _mm_unpacklo_epi8(src8, K_ZERO)));
            Store<align>((__m128i*)sums16 + 1, _mm_add_epi16(Load<align>((__m128i*)sums16 + 1), _mm_unpackhi_epi8(src8, K_ZERO)));
        }

        template <bool align> SIMD_INLINE void Sum32(__m128i src16, uint32_t* sums32)
        {
            Store<align>((__m128i*)sums32 + 0, _mm_add_epi32(Load<align>((__m128i*)sums32 + 0), _mm_unpacklo_epi16(src16, K_ZERO)));
            Store<align>((__m128i*)sums32 + 1, _mm_add_epi32(Load<align>((__m128i*)sums32 + 1), _mm_unpackhi_epi16(src16, K_ZERO)));
        }

        template <bool align> void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Array16u sums16(alignedHiWidth);
            Array32u sums32(alignedHiWidth);
            memset(sums32.data, 0, sizeof(uint32_t) * alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step * stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(sums16.data, 0, sizeof(uint16_t) * width);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m128i _src0 = Load<align>((__m128i*)(src + col + 0));
                        __m128i _src1 = Load<false>((__m128i*)(src + col + 1));
                        Sum16<true>(AbsDifferenceU8(_src0, _src1), sums16.data + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        __m128i _src0 = Load<false>((__m128i*)(src + width - A + 0));
                        __m128i _src1 = Load<false>((__m128i*)(src + width - A + 1));
                        Sum16<false>(_mm_and_si128(AbsDifferenceU8(_src0, _src1), tailMask), sums16.data + width - A);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    __m128i src16 = Load<true>((__m128i*)(sums16.data + col));
                    Sum32<true>(src16, sums32.data + col);
                }
            }
            memcpy(sums, sums32.data, sizeof(uint32_t) * width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }
    }
#endif
}

#if defined(__GNUC__)
# pragma GCC diagnostic pop
#endif
