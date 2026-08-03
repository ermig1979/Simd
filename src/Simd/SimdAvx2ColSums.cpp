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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*width + sizeof(uint32_t)*width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sums16;
                uint32_t * sums32;
            private:
                void *_p;
            };
        }

        template <bool align> SIMD_INLINE void Sum16x1(const uint8_t * src, size_t, uint16_t * dst)
        {
            __m256i sum0 = Load<true>((__m256i*)dst + 0);
            __m256i sum1 = Load<true>((__m256i*)dst + 1);
            __m256i src0 = Load<align>((__m256i*)src);
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            Store<true>((__m256i*)dst + 0, sum0);
            Store<true>((__m256i*)dst + 1, sum1);
        }

        template <bool align> SIMD_INLINE void Sum16x4(const uint8_t * src, size_t stride, uint16_t * dst)
        {
            __m256i sum0 = Load<true>((__m256i*)dst + 0);
            __m256i sum1 = Load<true>((__m256i*)dst + 1);
            __m256i src0 = Load<align>((__m256i*)src);
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 1 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 2 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 3 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            Store<true>((__m256i*)dst + 0, sum0);
            Store<true>((__m256i*)dst + 1, sum1);
        }

        template <bool align> SIMD_INLINE void Sum16x8(const uint8_t * src, size_t stride, uint16_t * dst)
        {
            __m256i sum0 = Load<true>((__m256i*)dst + 0);
            __m256i sum1 = Load<true>((__m256i*)dst + 1);
            __m256i src0 = Load<align>((__m256i*)src);
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 1 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 2 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 3 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 4 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 5 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 6 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            src0 = Load<align>((__m256i*)(src + 7 * stride));
            sum0 = _mm256_add_epi16(sum0, _mm256_unpacklo_epi8(src0, K_ZERO));
            sum1 = _mm256_add_epi16(sum1, _mm256_unpackhi_epi8(src0, K_ZERO));
            Store<true>((__m256i*)dst + 0, sum0);
            Store<true>((__m256i*)dst + 1, sum1);
        }

        SIMD_INLINE void Sum16(__m256i src8, uint16_t * sums16)
        {
            Store<true>((__m256i*)sums16 + 0, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 0), _mm256_unpacklo_epi8(src8, K_ZERO)));
            Store<true>((__m256i*)sums16 + 1, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 1), _mm256_unpackhi_epi8(src8, K_ZERO)));
        }

        SIMD_INLINE void Sum16To32(const uint16_t * src, uint32_t * dst)
        {
            __m256i lo = LoadPermuted<true>((__m256i*)src + 0);
            __m256i hi = LoadPermuted<true>((__m256i*)src + 1);
            Store<true>((__m256i*)dst + 0, _mm256_add_epi32(Load<true>((__m256i*)dst + 0), _mm256_unpacklo_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 1, _mm256_add_epi32(Load<true>((__m256i*)dst + 1), _mm256_unpacklo_epi16(hi, K_ZERO)));
            Store<true>((__m256i*)dst + 2, _mm256_add_epi32(Load<true>((__m256i*)dst + 2), _mm256_unpackhi_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 3, _mm256_add_epi32(Load<true>((__m256i*)dst + 3), _mm256_unpackhi_epi16(hi, K_ZERO)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                size_t rowEnd4 = AlignLo(rowEnd, 4);
                size_t rowEnd8 = AlignLo(rowEnd, 8);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                size_t row = rowStart;
                for (; row < rowEnd8; row += 8)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x8<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x8<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x4<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x4<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x1<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x1<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*alignedLoWidth);
            if (alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint32_t)*(width - alignedLoWidth));
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m256i _src0 = Load<align>((__m256i*)(src + col + 0));
                        __m256i _src1 = Load<false>((__m256i*)(src + col + 1));
                        Sum16(AbsDifferenceU8(_src0, _src1), buffer.sums16 + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        __m256i _src0 = Load<false>((__m256i*)(src + width - A + 0));
                        __m256i _src1 = Load<false>((__m256i*)(src + width - A + 1));
                        Sum16(AbsDifferenceU8(_src0, _src1), buffer.sums16 + alignedLoWidth);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*alignedLoWidth);
            if (alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint32_t)*(width - alignedLoWidth));
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }
    }
#endif
}
