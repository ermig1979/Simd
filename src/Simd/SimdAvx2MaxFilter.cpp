/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2020-2020 Andrey Turkin,
*               2026-2026 TianWei Lin.
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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare3x3(const uint8_t* y[3], size_t offset, __m256i a[9])
        {
            LoadNose3<align, step>(y[0] + offset, a + 0);
            LoadNose3<align, step>(y[1] + offset, a + 3);
            LoadNose3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare3x3(const uint8_t* y[3], size_t offset, __m256i a[9])
        {
            LoadBody3<align, step>(y[0] + offset, a + 0);
            LoadBody3<align, step>(y[1] + offset, a + 3);
            LoadBody3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare3x3(const uint8_t* y[3], size_t offset, __m256i a[9])
        {
            LoadTail3<align, step>(y[0] + offset, a + 0);
            LoadTail3<align, step>(y[1] + offset, a + 3);
            LoadTail3<align, step>(y[2] + offset, a + 6);
        }

        SIMD_INLINE __m256i Max9(__m256i a[9], int threshold)
        {
            __m256i max, result;
            max = a[0];
            for (int i = 1; i < 9; ++i)
            {
                max = _mm256_max_epu8(max, a[i]);
            }

            if(1 >= threshold)
            {
                return max;
            }

            result = _mm256_setzero_si256();
            for (int i = 0; i < 9; ++i)
            {
                __m256i cmp = _mm256_cmpeq_epi8(max, a[i]);
                __m256i ones = _mm256_and_si256(cmp, _mm256_set1_epi8(1));
                result = _mm256_add_epi8(result, ones);
            }

            __m256i mask = _mm256_cmpgt_epi8(result, _mm256_set1_epi8(threshold - 1));
            max = _mm256_blendv_epi8(a[4], max, mask);

            return max;
        }

        template <bool align, size_t step> void MaxFilterSquare3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride, int threshold)
        {
            assert(step*(width - 1) >= A);

            const uint8_t * y[3];
            __m256i a[9];

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride*(row - 1);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                if (row < 1)
                    y[0] = y[1];
                if (row >= height - 1)
                    y[2] = y[1];

                LoadNoseSquare3x3<align, step>(y, 0, a);
                Store<align>((__m256i*)(dst), Max9(a, threshold));

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare3x3<align, step>(y, col, a);
                    Store<align>((__m256i*)(dst + col), Max9(a, threshold));
                }

                size_t col = size - A;
                LoadTailSquare3x3<align, step>(y, col, a);
                Store<align>((__m256i*)(dst + col), Max9(a, threshold));
            }
        }

        template <bool align> void MaxFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare3x3<align, 1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare3x3<align, 2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare3x3<align, 3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare3x3<align, 4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }

        void MaxFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride, int threshold)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MaxFilterSquare3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride, threshold);
            else
                MaxFilterSquare3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride, threshold);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare5x5(const uint8_t* y[5], size_t offset, __m256i a[25])
        {
            LoadNose5<align, step>(y[0] + offset, a + 0);
            LoadNose5<align, step>(y[1] + offset, a + 5);
            LoadNose5<align, step>(y[2] + offset, a + 10);
            LoadNose5<align, step>(y[3] + offset, a + 15);
            LoadNose5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare5x5(const uint8_t* y[5], size_t offset, __m256i a[25])
        {
            LoadBody5<align, step>(y[0] + offset, a + 0);
            LoadBody5<align, step>(y[1] + offset, a + 5);
            LoadBody5<align, step>(y[2] + offset, a + 10);
            LoadBody5<align, step>(y[3] + offset, a + 15);
            LoadBody5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare5x5(const uint8_t* y[5], size_t offset, __m256i a[25])
        {
            LoadTail5<align, step>(y[0] + offset, a + 0);
            LoadTail5<align, step>(y[1] + offset, a + 5);
            LoadTail5<align, step>(y[2] + offset, a + 10);
            LoadTail5<align, step>(y[3] + offset, a + 15);
            LoadTail5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare5x6(const uint8_t* y[6], size_t offset, __m256i a[30])
        {
            LoadNose5<align, step>(y[0] + offset, a + 0);
            LoadNose5<align, step>(y[1] + offset, a + 5);
            LoadNose5<align, step>(y[2] + offset, a + 10);
            LoadNose5<align, step>(y[3] + offset, a + 15);
            LoadNose5<align, step>(y[4] + offset, a + 20);
            LoadNose5<align, step>(y[5] + offset, a + 25);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare5x6(const uint8_t* y[6], size_t offset, __m256i a[30])
        {
            LoadBody5<align, step>(y[0] + offset, a + 0);
            LoadBody5<align, step>(y[1] + offset, a + 5);
            LoadBody5<align, step>(y[2] + offset, a + 10);
            LoadBody5<align, step>(y[3] + offset, a + 15);
            LoadBody5<align, step>(y[4] + offset, a + 20);
            LoadBody5<align, step>(y[5] + offset, a + 25);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare5x6(const uint8_t* y[6], size_t offset, __m256i a[30])
        {
            LoadTail5<align, step>(y[0] + offset, a + 0);
            LoadTail5<align, step>(y[1] + offset, a + 5);
            LoadTail5<align, step>(y[2] + offset, a + 10);
            LoadTail5<align, step>(y[3] + offset, a + 15);
            LoadTail5<align, step>(y[4] + offset, a + 20);
            LoadTail5<align, step>(y[5] + offset, a + 25);
        }

        SIMD_INLINE __m256i Max25(__m256i a[25], int threshold)
        {
            __m256i max, result;
            max = a[0];
            for (int i = 1; i < 25; ++i)
            {
                max = _mm256_max_epu8(max, a[i]);
            }

            if(1 >= threshold)
            {
                return max;
            }

            result = _mm256_setzero_si256();
            for (int i = 0; i < 25; ++i)
            {
                __m256i cmp = _mm256_cmpeq_epi8(max, a[i]);
                __m256i ones = _mm256_and_si256(cmp, _mm256_set1_epi8(1));
                result = _mm256_add_epi8(result, ones);
            }

            __m256i mask = _mm256_cmpgt_epi8(result, _mm256_set1_epi8(threshold - 1));
            max = _mm256_blendv_epi8(a[12], max, mask);

            return max;
        }

        SIMD_INLINE void Max25x2(__m256i a[30], int threshold)
        {
            __m256i max_0, max_1, result_0, result_1;
            max_0 = a[0];
            max_1 = a[5];
            for (int i = 1; i < 25; ++i)
            {
                max_0 = _mm256_max_epu8(max_0, a[i]);
                max_1 = _mm256_max_epu8(max_1, a[i+5]);
            }

            if(1 >= threshold)
            {
                a[0] = max_0;
                a[1] = max_1;
                return;
            }

            result_0 = _mm256_setzero_si256();
            result_1 = _mm256_setzero_si256();
            for (int i = 0; i < 25; ++i)
            {
                __m256i cmp = _mm256_cmpeq_epi8(max_0, a[i]);
                __m256i ones = _mm256_and_si256(cmp, _mm256_set1_epi8(1));
                result_0 = _mm256_add_epi8(result_0, ones);
                cmp = _mm256_cmpeq_epi8(max_1, a[i+5]);
                ones = _mm256_and_si256(cmp, _mm256_set1_epi8(1));
                result_1 = _mm256_add_epi8(result_1, ones);
            }

            __m256i mask = _mm256_cmpgt_epi8(result_0, _mm256_set1_epi8(threshold - 1));
            a[0] = _mm256_blendv_epi8(a[12], max_0, mask);
            mask = _mm256_cmpgt_epi8(result_1, _mm256_set1_epi8(threshold - 1));
            a[1] = _mm256_blendv_epi8(a[17], max_1, mask);

            return;
        }

        template <bool align, size_t step> void MaxFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(step * (width - 2) >= A);

            const uint8_t* y[6];
            __m256i a[30];

            size_t size = step * width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            size_t row = 0;
            for (row = 0; row < height - 1; row += 2, dst += dstStride * 2)
            {
                y[0] = src + srcStride * (row - 2);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                y[3] = y[2] + srcStride;
                y[4] = y[3] + srcStride;
                y[5] = y[4] + srcStride;
                if (row < 2)
                {
                    y[0] = y[1] = y[2];
                }
                if (row >= height - 3)
                {
                    if (row >= height - 2)
                        y[4] = y[3];
                    y[5] = y[4];
                }

                LoadNoseSquare5x6<align, step>(y, 0, a);
                Max25x2(a, threshold);
                Store<align>((__m256i*)dst, a[0]);
                Store<align>((__m256i*)(dst + dstStride), a[1]);

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare5x6<align, step>(y, col, a);
                    Max25x2(a, threshold);
                    Store<align>((__m256i*)(dst + col), a[0]);
                    Store<align>((__m256i*)(dst + dstStride + col), a[1]);
                }

                size_t col = size - A;
                LoadTailSquare5x6<false, step>(y, col, a);
                Max25x2(a, threshold);
                Store<false>((__m256i*)(dst + col), a[0]);
                Store<false>((__m256i*)(dst + dstStride + col), a[1]);
            }

            for (; row < height; ++row, dst += dstStride)
            {
                y[0] = src + srcStride * (row - 2);
                y[1] = y[0] + srcStride;
                y[2] = y[1] + srcStride;
                y[3] = y[2] + srcStride;
                y[4] = y[3] + srcStride;
                if (row < 2)
                {
                    if (row < 1)
                        y[1] = y[2];
                    y[0] = y[1];
                }
                if (row >= height - 2)
                {
                    if (row >= height - 1)
                        y[3] = y[2];
                    y[4] = y[3];
                }

                LoadNoseSquare5x5<align, step>(y, 0, a);
                Store<align>((__m256i*)dst, Max25(a, threshold));

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare5x5<align, step>(y, col, a);
                    Store<align>((__m256i*)(dst + col), Max25(a, threshold));
                }

                size_t col = size - A;
                LoadTailSquare5x5<false, step>(y, col, a);
                Store<false>((__m256i*)(dst + col), Max25(a, threshold));
            }
        }

        template <bool align> void MaxFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare5x5<align, 1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare5x5<align, 2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare5x5<align, 3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare5x5<align, 4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }

        void MaxFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride, int threshold)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MaxFilterSquare5x5<true>(src, srcStride, width, height, channelCount, dst, dstStride, threshold);
            else
                MaxFilterSquare5x5<false>(src, srcStride, width, height, channelCount, dst, dstStride, threshold);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
