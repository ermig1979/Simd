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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare3x3(const uint8_t* y[3], size_t offset, __m512i a[9])
        {
            LoadNose3<align, step>(y[0] + offset, a + 0);
            LoadNose3<align, step>(y[1] + offset, a + 3);
            LoadNose3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare3x3(const uint8_t* y[3], size_t offset, __m512i a[9])
        {
            LoadBody3<align, step>(y[0] + offset, a + 0);
            LoadBody3<align, step>(y[1] + offset, a + 3);
            LoadBody3<align, step>(y[2] + offset, a + 6);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare3x3(const uint8_t* y[3], size_t offset, __m512i a[9])
        {
            LoadTail3<align, step>(y[0] + offset, a + 0);
            LoadTail3<align, step>(y[1] + offset, a + 3);
            LoadTail3<align, step>(y[2] + offset, a + 6);
        }

        SIMD_INLINE void Midpoint9(__m512i a[9])
        {
            __m512i max, min;
            max = min = a[0];
            for (int i = 1; i < 9; ++i)
            {
                max = _mm512_max_epu8(max, a[i]);
                min = _mm512_min_epu8(min, a[i]);
            }
            a[0] = _mm512_avg_epu8(min, max);
        }

        template <bool align, size_t step> void MidpointFilterSquare3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);

            const uint8_t * y[3];
            __m512i a[9];

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
                Midpoint9(a);
                Store<align>(dst, a[0]);

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare3x3<align, step>(y, col, a);
                    Midpoint9(a);
                    Store<align>(dst + col, a[0]);
                }

                size_t col = size - A;
                LoadTailSquare3x3<align, step>(y, col, a);
                Midpoint9(a);
                Store<align>(dst + col, a[0]);
            }
        }

        template <bool align> void MidpointFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MidpointFilterSquare3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MidpointFilterSquare3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MidpointFilterSquare3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MidpointFilterSquare3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void MidpointFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MidpointFilterSquare3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                MidpointFilterSquare3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare5x5(const uint8_t* y[5], size_t offset, __m512i a[25])
        {
            LoadNose5<align, step>(y[0] + offset, a + 0);
            LoadNose5<align, step>(y[1] + offset, a + 5);
            LoadNose5<align, step>(y[2] + offset, a + 10);
            LoadNose5<align, step>(y[3] + offset, a + 15);
            LoadNose5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare5x5(const uint8_t* y[5], size_t offset, __m512i a[25])
        {
            LoadBody5<align, step>(y[0] + offset, a + 0);
            LoadBody5<align, step>(y[1] + offset, a + 5);
            LoadBody5<align, step>(y[2] + offset, a + 10);
            LoadBody5<align, step>(y[3] + offset, a + 15);
            LoadBody5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare5x5(const uint8_t* y[5], size_t offset, __m512i a[25])
        {
            LoadTail5<align, step>(y[0] + offset, a + 0);
            LoadTail5<align, step>(y[1] + offset, a + 5);
            LoadTail5<align, step>(y[2] + offset, a + 10);
            LoadTail5<align, step>(y[3] + offset, a + 15);
            LoadTail5<align, step>(y[4] + offset, a + 20);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNoseSquare5x6(const uint8_t* y[6], size_t offset, __m512i a[30])
        {
            LoadNose5<align, step>(y[0] + offset, a + 0);
            LoadNose5<align, step>(y[1] + offset, a + 5);
            LoadNose5<align, step>(y[2] + offset, a + 10);
            LoadNose5<align, step>(y[3] + offset, a + 15);
            LoadNose5<align, step>(y[4] + offset, a + 20);
            LoadNose5<align, step>(y[5] + offset, a + 25);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBodySquare5x6(const uint8_t* y[6], size_t offset, __m512i a[30])
        {
            LoadBody5<align, step>(y[0] + offset, a + 0);
            LoadBody5<align, step>(y[1] + offset, a + 5);
            LoadBody5<align, step>(y[2] + offset, a + 10);
            LoadBody5<align, step>(y[3] + offset, a + 15);
            LoadBody5<align, step>(y[4] + offset, a + 20);
            LoadBody5<align, step>(y[5] + offset, a + 25);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTailSquare5x6(const uint8_t* y[6], size_t offset, __m512i a[30])
        {
            LoadTail5<align, step>(y[0] + offset, a + 0);
            LoadTail5<align, step>(y[1] + offset, a + 5);
            LoadTail5<align, step>(y[2] + offset, a + 10);
            LoadTail5<align, step>(y[3] + offset, a + 15);
            LoadTail5<align, step>(y[4] + offset, a + 20);
            LoadTail5<align, step>(y[5] + offset, a + 25);
        }

        SIMD_INLINE void Midpoint25(__m512i a[25])
        {
            __m512i max, min;
            max = min = a[0];
            for(int i = 1; i < 25; ++i)
            {
                max = _mm512_max_epu8(max, a[i]);
                min = _mm512_min_epu8(min, a[i]);
            }
            a[0] = _mm512_avg_epu8(min, max);
        }

        SIMD_INLINE void Midpoint25x2(__m512i a[30])
        {
            __m512i max_0, max_1, min_0, min_1;
            max_0 = min_0 = a[0];
            max_1 = min_1 = a[5];
            for(int i = 1; i < 25; ++i)
            {
                max_0 = _mm512_max_epu8(max_0, a[i]);
                min_0 = _mm512_min_epu8(min_0, a[i]);
                max_1 = _mm512_max_epu8(max_1, a[i+5]);
                min_1 = _mm512_min_epu8(min_1, a[i+5]);
            }
            a[0] = _mm512_avg_epu8(min_0, max_0);
            a[1] = _mm512_avg_epu8(min_1, max_1);
        }

        template <bool align, size_t step> void MidpointFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(step * (width - 2) >= A);

            const uint8_t* y[6];
            __m512i a[30];

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
                Midpoint25x2(a);
                Store<align>(dst, a[0]);
                Store<align>(dst + dstStride, a[1]);

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare5x6<align, step>(y, col, a);
                    Midpoint25x2(a);
                    Store<align>(dst + col, a[0]);
                    Store<align>(dst + dstStride + col, a[1]);
                }

                size_t col = size - A;
                LoadTailSquare5x6<false, step>(y, col, a);
                Midpoint25x2(a);
                Store<false>(dst + col, a[0]);
                Store<false>(dst + dstStride + col, a[1]);
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
                Midpoint25(a);
                Store<align>(dst, a[0]);

                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBodySquare5x5<align, step>(y, col, a);
                    Midpoint25(a);
                    Store<align>(dst + col, a[0]);
                }

                size_t col = size - A;
                LoadTailSquare5x5<false, step>(y, col, a);
                Midpoint25(a);
                Store<false>(dst + col, a[0]);
            }
        }

        template <bool align> void MidpointFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MidpointFilterSquare5x5<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MidpointFilterSquare5x5<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MidpointFilterSquare5x5<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MidpointFilterSquare5x5<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void MidpointFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(width) && Aligned(dst) && Aligned(dstStride))
                MidpointFilterSquare5x5<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                MidpointFilterSquare5x5<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
