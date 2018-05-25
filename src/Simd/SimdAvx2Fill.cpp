/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
        template <bool align> void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            size_t size = width * 3;
            size_t step = A * 3;
            size_t alignedSize = AlignLo(width, A) * 3;

            uint32_t bgrb = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(blue) << 24);
            uint32_t grbg = uint32_t(green) | (uint32_t(red) << 8) | (uint32_t(blue) << 16) | (uint32_t(green) << 24);
            uint32_t rbgr = uint32_t(red) | (uint32_t(blue) << 8) | (uint32_t(green) << 16) | (uint32_t(red) << 24);

            __m256i bgrs[3];
            bgrs[0] = _mm256_setr_epi32(bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg);
            bgrs[1] = _mm256_setr_epi32(rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb);
            bgrs[2] = _mm256_setr_epi32(grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize; offset += step)
                {
                    Store<align>((__m256i*)(dst + offset) + 0, bgrs[0]);
                    Store<align>((__m256i*)(dst + offset) + 1, bgrs[1]);
                    Store<align>((__m256i*)(dst + offset) + 2, bgrs[2]);
                }
                if (offset < size)
                {
                    offset = size - step;
                    Store<false>((__m256i*)(dst + offset) + 0, bgrs[0]);
                    Store<false>((__m256i*)(dst + offset) + 1, bgrs[1]);
                    Store<false>((__m256i*)(dst + offset) + 2, bgrs[2]);
                }
                dst += stride;
            }
        }

        void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            if (Aligned(dst) && Aligned(stride))
                FillBgr<true>(dst, stride, width, height, blue, green, red);
            else
                FillBgr<false>(dst, stride, width, height, blue, green, red);
        }

        template <bool align> void FillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            uint32_t bgra32 = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);
            size_t alignedWidth = AlignLo(width, 8);
            __m256i bgra256 = _mm256_set1_epi32(bgra32);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += 8)
                    Store<align>((__m256i*)((uint32_t*)dst + col), bgra256);
                if (width != alignedWidth)
                    Store<false>((__m256i*)((uint32_t*)dst + width - 8), bgra256);
                dst += stride;
            }
        }

        void FillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            if (Aligned(dst) && Aligned(stride))
                FillBgra<true>(dst, stride, width, height, blue, green, red, alpha);
            else
                FillBgra<false>(dst, stride, width, height, blue, green, red, alpha);
        }

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const __m256i & pixel)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            size_t fullAlignedWidth = AlignLo(width, QA);
            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                {
                    Store<align>((__m256i*)(dst + col) + 0, pixel);
                    Store<align>((__m256i*)(dst + col) + 1, pixel);
                    Store<align>((__m256i*)(dst + col) + 2, pixel);
                    Store<align>((__m256i*)(dst + col) + 3, pixel);
                }
                for (; col < alignedWidth; col += A)
                    Store<align>((__m256i*)(dst + col), pixel);
                if (col < width)
                    Store<false>((__m256i*)(dst + width - A), pixel);
                dst += stride;
            }
        }

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
        {
            if (pixelSize == 3)
                FillBgr<align>(dst, stride, width, height, pixel[0], pixel[1], pixel[2]);
            else
            {
                __m256i _pixel;
                switch (pixelSize)
                {
                case 1:
                    _pixel = _mm256_set1_epi8(*pixel);
                    break;
                case 2:
                    _pixel = _mm256_set1_epi16(*(uint16_t*)pixel);
                    break;
                case 4:
                    _pixel = _mm256_set1_epi32(*(uint32_t*)pixel);
                    break;
                default:
                    assert(0);
                }
                FillPixel<align>(dst, stride, width*pixelSize, height, _pixel);
            }
        }

        void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
        {
            if (Aligned(dst) && Aligned(stride))
                FillPixel<true>(dst, stride, width, height, pixel, pixelSize);
            else
                FillPixel<false>(dst, stride, width, height, pixel, pixelSize);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
