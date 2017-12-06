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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void FillBgr(uint8_t * dst, const __m512i bgrs[3], const __mmask64 * tails)
        {
            Store<align, mask>(dst + 0 * A, bgrs[0], tails[0]);
            Store<align, mask>(dst + 1 * A, bgrs[1], tails[1]);
            Store<align, mask>(dst + 2 * A, bgrs[2], tails[2]);
        }

        template <bool align> SIMD_INLINE void FillBgr2(uint8_t * dst, const __m512i bgrs[3])
        {
            Store<align>(dst + 0 * A, bgrs[0]);
            Store<align>(dst + 1 * A, bgrs[1]);
            Store<align>(dst + 2 * A, bgrs[2]);
            Store<align>(dst + 3 * A, bgrs[0]);
            Store<align>(dst + 4 * A, bgrs[1]);
            Store<align>(dst + 5 * A, bgrs[2]);
        }

        template <bool align> void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            size_t size = width * 3;
            size_t step = A * 3;
            size_t alignedSize = AlignLo(width, A) * 3;
            __mmask64 tailMasks[3];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64(size - alignedSize - A*c);
            size_t step2 = 2 * step;
            size_t alignedSize2 = AlignLo(width, 2 * A) * 3;

            uint32_t bgrb = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(blue) << 24);
            uint32_t grbg = uint32_t(green) | (uint32_t(red) << 8) | (uint32_t(blue) << 16) | (uint32_t(green) << 24);
            uint32_t rbgr = uint32_t(red) | (uint32_t(blue) << 8) | (uint32_t(green) << 16) | (uint32_t(red) << 24);

            __m512i bgrs[3];
            bgrs[0] = _mm512_setr_epi32(bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb);
            bgrs[1] = _mm512_setr_epi32(grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg);
            bgrs[2] = _mm512_setr_epi32(rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr, bgrb, grbg, rbgr);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize2; offset += step2)
                    FillBgr2<align>(dst + offset, bgrs);
                for (; offset < alignedSize; offset += step)
                    FillBgr<align, false>(dst + offset, bgrs, tailMasks);
                if (offset < size)
                    FillBgr<align, true>(dst + offset, bgrs, tailMasks);
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
            size_t size = width * 4;
            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);

            uint32_t bgra32 = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);
            __m512i bgra512 = _mm512_set1_epi32(bgra32);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < fullAlignedSize; offset += QA)
                {
                    Store<align>(dst + offset + 0 * A, bgra512);
                    Store<align>(dst + offset + 1 * A, bgra512);
                    Store<align>(dst + offset + 2 * A, bgra512);
                    Store<align>(dst + offset + 3 * A, bgra512);
                }
                for (; offset < alignedSize; offset += A)
                    Store<align, false>(dst + offset, bgra512, tailMask);
                if (offset < size)
                    Store<align, true>(dst + offset, bgra512, tailMask);
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

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const __m512i & pixel)
        {
            size_t fullAlignedWidth = AlignLo(width, QA);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                {
                    Store<align>(dst + col + 0 * A, pixel);
                    Store<align>(dst + col + 1 * A, pixel);
                    Store<align>(dst + col + 2 * A, pixel);
                    Store<align>(dst + col + 3 * A, pixel);
                }
                for (; col < alignedWidth; col += A)
                    Store<align>(dst + col, pixel);
                if (col < width)
                    Store<align, true>(dst + col, pixel, tailMask);
                dst += stride;
            }
        }

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
        {
            if (pixelSize == 3)
                FillBgr<align>(dst, stride, width, height, pixel[0], pixel[1], pixel[2]);
            else
            {
                __m512i _pixel;
                switch (pixelSize)
                {
                case 1:
                    _pixel = _mm512_set1_epi8(*pixel);
                    break;
                case 2:
                    _pixel = _mm512_set1_epi16(*(uint16_t*)pixel);
                    break;
                case 4:
                    _pixel = _mm512_set1_epi32(*(uint32_t*)pixel);
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
#endif// SIMD_AVX512BW_ENABLE
}
