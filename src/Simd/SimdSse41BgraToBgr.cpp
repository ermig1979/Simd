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
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE  
    namespace Sse41
    {
        template <bool align> SIMD_INLINE void BgraToBgrBody(const uint8_t * bgra, uint8_t * bgr, __m128i k[3][2])
        {
            Store<align>((__m128i*)(bgr + 0), _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 0), k[0][0]));
            Store<false>((__m128i*)(bgr + 12), _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 1), k[0][0]));
            Store<false>((__m128i*)(bgr + 24), _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 2), k[0][0]));
            Store<false>((__m128i*)(bgr + 36), _mm_shuffle_epi8(Load<align>((__m128i*)bgra + 3), k[0][0]));
        }

        template <bool align> SIMD_INLINE void BgraToBgr(const uint8_t * bgra, uint8_t * bgr, __m128i k[3][2])
        {
            __m128i bgra0 = Load<align>((__m128i*)bgra + 0);
            __m128i bgra1 = Load<align>((__m128i*)bgra + 1);
            __m128i bgra2 = Load<align>((__m128i*)bgra + 2);
            __m128i bgra3 = Load<align>((__m128i*)bgra + 3);
            Store<align>((__m128i*)bgr + 0, _mm_or_si128(_mm_shuffle_epi8(bgra0, k[0][0]), _mm_shuffle_epi8(bgra1, k[0][1])));
            Store<align>((__m128i*)bgr + 1, _mm_or_si128(_mm_shuffle_epi8(bgra1, k[1][0]), _mm_shuffle_epi8(bgra2, k[1][1])));
            Store<align>((__m128i*)bgr + 2, _mm_or_si128(_mm_shuffle_epi8(bgra2, k[2][0]), _mm_shuffle_epi8(bgra3, k[2][1])));
        }

        template <bool align> void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t widthA = AlignLo(width, A) - A;

            __m128i k[3][2];
            k[0][0] = _mm_setr_epi8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
            k[0][1] = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x1, 0x2, 0x4);
            k[1][0] = _mm_setr_epi8(0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1, -1, -1, -1, -1);
            k[1][1] = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9);
            k[2][0] = _mm_setr_epi8(0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            k[2][1] = _mm_setr_epi8(-1, -1, -1, -1, 0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgraToBgrBody<align>(bgra + 4 * col, bgr + 3 * col, k);
                BgraToBgr<align>(bgra + 4 * widthA, bgr + 3 * widthA, k);
                if (widthA + A !=  width)
                    BgraToBgr<false>(bgra + 4 * (width - A), bgr + 3 * (width - A), k);
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
                BgraToBgr<true>(bgra, width, height, bgraStride, bgr, bgrStride);
            else
                BgraToBgr<false>(bgra, width, height, bgraStride, bgr, bgrStride);
        }

        //---------------------------------------------------------------------

        template <bool align> void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride));

            size_t widthA = AlignLo(width, A) - A;

            __m128i k[3][2];
            k[0][0] = _mm_setr_epi8(0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1);
            k[0][1] = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x1, 0x0, 0x6);
            k[1][0] = _mm_setr_epi8(0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1, -1, -1, -1, -1);
            k[1][1] = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9);
            k[2][0] = _mm_setr_epi8(0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            k[2][1] = _mm_setr_epi8(-1, -1, -1, -1, 0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgraToBgrBody<align>(bgra + 4 * col, rgb + 3 * col, k);
                BgraToBgr<align>(bgra + 4 * widthA, rgb + 3 * widthA, k);
                if (widthA + A != width)
                    BgraToBgr<false>(bgra + 4 * (width - A), rgb + 3 * (width - A), k);
                bgra += bgraStride;
                rgb += rgbStride;
            }
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(rgb) && Aligned(rgbStride))
                BgraToRgb<true>(bgra, width, height, bgraStride, rgb, rgbStride);
            else
                BgraToRgb<false>(bgra, width, height, bgraStride, rgb, rgbStride);
        }

        //---------------------------------------------------------------------

        const __m128i K8_BGRA_TO_RGBA = SIMD_MM_SETR_EPI8(0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF);

        template <bool align> SIMD_INLINE void BgraToRgba(const uint8_t* bgra, uint8_t* rgba)
        {
            Store<align>((__m128i*)rgba, _mm_shuffle_epi8(Load<align>((__m128i*)bgra), K8_BGRA_TO_RGBA));
        }

        template <bool align> void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(rgba) && Aligned(rgbaStride));

            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t i = 0; i < sizeA; i += A)
                    BgraToRgba<align>(bgra + i, rgba + i);
                if (size != sizeA)
                    BgraToRgba<false>(bgra + size - A, rgba + size - A);
                bgra += bgraStride;
                rgba += rgbaStride;
            }
        }

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(rgba) && Aligned(rgbaStride))
                BgraToRgba<true>(bgra, width, height, bgraStride, rgba, rgbaStride);
            else
                BgraToRgba<false>(bgra, width, height, bgraStride, rgba, rgbaStride);
        }
    }
#endif
}
