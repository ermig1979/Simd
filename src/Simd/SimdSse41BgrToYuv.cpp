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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE void LoadBgr(const __m128i * p, __m128i & blue, __m128i & green, __m128i & red)
        {
            __m128i bgr[3];
            bgr[0] = Load<align>(p + 0);
            bgr[1] = Load<align>(p + 1);
            bgr[2] = Load<align>(p + 2);
            blue = BgrToBlue(bgr);
            green = BgrToGreen(bgr);
            red = BgrToRed(bgr);
        }

        SIMD_INLINE __m128i Average16(const __m128i & s0, const __m128i & s1)
        {
            return _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(s0, K8_01), _mm_maddubs_epi16(s1, K8_01)), K16_0002), 2);
        }

        template <bool align> SIMD_INLINE void BgrToYuv420p(const uint8_t * bgr0, size_t bgrStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgr1 = bgr0 + bgrStride;
            uint8_t * y1 = y0 + yStride;

            __m128i blue[2][2], green[2][2], red[2][2];

            LoadBgr<align>((__m128i*)bgr0 + 0, blue[0][0], green[0][0], red[0][0]);
            Store<align>((__m128i*)y0 + 0, BgrToY8(blue[0][0], green[0][0], red[0][0]));

            LoadBgr<align>((__m128i*)bgr0 + 3, blue[0][1], green[0][1], red[0][1]);
            Store<align>((__m128i*)y0 + 1, BgrToY8(blue[0][1], green[0][1], red[0][1]));

            LoadBgr<align>((__m128i*)bgr1 + 0, blue[1][0], green[1][0], red[1][0]);
            Store<align>((__m128i*)y1 + 0, BgrToY8(blue[1][0], green[1][0], red[1][0]));

            LoadBgr<align>((__m128i*)bgr1 + 3, blue[1][1], green[1][1], red[1][1]);
            Store<align>((__m128i*)y1 + 1, BgrToY8(blue[1][1], green[1][1], red[1][1]));

            blue[0][0] = Average16(blue[0][0], blue[1][0]);
            blue[0][1] = Average16(blue[0][1], blue[1][1]);
            green[0][0] = Average16(green[0][0], green[1][0]);
            green[0][1] = Average16(green[0][1], green[1][1]);
            red[0][0] = Average16(red[0][0], red[1][0]);
            red[0][1] = Average16(red[0][1], red[1][1]);

            Store<align>((__m128i*)u, _mm_packus_epi16(BgrToU16(blue[0][0], green[0][0], red[0][0]), BgrToU16(blue[0][1], green[0][1], red[0][1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(BgrToV16(blue[0][0], green[0][0], red[0][0]), BgrToV16(blue[0][1], green[0][1], red[0][1])));
        }

        template <bool align> void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A6 = A * 6;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < alignedWidth; colY += DA, colUV += A, colBgr += A6)
                    BgrToYuv420p<align>(bgr + colBgr, bgrStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgrToYuv420p<false>(bgr + offset * 3, bgrStride, y + offset, yStride, u + offset / 2, v + offset / 2);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv420p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv420p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE void Average16(__m128i & a)
        {
            a = _mm_srli_epi16(_mm_add_epi16(_mm_maddubs_epi16(a, K8_01), K16_0001), 1);
        }

        template <bool align> SIMD_INLINE void BgrToYuv422p(const uint8_t * bgr, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            __m128i blue[2], green[2], red[2];

            LoadBgr<align>((__m128i*)bgr + 0, blue[0], green[0], red[0]);
            Store<align>((__m128i*)y + 0, BgrToY8(blue[0], green[0], red[0]));

            LoadBgr<align>((__m128i*)bgr + 3, blue[1], green[1], red[1]);
            Store<align>((__m128i*)y + 1, BgrToY8(blue[1], green[1], red[1]));

            Average16(blue[0]);
            Average16(blue[1]);
            Average16(green[0]);
            Average16(green[1]);
            Average16(red[0]);
            Average16(red[1]);

            Store<align>((__m128i*)u, _mm_packus_epi16(BgrToU16(blue[0], green[0], red[0]), BgrToU16(blue[1], green[1], red[1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(BgrToV16(blue[0], green[0], red[0]), BgrToV16(blue[1], green[1], red[1])));
        }

        template <bool align> void BgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A6 = A * 6;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < alignedWidth; colY += DA, colUV += A, colBgr += A6)
                    BgrToYuv422p<align>(bgr + colBgr, y + colY, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgrToYuv422p<false>(bgr + offset * 3, y + offset, u + offset / 2, v + offset / 2);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv422p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv422p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }

        template <bool align> SIMD_INLINE void BgrToYuv444p(const uint8_t * bgr, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            __m128i blue, green, red;
            LoadBgr<align>((__m128i*)bgr, blue, green, red);
            Store<align>((__m128i*)y, BgrToY8(blue, green, red));
            Store<align>((__m128i*)u, BgrToU8(blue, green, red));
            Store<align>((__m128i*)v, BgrToV8(blue, green, red));
        }

        template <bool align> void BgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            const size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgr = 0; col < alignedWidth; col += A, colBgr += A3)
                    BgrToYuv444p<align>(bgr + colBgr, y + col, u + col, v + col);
                if (width != alignedWidth)
                {
                    size_t col = width - A;
                    BgrToYuv444p<false>(bgr + col * 3, y + col, u + col, v + col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv444p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv444p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }
    }
#endif
}
