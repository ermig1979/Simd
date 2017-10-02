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
#include "Simd/SimdConversion.h"

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        const size_t A3 = A * 3;
        const size_t A6 = A * 6;

        SIMD_INLINE uint16x8_t Average(uint8x16_t a, uint8x16_t b)
        {
            return vshrq_n_u16(vpadalq_u8(vpadalq_u8(K16_0002, a), b), 2);
        }

        template <bool align> SIMD_INLINE void BgrToYuv420p(const uint8_t * bgr0, size_t bgrStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgr1 = bgr0 + bgrStride;
            uint8_t * y1 = y0 + yStride;

            uint8x16x3_t bgr00 = Load3<align>(bgr0);
            Store<align>(y0 + 0, BgrToY(bgr00.val[0], bgr00.val[1], bgr00.val[2]));

            uint8x16x3_t bgr01 = Load3<align>(bgr0 + A3);
            Store<align>(y0 + A, BgrToY(bgr01.val[0], bgr01.val[1], bgr01.val[2]));

            uint8x16x3_t bgr10 = Load3<align>(bgr1);
            Store<align>(y1 + 0, BgrToY(bgr10.val[0], bgr10.val[1], bgr10.val[2]));

            uint8x16x3_t bgr11 = Load3<align>(bgr1 + A3);
            Store<align>(y1 + A, BgrToY(bgr11.val[0], bgr11.val[1], bgr11.val[2]));

            uint16x8_t b0 = Average(bgr00.val[0], bgr10.val[0]);
            uint16x8_t g0 = Average(bgr00.val[1], bgr10.val[1]);
            uint16x8_t r0 = Average(bgr00.val[2], bgr10.val[2]);

            uint16x8_t b1 = Average(bgr01.val[0], bgr11.val[0]);
            uint16x8_t g1 = Average(bgr01.val[1], bgr11.val[1]);
            uint16x8_t r1 = Average(bgr01.val[2], bgr11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU(b0, g0, r0), BgrToU(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV(b0, g0, r0), BgrToV(b1, g1, r1)));
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

        SIMD_INLINE uint16x8_t Average(uint8x16_t value)
        {
            return vshrq_n_u16(vpadalq_u8(K16_0001, value), 1);
        }

        template <bool align> SIMD_INLINE void BgrToYuv422p(const uint8_t * bgr, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            uint8x16x3_t bgr0 = Load3<align>(bgr);
            Store<align>(y + 0, BgrToY(bgr0.val[0], bgr0.val[1], bgr0.val[2]));

            uint16x8_t b0 = Average(bgr0.val[0]);
            uint16x8_t g0 = Average(bgr0.val[1]);
            uint16x8_t r0 = Average(bgr0.val[2]);

            uint8x16x3_t bgr1 = Load3<align>(bgr + A3);
            Store<align>(y + A, BgrToY(bgr1.val[0], bgr1.val[1], bgr1.val[2]));

            uint16x8_t b1 = Average(bgr1.val[0]);
            uint16x8_t g1 = Average(bgr1.val[1]);
            uint16x8_t r1 = Average(bgr1.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU(b0, g0, r0), BgrToU(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV(b0, g0, r0), BgrToV(b1, g1, r1)));
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
            uint8x16x3_t _bgr = Load3<align>(bgr);
            Store<align>(y, BgrToY(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(u, BgrToU(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(v, BgrToV(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
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
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgr = 0; col < alignedWidth; col += A, colBgr += A3)
                    BgrToYuv444p<align>(bgr + colBgr, y + col, u + col, v + col);
                if (width != alignedWidth)
                {
                    size_t offset = width - A;
                    BgrToYuv444p<false>(bgr + offset * 3, y + offset, u + offset, v + offset);
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
#endif// SIMD_NEON_ENABLE
}
