/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint16x8_t Average(uint8x16_t a, uint8x16_t b)
        {
            return vshrq_n_u16(vpadalq_u8(vpadalq_u8(K16_0002, a), b), 2);
        }

        template <bool align> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            uint8x16x4_t bgra00 = Load4<align>(bgra0);
            Store<align>(y0 + 0, BgrToY8(bgra00.val[0], bgra00.val[1], bgra00.val[2]));

            uint8x16x4_t bgra01 = Load4<align>(bgra0 + QA);
            Store<align>(y0 + A, BgrToY8(bgra01.val[0], bgra01.val[1], bgra01.val[2]));

            uint8x16x4_t bgra10 = Load4<align>(bgra1);
            Store<align>(y1 + 0, BgrToY8(bgra10.val[0], bgra10.val[1], bgra10.val[2]));

            uint8x16x4_t bgra11 = Load4<align>(bgra1 + QA);
            Store<align>(y1 + A, BgrToY8(bgra11.val[0], bgra11.val[1], bgra11.val[2]));

            uint16x8_t b0 = Average(bgra00.val[0], bgra10.val[0]);
            uint16x8_t g0 = Average(bgra00.val[1], bgra10.val[1]);
            uint16x8_t r0 = Average(bgra00.val[2], bgra10.val[2]);

            uint16x8_t b1 = Average(bgra01.val[0], bgra11.val[0]);
            uint16x8_t g1 = Average(bgra01.val[1], bgra11.val[1]);
            uint16x8_t r1 = Average(bgra01.val[2], bgra11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16(b0, g0, r0), BgrToU16(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16(b0, g0, r0), BgrToV16(b1, g1, r1)));
        }

        template <bool align> void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < alignedWidth; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv420p<align>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuv420p<false>(bgra + offset * 4, bgraStride, y + offset, yStride, u + offset / 2, v + offset / 2);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv420p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv420p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE uint16x8_t Average(uint8x16_t value)
        {
            return vshrq_n_u16(vpadalq_u8(K16_0001, value), 1);
        }

        template <bool align> SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            uint8x16x4_t bgra0 = Load4<align>(bgra);
            Store<align>(y + 0, BgrToY8(bgra0.val[0], bgra0.val[1], bgra0.val[2]));

            uint16x8_t b0 = Average(bgra0.val[0]);
            uint16x8_t g0 = Average(bgra0.val[1]);
            uint16x8_t r0 = Average(bgra0.val[2]);

            uint8x16x4_t bgra1 = Load4<align>(bgra + QA);
            Store<align>(y + A, BgrToY8(bgra1.val[0], bgra1.val[1], bgra1.val[2]));

            uint16x8_t b1 = Average(bgra1.val[0]);
            uint16x8_t g1 = Average(bgra1.val[1]);
            uint16x8_t r1 = Average(bgra1.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16(b0, g0, r0), BgrToU16(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16(b0, g0, r0), BgrToV16(b1, g1, r1)));
        }

        template <bool align> void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < alignedWidth; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv422p<align>(bgra + colBgra, y + colY, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuv422p<false>(bgra + offset * 4, y + offset, u + offset / 2, v + offset / 2);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv422p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv422p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void BgraToYuv444p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            uint8x16x4_t _bgra = Load4<align>(bgra);
            Store<align>(y, BgrToY8(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
            Store<align>(u, BgrToU8(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
            Store<align>(v, BgrToV8(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
        }

        template <bool align> void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < alignedWidth; col += A, colBgra += QA)
                    BgraToYuv444p<align>(bgra + colBgra, y + col, u + col, v + col);
                if (width != alignedWidth)
                {
                    size_t offset = width - A;
                    BgraToYuv444p<false>(bgra + offset * 4, y + offset, u + offset, v + offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv444p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv444p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void BgraToYuva420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v, uint8_t * a0, size_t aStride)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;
            uint8_t * a1 = a0 + aStride;

            uint8x16x4_t bgra00 = Load4<align>(bgra0);
            Store<align>(y0 + 0, BgrToY8(bgra00.val[0], bgra00.val[1], bgra00.val[2]));
            Store<align>(a0 + 0, bgra00.val[3]);

            uint8x16x4_t bgra01 = Load4<align>(bgra0 + QA);
            Store<align>(y0 + A, BgrToY8(bgra01.val[0], bgra01.val[1], bgra01.val[2]));
            Store<align>(a0 + A, bgra01.val[3]);

            uint8x16x4_t bgra10 = Load4<align>(bgra1);
            Store<align>(y1 + 0, BgrToY8(bgra10.val[0], bgra10.val[1], bgra10.val[2]));
            Store<align>(a1 + 0, bgra10.val[3]);

            uint8x16x4_t bgra11 = Load4<align>(bgra1 + QA);
            Store<align>(y1 + A, BgrToY8(bgra11.val[0], bgra11.val[1], bgra11.val[2]));
            Store<align>(a1 + A, bgra11.val[3]);

            uint16x8_t b0 = Average(bgra00.val[0], bgra10.val[0]);
            uint16x8_t g0 = Average(bgra00.val[1], bgra10.val[1]);
            uint16x8_t r0 = Average(bgra00.val[2], bgra10.val[2]);

            uint16x8_t b1 = Average(bgra01.val[0], bgra11.val[0]);
            uint16x8_t g1 = Average(bgra01.val[1], bgra11.val[1]);
            uint16x8_t r1 = Average(bgra01.val[2], bgra11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16(b0, g0, r0), BgrToU16(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16(b0, g0, r0), BgrToV16(b1, g1, r1)));
        }

        template <bool align> void BgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colYA = 0, colBgra = 0; colYA < alignedWidth; colYA += DA, colUV += A, colBgra += A8)
                    BgraToYuva420p<align>(bgra + colBgra, bgraStride, y + colYA, yStride, u + colUV, v + colUV, a + colYA, aStride);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuva420p<false>(bgra + offset * 4, bgraStride, y + offset, yStride, u + offset / 2, v + offset / 2, a + offset, aStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride)
                && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuva420p<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
            else
                BgraToYuva420p<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            uint8x16x4_t bgra00 = Load4<align>(bgra0);
            Store<align>(y0 + 0, BgrToY8<T>(bgra00.val[0], bgra00.val[1], bgra00.val[2]));

            uint8x16x4_t bgra01 = Load4<align>(bgra0 + QA);
            Store<align>(y0 + A, BgrToY8<T>(bgra01.val[0], bgra01.val[1], bgra01.val[2]));

            uint8x16x4_t bgra10 = Load4<align>(bgra1);
            Store<align>(y1 + 0, BgrToY8<T>(bgra10.val[0], bgra10.val[1], bgra10.val[2]));

            uint8x16x4_t bgra11 = Load4<align>(bgra1 + QA);
            Store<align>(y1 + A, BgrToY8<T>(bgra11.val[0], bgra11.val[1], bgra11.val[2]));

            uint16x8_t b0 = Average(bgra00.val[0], bgra10.val[0]);
            uint16x8_t g0 = Average(bgra00.val[1], bgra10.val[1]);
            uint16x8_t r0 = Average(bgra00.val[2], bgra10.val[2]);

            uint16x8_t b1 = Average(bgra01.val[0], bgra11.val[0]);
            uint16x8_t g1 = Average(bgra01.val[1], bgra11.val[1]);
            uint16x8_t r1 = Average(bgra01.val[2], bgra11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((height % 2 == 0) && (width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgraToYuv420pV2<T, align>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgraToYuv420pV2<T, false>(bgra + col * 4, bgraStride, y + col, yStride, u + col / 2, v + col / 2);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Base::Bt601, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Base::Bt709, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Base::Bt2020, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Base::Trect871, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv420pV2<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgraToYuv420pV2<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgraToYuv422pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x4_t bgra0 = Load4<align>(bgra + 0 * A);
            Store<align>(y + 0, BgrToY8<T>(bgra0.val[0], bgra0.val[1], bgra0.val[2]));

            uint16x8_t b0 = Average(bgra0.val[0]);
            uint16x8_t g0 = Average(bgra0.val[1]);
            uint16x8_t r0 = Average(bgra0.val[2]);

            uint8x16x4_t bgra1 = Load4<align>(bgra + 4 * A);
            Store<align>(y + A, BgrToY8<T>(bgra1.val[0], bgra1.val[1], bgra1.val[2]));

            uint16x8_t b1 = Average(bgra1.val[0]);
            uint16x8_t g1 = Average(bgra1.val[1]);
            uint16x8_t r1 = Average(bgra1.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgraToYuv422pV2<T, align>(bgra + colY * 4, y + colY, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgraToYuv422pV2<T, false>(bgra + col * 4, y + col, u + col / 2, v + col/ 2);
                }
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422pV2<Base::Bt601, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv422pV2<Base::Bt709, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv422pV2<Base::Bt2020, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv422pV2<Base::Trect871, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv422pV2<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgraToYuv422pV2<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x4_t _bgra = Load4<align>(bgra);
            Store<align>(y, BgrToY8<T>(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
            Store<align>(u, BgrToU8<T>(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
            Store<align>(v, BgrToV8<T>(_bgra.val[0], _bgra.val[1], _bgra.val[2]));
        }

        template <class T, bool align> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgraToYuv444pV2<T, align>(bgra + col * 4, y + col, u + col, v + col);
                if (width != widthA)
                {
                    size_t col = width - A;
                    BgraToYuv444pV2<T, false>(bgra + col * 4, y + col, u + col, v + col);
                }
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv444pV2<Base::Bt601, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Base::Bt709, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Base::Bt2020, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Base::Trect871, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv444pV2<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgraToYuv444pV2<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgraToYuva420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, uint8_t* a0, size_t aStride)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;
            uint8_t* a1 = a0 + yStride;

            uint8x16x4_t bgra00 = Load4<align>(bgra0);
            Store<align>(y0 + 0, BgrToY8<T>(bgra00.val[0], bgra00.val[1], bgra00.val[2]));
            Store<align>(a0 + 0, bgra00.val[3]);

            uint8x16x4_t bgra01 = Load4<align>(bgra0 + QA);
            Store<align>(y0 + A, BgrToY8<T>(bgra01.val[0], bgra01.val[1], bgra01.val[2]));
            Store<align>(a0 + A, bgra01.val[3]);

            uint8x16x4_t bgra10 = Load4<align>(bgra1);
            Store<align>(y1 + 0, BgrToY8<T>(bgra10.val[0], bgra10.val[1], bgra10.val[2]));
            Store<align>(a1 + 0, bgra10.val[3]);

            uint8x16x4_t bgra11 = Load4<align>(bgra1 + QA);
            Store<align>(y1 + A, BgrToY8<T>(bgra11.val[0], bgra11.val[1], bgra11.val[2]));
            Store<align>(a1 + A, bgra11.val[3]);

            uint16x8_t b0 = Average(bgra00.val[0], bgra10.val[0]);
            uint16x8_t g0 = Average(bgra00.val[1], bgra10.val[1]);
            uint16x8_t r0 = Average(bgra00.val[2], bgra10.val[2]);

            uint16x8_t b1 = Average(bgra01.val[0], bgra11.val[0]);
            uint16x8_t g1 = Average(bgra01.val[1], bgra11.val[1]);
            uint16x8_t r1 = Average(bgra01.val[2], bgra11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride)
        {
            assert((height % 2 == 0) && (width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
                assert(Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgraToYuva420pV2<T, align>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colUV, v + colUV, a + colY, aStride);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgraToYuva420pV2<T, false>(bgra + col * 4, bgraStride, y + col, yStride, u + col / 2, v + col / 2, a + col, aStride);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
            }
        }

        template <bool align> void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuva420pV2<Base::Bt601, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt709: BgraToYuva420pV2<Base::Bt709, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt2020: BgraToYuva420pV2<Base::Bt2020, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvTrect871: BgraToYuva420pV2<Base::Trect871, align>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            default:
                assert(0);
            }
        }

        void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && 
                Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuva420pV2<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride, yuvType);
            else
                BgraToYuva420pV2<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            uint8x16x3_t bgr00 = Load3<align>(bgr0 + 0 * A);
            Store<align>(y0 + 0, BgrToY8<T>(bgr00.val[0], bgr00.val[1], bgr00.val[2]));

            uint8x16x3_t bgr01 = Load3<align>(bgr0 + 3 * A);
            Store<align>(y0 + A, BgrToY8<T>(bgr01.val[0], bgr01.val[1], bgr01.val[2]));

            uint8x16x3_t bgr10 = Load3<align>(bgr1 + 0 * A);
            Store<align>(y1 + 0, BgrToY8<T>(bgr10.val[0], bgr10.val[1], bgr10.val[2]));

            uint8x16x3_t bgr11 = Load3<align>(bgr1 + 3 * A);
            Store<align>(y1 + A, BgrToY8<T>(bgr11.val[0], bgr11.val[1], bgr11.val[2]));

            uint16x8_t b0 = Average(bgr00.val[0], bgr10.val[0]);
            uint16x8_t g0 = Average(bgr00.val[1], bgr10.val[1]);
            uint16x8_t r0 = Average(bgr00.val[2], bgr10.val[2]);

            uint16x8_t b1 = Average(bgr01.val[0], bgr11.val[0]);
            uint16x8_t g1 = Average(bgr01.val[1], bgr11.val[1]);
            uint16x8_t r1 = Average(bgr01.val[2], bgr11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((height % 2 == 0) && (width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgrToYuv420pV2<T, align>(bgr + colY * 3, bgrStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgrToYuv420pV2<T, false>(bgr + col * 3, bgrStride, y + col, yStride, u + col / 2, v + col / 2);
                }
                bgr += 2 * bgrStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv420pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv420pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv422pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x3_t bgr0 = Load3<align>(bgr + 0 * A);
            Store<align>(y + 0, BgrToY8<T>(bgr0.val[0], bgr0.val[1], bgr0.val[2]));

            uint16x8_t b0 = Average(bgr0.val[0]);
            uint16x8_t g0 = Average(bgr0.val[1]);
            uint16x8_t r0 = Average(bgr0.val[2]);

            uint8x16x4_t bgr1 = Load4<align>(bgr + 3 * A);
            Store<align>(y + A, BgrToY8<T>(bgr1.val[0], bgr1.val[1], bgr1.val[2]));

            uint16x8_t b1 = Average(bgr1.val[0]);
            uint16x8_t g1 = Average(bgr1.val[1]);
            uint16x8_t r1 = Average(bgr1.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgrToYuv422pV2<T, align>(bgr + colY * 3, y + colY, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgrToYuv422pV2<T, false>(bgr + col * 3, y + col, u + col / 2, v + col / 2);
                }
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv422pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv422pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv422pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv422pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv422pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv444pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x3_t _bgr = Load3<align>(bgr);
            Store<align>(y, BgrToY8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(u, BgrToU8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(v, BgrToV8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
        }

        template <class T, bool align> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgrToYuv444pV2<T, align>(bgr + col * 3, y + col, u + col, v + col);
                if (width != widthA)
                {
                    size_t col = width - A;
                    BgrToYuv444pV2<T, false>(bgr + col * 3, y + col, u + col, v + col);
                }
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv444pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv444pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv444pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv444pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv444pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv444pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }
    }
#endif// SIMD_NEON_ENABLE
}
