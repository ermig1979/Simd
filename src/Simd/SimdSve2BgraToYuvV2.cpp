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
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template <class T> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v, const svbool_t& mask)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svuint8_t blue = svget4(_bgra, 0);
            svuint8_t green = svget4(_bgra, 1);
            svuint8_t red = svget4(_bgra, 2);
            svst1_u8(mask, y, BgrToY8<T>(blue, green, red));
            svst1_u8(mask, u, BgrToU8<T>(blue, green, red));
            svst1_u8(mask, v, BgrToV8<T>(blue, green, red));
        }

        template <class T> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgra = 0, col = 0;
                for (; col < widthA; colBgra += A4, col += A)
                    BgraToYuv444pV2<T>(bgra + colBgra, y + col, u + col, v + col, body);
                if (col < width)
                    BgraToYuv444pV2<T>(bgra + colBgra, y + col, u + col, v + col, tail);
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv444pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        template <class T> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride,
            uint8_t* u, uint8_t* v, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x4_t bgra00 = svld4_u8(maskY, bgra0);
            svuint8x4_t bgra10 = svld4_u8(maskY, bgra1);

            svst1_u8(maskY, y0, BgrToY8<T>(svget4(bgra00, 0), svget4(bgra00, 1), svget4(bgra00, 2)));
            svst1_u8(maskY, y1, BgrToY8<T>(svget4(bgra10, 0), svget4(bgra10, 1), svget4(bgra10, 2)));

            svuint16_t blue = AverageUv(svget4(bgra00, 0), svget4(bgra10, 0), maskY);
            svuint16_t green = AverageUv(svget4(bgra00, 1), svget4(bgra10, 1), maskY);
            svuint16_t red = AverageUv(svget4(bgra00, 2), svget4(bgra10, 2), maskY);

            svst1_u8(maskUv, u, PackSeqI16ToU8(BgrToU16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
            svst1_u8(maskUv, v, PackSeqI16ToU8(BgrToV16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
        }

        template <class T> void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, tailY, tailUv);
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgraToYuva420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride,
            uint8_t* u, uint8_t* v, uint8_t* a0, size_t aStride, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;
            uint8_t* a1 = a0 + aStride;

            svuint8x4_t bgra00 = svld4_u8(maskY, bgra0);
            svuint8x4_t bgra10 = svld4_u8(maskY, bgra1);

            svst1_u8(maskY, y0, BgrToY8<T>(svget4(bgra00, 0), svget4(bgra00, 1), svget4(bgra00, 2)));
            svst1_u8(maskY, y1, BgrToY8<T>(svget4(bgra10, 0), svget4(bgra10, 1), svget4(bgra10, 2)));
            svst1_u8(maskY, a0, svget4(bgra00, 3));
            svst1_u8(maskY, a1, svget4(bgra10, 3));

            svuint16_t blue = AverageUv(svget4(bgra00, 0), svget4(bgra10, 0), maskY);
            svuint16_t green = AverageUv(svget4(bgra00, 1), svget4(bgra10, 1), maskY);
            svuint16_t red = AverageUv(svget4(bgra00, 2), svget4(bgra10, 2), maskY);

            svst1_u8(maskUv, u, PackSeqI16ToU8(BgrToU16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
            svst1_u8(maskUv, v, PackSeqI16ToU8(BgrToV16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
        }

        template <class T> void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    BgraToYuva420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, a + colY, aStride, bodyY, bodyUv);
                if (colY < width)
                    BgraToYuva420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, a + colY, aStride, tailY, tailUv);
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
            }
        }

        void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuva420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt709: BgraToYuva420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt2020: BgraToYuva420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvTrect871: BgraToYuva420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgraToYuv422pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v,
            const svbool_t& maskY, const svbool_t& maskUv)
        {
            svuint8x4_t _bgra = svld4_u8(maskY, bgra);

            svst1_u8(maskY, y, BgrToY8<T>(svget4(_bgra, 0), svget4(_bgra, 1), svget4(_bgra, 2)));

            svuint16_t blue = AverageUv(svget4(_bgra, 0), maskY);
            svuint16_t green = AverageUv(svget4(_bgra, 1), maskY);
            svuint16_t red = AverageUv(svget4(_bgra, 2), maskY);

            svst1_u8(maskUv, u, PackSeqI16ToU8(BgrToU16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
            svst1_u8(maskUv, v, PackSeqI16ToU8(BgrToV16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()));
        }

        template <class T> void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUv, v + colUv, tailY, tailUv);
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv422pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv422pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv422pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
