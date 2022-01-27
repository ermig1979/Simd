/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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
#include "Simd/SimdConversion.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            y0[0] = BgrToY(bgra0[0], bgra0[1], bgra0[2]);
            y0[1] = BgrToY(bgra0[4], bgra0[5], bgra0[6]);
            y1[0] = BgrToY(bgra1[0], bgra1[1], bgra1[2]);
            y1[1] = BgrToY(bgra1[4], bgra1[5], bgra1[6]);

            int blue = Average(bgra0[0], bgra0[4], bgra1[0], bgra1[4]);
            int green = Average(bgra0[1], bgra0[5], bgra1[1], bgra1[5]);
            int red = Average(bgra0[2], bgra0[6], bgra1[2], bgra1[6]);

            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                    BgraToYuv420p(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            y[0] = BgrToY(bgra[0], bgra[1], bgra[2]);
            y[1] = BgrToY(bgra[4], bgra[5], bgra[6]);

            int blue = Average(bgra[0], bgra[4]);
            int green = Average(bgra[1], bgra[5]);
            int red = Average(bgra[2], bgra[6]);

            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                    BgraToYuv422p(bgra + colBgra, y + colY, u + colUV, v + colUV);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        SIMD_INLINE void BgraToYuv444p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            const int blue = bgra[0], green = bgra[1], red = bgra[2];
            y[0] = BgrToY(blue, green, red);
            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < width; ++col, colBgra += 4)
                    BgraToYuv444p(bgra + colBgra, y + col, u + col, v + col);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        SIMD_INLINE void BgraToYuva420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v, uint8_t * a0, size_t aStride)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;
            uint8_t * a1 = a0 + aStride;

            y0[0] = BgrToY(bgra0[0], bgra0[1], bgra0[2]);
            y0[1] = BgrToY(bgra0[4], bgra0[5], bgra0[6]);
            y1[0] = BgrToY(bgra1[0], bgra1[1], bgra1[2]);
            y1[1] = BgrToY(bgra1[4], bgra1[5], bgra1[6]);

            int blue = Average(bgra0[0], bgra0[4], bgra1[0], bgra1[4]);
            int green = Average(bgra0[1], bgra0[5], bgra1[1], bgra1[5]);
            int red = Average(bgra0[2], bgra0[6], bgra1[2], bgra1[6]);

            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);

            a0[0] = bgra0[3];
            a0[1] = bgra0[7];
            a1[0] = bgra1[3];
            a1[1] = bgra1[7];
        }

        void BgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colYA = 0, colBgra = 0; colYA < width; colYA += 2, colUV++, colBgra += 8)
                    BgraToYuva420p(bgra + colBgra, bgraStride, y + colYA, yStride, u + colUV, v + colUV, a + colYA, aStride);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            const int blue = bgra[0], green = bgra[1], red = bgra[2];
            y[0] = BgrToY<YuvType>(blue, green, red);
            u[0] = BgrToU<YuvType>(blue, green, red);
            v[0] = BgrToV<YuvType>(blue, green, red);
        }

        template <class YuvType> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < width; ++col, colBgra += 4)
                    BgraToYuv444pV2<YuvType>(bgra + colBgra, y + col, u + col, v + col);
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
            case SimdYuvBt601: BgraToYuv444pV2<Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            y0[0] = BgrToY<YuvType>(bgra0[0], bgra0[1], bgra0[2]);
            y0[1] = BgrToY<YuvType>(bgra0[4], bgra0[5], bgra0[6]);
            y1[0] = BgrToY<YuvType>(bgra1[0], bgra1[1], bgra1[2]);
            y1[1] = BgrToY<YuvType>(bgra1[4], bgra1[5], bgra1[6]);

            int blue = Average(bgra0[0], bgra0[4], bgra1[0], bgra1[4]);
            int green = Average(bgra0[1], bgra0[5], bgra1[1], bgra1[5]);
            int red = Average(bgra0[2], bgra0[6], bgra1[2], bgra1[6]);

            u[0] = BgrToU<YuvType>(blue, green, red);
            v[0] = BgrToV<YuvType>(blue, green, red);
        }

        template <class YuvType> void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                    BgraToYuv420pV2<YuvType>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
}
