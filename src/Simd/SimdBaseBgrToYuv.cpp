/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar,
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

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void BgrToYuv420p(const uint8_t * bgr0, size_t bgrStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgr1 = bgr0 + bgrStride;
            uint8_t * y1 = y0 + yStride;

            y0[0] = BgrToY(bgr0[0], bgr0[1], bgr0[2]);
            y0[1] = BgrToY(bgr0[3], bgr0[4], bgr0[5]);
            y1[0] = BgrToY(bgr1[0], bgr1[1], bgr1[2]);
            y1[1] = BgrToY(bgr1[3], bgr1[4], bgr1[5]);

            int blue = Average(bgr0[0], bgr0[3], bgr1[0], bgr1[3]);
            int green = Average(bgr0[1], bgr0[4], bgr1[1], bgr1[4]);
            int red = Average(bgr0[2], bgr0[5], bgr1[2], bgr1[5]);

            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                {
                    BgrToYuv420p(bgr + colBgr, bgrStride, y + colY, yStride, u + colUV, v + colUV);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        SIMD_INLINE void BgrToYuv422p(const uint8_t * bgr, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            y[0] = BgrToY(bgr[0], bgr[1], bgr[2]);
            y[1] = BgrToY(bgr[3], bgr[4], bgr[5]);

            int blue = Average(bgr[0], bgr[3]);
            int green = Average(bgr[1], bgr[4]);
            int red = Average(bgr[2], bgr[5]);

            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                    BgrToYuv422p(bgr + colBgr, y + colY, u + colUV, v + colUV);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        SIMD_INLINE void BgrToYuv444p(const uint8_t * bgr, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            const int blue = bgr[0], green = bgr[1], red = bgr[2];
            y[0] = BgrToY(blue, green, red);
            u[0] = BgrToU(blue, green, red);
            v[0] = BgrToV(blue, green, red);
        }

        void BgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgr = 0; col < width; ++col, colBgr += 3)
                    BgrToYuv444p(bgr + colBgr, y + col, u + col, v + col);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }
    }
}
