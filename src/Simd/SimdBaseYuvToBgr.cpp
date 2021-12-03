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
#include "Simd/SimdConversion.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void Yuv422pToBgr(const uint8_t *y, int u, int v, uint8_t * bgr)
        {
            YuvToBgr(y[0], u, v, bgr);
            YuvToBgr(y[1], u, v, bgr + 3);
        }

        void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                {
                    int u_ = u[colUV];
                    int v_ = v[colUV];
                    Yuv422pToBgr(y + colY, u_, v_, bgr + colBgr);
                    Yuv422pToBgr(y + yStride + colY, u_, v_, bgr + bgrStride + colBgr);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void Yuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < width; colY += 2, colUV++, colBgr += 6)
                    Yuv422pToBgr(y + colY, u[colUV], v[colUV], bgr + colBgr);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgr = 0; col < width; col++, colBgr += 3)
                    YuvToBgr(y[col], u[col], v[col], bgr + colBgr);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void Yuv422pToRgb(const uint8_t* y, int u, int v, uint8_t* rgb)
        {
            YuvToRgb(y[0], u, v, rgb);
            YuvToRgb(y[1], u, v, rgb + 3);
        }

        void Yuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < width; colY += 2, colUV++, colRgb += 6)
                {
                    int u_ = u[colUV];
                    int v_ = v[colUV];
                    Yuv422pToRgb(y + colY, u_, v_, rgb + colRgb);
                    Yuv422pToRgb(y + yStride + colY, u_, v_, rgb + rgbStride + colRgb);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                rgb += 2 * rgbStride;
            }
        }

        void Yuv422pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < width; colY += 2, colUV++, colRgb += 6)
                    Yuv422pToRgb(y + colY, u[colUV], v[colUV], rgb + colRgb);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv444pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colRgb = 0; col < width; col++, colRgb += 3)
                    YuvToRgb(y[col], u[col], v[col], rgb + colRgb);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }
    }
}
