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
        SIMD_INLINE void Yuva422pToBgra(const uint8_t * y, int u, int v, const uint8_t * a, uint8_t * bgra)
        {
            YuvToBgra(y[0], u, v, a[0], bgra + 0);
            YuvToBgra(y[1], u, v, a[1], bgra + 4);
        }

        void Yuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                {
                    int u_ = u[colUV];
                    int v_ = v[colUV];
                    Yuva422pToBgra(y + colY, u_, v_, a + colY, bgra + colBgra);
                    Yuva422pToBgra(y + yStride + colY, u_, v_, a + aStride + colY, bgra + bgraStride + colBgra);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        SIMD_INLINE void Yuv422pToBgra(const uint8_t *y, int u, int v, int alpha, uint8_t * bgra)
        {
            YuvToBgra(y[0], u, v, alpha, bgra + 0);
            YuvToBgra(y[1], u, v, alpha, bgra + 4);
        }

        void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                {
                    int u_ = u[colUV];
                    int v_ = v[colUV];
                    Yuv422pToBgra(y + colY, u_, v_, alpha, bgra + colBgra);
                    Yuv422pToBgra(y + yStride + colY, u_, v_, alpha, bgra + bgraStride + colBgra);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (width >= 2));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                    Yuv422pToBgra(y + colY, u[colUV], v[colUV], alpha, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < width; col++, colBgra += 4)
                    YuvToBgra(y[col], u[col], v[col], alpha, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <class YuvType> SIMD_INLINE void Yuv422pToBgra(const uint8_t* y, int u, int v, int alpha, uint8_t* bgra)
        {
            YuvToBgra<YuvType>(y[0], u, v, alpha, bgra + 0);
            YuvToBgra<YuvType>(y[1], u, v, alpha, bgra + 4);
        }

        template <class YuvType> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                {
                    int _u = u[colUV];
                    int _v = v[colUV];
                    Yuv422pToBgra<YuvType>(y + colY, _u, _v, alpha, bgra + colBgra);
                    Yuv422pToBgra<YuvType>(y + yStride + colY, _u, _v, alpha, bgra + bgraStride + colBgra);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgraV2<Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv420pToBgraV2<Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv420pToBgraV2<Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv420pToBgraV2<Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        template <class YuvType> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < width; col++, colBgra += 4)
                    YuvToBgra<YuvType>(y[col], u[col], v[col], alpha, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgraV2<Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv444pToBgraV2<Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToBgraV2<Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToBgraV2<Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }
    }
}
