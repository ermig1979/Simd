/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        void BgrToBgra(const uint8_t *bgr, size_t size, uint8_t *bgra, bool fillAlpha, bool lastRow, uint8_t alpha)
        {
            if (fillAlpha)
            {
#ifdef SIMD_BIG_ENDIAN
                const int32_t alphaMask = alpha;
#else
                const int32_t alphaMask = alpha << 24;
#endif
                for (size_t i = (lastRow ? 1 : 0); i < size; ++i, bgr += 3, bgra += 4)
                {
                    *(int32_t*)bgra = (*(int32_t*)bgr) | alphaMask;
                }
                if (lastRow)
                {
                    bgra[0] = bgr[0];
                    bgra[1] = bgr[1];
                    bgra[2] = bgr[2];
                    bgra[3] = alpha;
                }
            }
            else
            {
                for (size_t i = (lastRow ? 1 : 0); i < size; ++i, bgr += 3, bgra += 4)
                {
                    *(int32_t*)bgra = (*(int32_t*)bgr);
                }
                if (lastRow)
                {
                    bgra[0] = bgr[0];
                    bgra[1] = bgr[1];
                    bgra[2] = bgr[2];
                }
            }
        }

        void BgrToBgra(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            for (size_t row = 1; row < height; ++row)
            {
                BgrToBgra(bgr, width, bgra, true, false, alpha);
                bgr += bgrStride;
                bgra += bgraStride;
            }
            BgrToBgra(bgr, width, bgra, true, true, alpha);
        }

        void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * pBlue = blue;
                const uint8_t * pGreen = green;
                const uint8_t * pRed = red;
                uint8_t * pBgra = bgra;
                for (size_t col = 0; col < width; ++col)
                {
#ifdef SIMD_BIG_ENDIAN
                    pBgra[0] = pBlue[1];
                    pBgra[1] = pGreen[1];
                    pBgra[2] = pRed[1];
#else
                    pBgra[0] = pBlue[0];
                    pBgra[1] = pGreen[0];
                    pBgra[2] = pRed[0];
#endif
                    pBgra[3] = alpha;
                    pBlue += 2;
                    pGreen += 2;
                    pRed += 2;
                    pBgra += 4;
                }
                blue += blueStride;
                green += greenStride;
                red += redStride;
                bgra += bgraStride;
            }
        }

        void RgbToBgra(const uint8_t * rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            size_t rgbGap = rgbStride - width * 3;
            size_t bgraGap = bgraStride - width * 4;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col, rgb += 3, bgra += 4)
                {
                    bgra[0] = rgb[2];
                    bgra[1] = rgb[1];
                    bgra[2] = rgb[0];
                    bgra[3] = alpha;
                }
                rgb += rgbGap;
                bgra += bgraGap;
            }
        }
    }
}
