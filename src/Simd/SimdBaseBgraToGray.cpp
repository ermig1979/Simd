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

namespace Simd
{
    namespace Base
    {
        void BgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * pBgra = bgra + row*bgraStride;
                uint8_t * pGray = gray + row*grayStride;
                for (const uint8_t *pGrayEnd = pGray + width; pGray < pGrayEnd; pGray += 1, pBgra += 4)
                {
                    *pGray = BgrToGray(pBgra[0], pBgra[1], pBgra[2]);
                }
            }
        }

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pRgba = rgba + row * rgbaStride;
                uint8_t* pGray = gray + row * grayStride;
                for (const uint8_t* pGrayEnd = pGray + width; pGray < pGrayEnd; pGray += 1, pRgba += 4)
                {
                    *pGray = BgrToGray(pRgba[2], pRgba[1], pRgba[0]);
                }
            }
        }
    }
}
