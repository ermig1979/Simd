/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        void BgrToBgra(const uint8_t *bgr, size_t size, uint8_t *bgra, bool fillAlpha, bool lastRow, uint8_t alpha)
        {
            if(fillAlpha)
            {
				const int32_t alphaMask = alpha << 24;
                for(size_t i = (lastRow ? 1 : 0); i < size; ++i, bgr += 3, bgra += 4)
                {
                    *(int32_t*)bgra = (*(int32_t*)bgr) | alphaMask;
                }
                if(lastRow)
                {
                    bgra[0] = bgr[0];
                    bgra[1] = bgr[1];
                    bgra[2] = bgr[2];
                    bgra[3] = alpha;
                }
            }
            else
            {
                for(size_t i = (lastRow ? 1 : 0); i < size; ++i, bgr += 3, bgra += 4)
                {
                    *(int32_t*)bgra = (*(int32_t*)bgr);
                }
                if(lastRow)
                {
                    bgra[0] = bgr[0];
                    bgra[1] = bgr[1];
                    bgra[2] = bgr[2];
                }
            }
        }

        void BgrToBgra(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            for(size_t row = 1; row < height; ++row)
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
            for(size_t row = 0; row < height; ++row)
            {
                const uint8_t * pBlue = blue;
                const uint8_t * pGreen = green;
                const uint8_t * pRed = red;
                uint8_t * pBgra = bgra;
                for(size_t col = 0; col < width; ++col)
                {
                    pBgra[0] = *pBlue;
                    pBgra[1] = *pGreen;
                    pBgra[2] = *pRed;
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
    }
}