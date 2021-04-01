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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        void BgraToBgr(const uint8_t *bgra, size_t size, uint8_t *bgr, bool lastRow)
        {
            for (size_t i = (lastRow ? 1 : 0); i < size; ++i, bgr += 3, bgra += 4)
            {
                *(int32_t*)bgr = (*(int32_t*)bgra);
            }
            if (lastRow)
            {
                bgr[0] = bgra[0];
                bgr[1] = bgra[1];
                bgr[2] = bgra[2];
            }
        }

        void BgraToBgr(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *bgr, size_t bgrStride)
        {
            for (size_t row = 1; row < height; ++row)
            {
                BgraToBgr(bgra, width, bgr, false);
                bgr += bgrStride;
                bgra += bgraStride;
            }
            BgraToBgr(bgra, width, bgr, true);
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            size_t bgraGap = bgraStride - width * 4;
            size_t rgbGap = rgbStride - width * 3;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col, bgra += 4, rgb += 3)
                {
                    rgb[2] = bgra[0];
                    rgb[1] = bgra[1];
                    rgb[0] = bgra[2];
                }
                bgra += bgraGap;
                rgb += rgbGap;
            }
        }

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            size_t bgraGap = bgraStride - width * 4;
            size_t rgbaGap = rgbaStride - width * 4;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col, bgra += 4, rgba += 4)
                {
                    rgba[2] = bgra[0];
                    rgba[1] = bgra[1];
                    rgba[0] = bgra[2];
                    rgba[3] = bgra[3];
                }
                bgra += bgraGap;
                rgba += rgbaGap;
            }
        }
    }
}
