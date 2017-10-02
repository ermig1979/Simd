/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
        SIMD_INLINE int YuvToHue(int y, int u, int v)
        {
            int red = YuvToRed(y, v);
            int green = YuvToGreen(y, u, v);
            int blue = YuvToBlue(y, u);

            int max = Max(red, Max(green, blue));
            int min = Min(red, Min(green, blue));
            int range = max - min;

            if (range)
            {
                int dividend;

                if (red == max)
                    dividend = green - blue + 6 * range;
                else if (green == max)
                    dividend = blue - red + 2 * range;
                else
                    dividend = red - green + 4 * range;

                return int(KF_255_DIV_6*float(dividend) / float(range)
#if defined(_MSC_VER)
                    +0.00001f
#endif
                    );
            }
            return 0;
        }

        void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col1 = 0, col2 = 0; col2 < width; col2 += 2, col1++)
                {
                    int u_ = u[col1];
                    int v_ = v[col1];
                    hue[col2] = YuvToHue(y[col2], u_, v_);
                    hue[col2 + 1] = YuvToHue(y[col2 + 1], u_, v_);
                    hue[col2 + hueStride] = YuvToHue(y[col2 + yStride], u_, v_);
                    hue[col2 + hueStride + 1] = YuvToHue(y[col2 + yStride + 1], u_, v_);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                hue += 2 * hueStride;
            }
        }

        void Yuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    hue[col] = YuvToHue(y[col], u[col], v[col]);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hue += hueStride;
            }
        }
    }
}
