/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
    namespace Base
    {
        template<int N> void DrawLine(uint8_t* canvas, size_t stride, size_t width, size_t height, int x1, int y1, int x2, int y2, const uint8_t* color, int lineWidth)
        {
            int w = (int)width - 1;
            int h = (int)height - 1;

            if (x1 < 0 || y1 < 0 || x1 > w || y1 > h || x2 < 0 || y2 < 0 || x2 > w || y2 > h)
            {
                if ((x1 < 0 && x2 < 0) || (y1 < 0 && y2 < 0) || (x1 > w && x2 > w) || (y1 > h && y2 > h))
                    return;

                if (y1 == y2)
                {
                    x1 = Simd::RestrictRange(x1, 0, w);
                    x2 = Simd::RestrictRange(x2, 0, w);
                }
                else if (x1 == x2)
                {
                    y1 = Simd::RestrictRange(y1, 0, h);
                    y2 = Simd::RestrictRange(y2, 0, h);
                }
                else
                {
                    int x0 = (x1 * y2 - y1 * x2) / (y2 - y1);
                    int y0 = (y1 * x2 - x1 * y2) / (x2 - x1);
                    int xh = (x1 * y2 - y1 * x2 + h * (x2 - x1)) / (y2 - y1);
                    int yw = (y1 * x2 - x1 * y2 + w * (y2 - y1)) / (x2 - x1);

                    if ((x0 < 0 && xh < 0) || (x0 > w && xh > w) || (y0 < 0 && yw < 0) || (y0 > h && yw > h))
                        return;

                    if (x1 < 0)
                    {
                        x1 = 0;
                        y1 = y0;
                    }
                    if (x2 < 0)
                    {
                        x2 = 0;
                        y2 = y0;
                    }

                    if (x1 > w)
                    {
                        x1 = w;
                        y1 = yw;
                    }
                    if (x2 > w)
                    {
                        x2 = w;
                        y2 = yw;
                    }

                    if (y1 < 0)
                    {
                        x1 = x0;
                        y1 = 0;
                    }
                    if (y2 < 0)
                    {
                        x2 = x0;
                        y2 = 0;
                    }

                    if (y1 > h)
                    {
                        x1 = xh;
                        y1 = h;
                    }
                    if (y2 > h)
                    {
                        x2 = xh;
                        y2 = h;
                    }
                }
            }

            const bool inverse = std::abs(y2 - y1) > std::abs(x2 - x1);
            if (inverse)
            {
                Simd::Swap(x1, y1);
                Simd::Swap(x2, y2);
            }

            if (x1 > x2)
            {
                Simd::Swap(x1, x2);
                Simd::Swap(y1, y2);
            }

            const float dx = float(x2 - x1);
            const float dy = (float)std::abs(y2 - y1);

            float error = dx / 2.0f;
            const int yStep = (y1 < y2) ? 1 : -1;
            int y0 = y1 - lineWidth / 2;

            for (int x = x1; x <= x2; x++)
            {
                for (int i = 0; i < lineWidth; ++i)
                {
                    int y = y0 + i;
                    if (y >= 0)
                    {
                        if (inverse)
                        {
                            if (y <= w)
                            {
                                assert(y >= 0 && y <= w && x >= 0 && x <= h);
                                CopyPixel<N>(color, canvas + x * stride + y * N);
                            }
                        }
                        else
                        {
                            if (y <= h)
                            {
                                assert(y >= 0 && y <= h && x >= 0 && x <= w);
                                CopyPixel<N>(color, canvas + y * stride + x * N);
                            }
                        }
                    }
                }

                error -= dy;
                if (error < 0)
                {
                    y0 += yStep;
                    error += dx;
                }
            }
        }

        void DrawLine(uint8_t* canvas, size_t stride, size_t width, size_t height, size_t channels, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const uint8_t* color, size_t lineWidth)
        {
            switch (channels)
            {
            case 1: DrawLine<1>(canvas, stride, width, height, (int)x1, (int)y1, (int)x2, (int)y2, color, (int)lineWidth); break;
            case 2: DrawLine<2>(canvas, stride, width, height, (int)x1, (int)y1, (int)x2, (int)y2, color, (int)lineWidth); break;
            case 3: DrawLine<3>(canvas, stride, width, height, (int)x1, (int)y1, (int)x2, (int)y2, color, (int)lineWidth); break;
            case 4: DrawLine<4>(canvas, stride, width, height, (int)x1, (int)y1, (int)x2, (int)y2, color, (int)lineWidth); break;
            }
        }

        //--------------------------------------------------------------------------------------------------

        void DrawRectangle(uint8_t* canvas, size_t stride, size_t width, size_t height, size_t channels, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, const uint8_t* color, size_t lineWidth)
        {
            DrawLine(canvas, stride, width, height, channels, left, top, right, top, color, lineWidth);
            DrawLine(canvas, stride, width, height, channels, right, top, right, bottom, color, lineWidth);
            DrawLine(canvas, stride, width, height, channels, right, bottom, left, bottom, color, lineWidth);
            DrawLine(canvas, stride, width, height, channels, left, bottom, left, top, color, lineWidth);
        }
    }
}
