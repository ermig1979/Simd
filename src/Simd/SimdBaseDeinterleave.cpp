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
        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += 2)
                {
                    u[col] = uv[offset];
                    v[col] = uv[offset + 1];
                }
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += 3)
                {
                    b[col] = bgr[offset + 0];
                    g[col] = bgr[offset + 1];
                    r[col] = bgr[offset + 2];
                }
                bgr += bgrStride;
                b += bStride;
                g += gStride;
                r += rStride;
            }
        }

        void DeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
        {
            if (a)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0, offset = 0; col < width; ++col, offset += 4)
                    {
                        b[col] = bgra[offset + 0];
                        g[col] = bgra[offset + 1];
                        r[col] = bgra[offset + 2];
                        a[col] = bgra[offset + 3];
                    }
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                    a += aStride;
                }
            }
            else
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0, offset = 0; col < width; ++col, offset += 4)
                    {
                        b[col] = bgra[offset + 0];
                        g[col] = bgra[offset + 1];
                        r[col] = bgra[offset + 2];
                    }
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                }
            }
        }
    }
}
