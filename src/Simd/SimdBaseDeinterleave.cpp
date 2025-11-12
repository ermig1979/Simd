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
        template<int U, int V> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += 2)
                {
                    if (U) u[col] = uv[offset];
                    if (V) v[col] = uv[offset + 1];
                }
                uv += uvStride;
                if (U) u += uStride;
                if (V) v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t* uv, size_t uvStride, size_t width, size_t height,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            if (u && v)
                DeinterleaveUv<1, 1>(uv, uvStride, width, height, u, uStride, v, vStride);
            else if(u)
                DeinterleaveUv<1, 0>(uv, uvStride, width, height, u, uStride, v, vStride);
            else if (v)
                DeinterleaveUv<0, 1>(uv, uvStride, width, height, u, uStride, v, vStride);
        }

        //------------------------------------------------------------------------------------------------

        template<int B, int G, int R> void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += 3)
                {
                    if (B) b[col] = bgr[offset + 0];
                    if (G) g[col] = bgr[offset + 1];
                    if (R) r[col] = bgr[offset + 2];
                }
                bgr += bgrStride;
                if (B) b += bStride;
                if (G) g += gStride;
                if (R) r += rStride;
            }
        }

        void DeinterleaveBgr(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride)
        {
            if (b && g && r)
                DeinterleaveBgr<1, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && g)
                DeinterleaveBgr<1, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && r)
                DeinterleaveBgr<1, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g && r)
                DeinterleaveBgr<0, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b)
                DeinterleaveBgr<1, 0, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g)
                DeinterleaveBgr<0, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (r)
                DeinterleaveBgr<0, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
        }

        //------------------------------------------------------------------------------------------------

        template<int B, int G, int R, int A> void DeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += 4)
                {
                    if (B) b[col] = bgra[offset + 0];
                    if (G) g[col] = bgra[offset + 1];
                    if (R) r[col] = bgra[offset + 2];
                    if (A) a[col] = bgra[offset + 3];
                }
                bgra += bgraStride;
                if (B) b += bStride;
                if (G) g += gStride;
                if (R) r += rStride;
                if (A) a += aStride;
            }
        }

        void DeinterleaveBgra(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride, uint8_t* a, size_t aStride)
        {
            if (b && g && r && a)
                DeinterleaveBgra<1, 1, 1, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if(b && g && r)
                DeinterleaveBgra<1, 1, 1, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b && g && a)
                DeinterleaveBgra<1, 1, 0, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b && r && a)
                DeinterleaveBgra<1, 0, 1, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (g && r && a)
                DeinterleaveBgra<0, 1, 1, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b && g)
                DeinterleaveBgra<1, 1, 0, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b && r)
                DeinterleaveBgra<1, 0, 1, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b && a)
                DeinterleaveBgra<1, 0, 0, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (g && r)
                DeinterleaveBgra<0, 1, 1, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (g && a)
                DeinterleaveBgra<0, 1, 0, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (r && a)
                DeinterleaveBgra<0, 0, 1, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (b)
                DeinterleaveBgra<1, 0, 0, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (g)
                DeinterleaveBgra<0, 1, 0, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (r)
                DeinterleaveBgra<0, 0, 1, 0>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else if (a)
                DeinterleaveBgra<0, 0, 0, 1>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
        }
    }
}
