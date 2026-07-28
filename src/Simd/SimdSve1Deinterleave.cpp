/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE    
    namespace Sve
    {
        template <int B, int G, int R> void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                {
                    svuint8x3_t _bgr = svld3_u8(body, bgr + offset);
                    if (B) svst1_u8(body, b + col, svget3(_bgr, 0));
                    if (G) svst1_u8(body, g + col, svget3(_bgr, 1));
                    if (R) svst1_u8(body, r + col, svget3(_bgr, 2));
                }
                if (widthA < width)
                {
                    svuint8x3_t _bgr = svld3_u8(tail, bgr + offset);
                    if (B) svst1_u8(tail, b + col, svget3(_bgr, 0));
                    if (G) svst1_u8(tail, g + col, svget3(_bgr, 1));
                    if (R) svst1_u8(tail, r + col, svget3(_bgr, 2));
                }
                bgr += bgrStride;
                if (B) b += bStride;
                if (G) g += gStride;
                if (R) r += rStride;
            }
        }

        void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            if (b && g && r) DeinterleaveBgr<1, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && g) DeinterleaveBgr<1, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && r) DeinterleaveBgr<1, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g && r) DeinterleaveBgr<0, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b) DeinterleaveBgr<1, 0, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g) DeinterleaveBgr<0, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (r) DeinterleaveBgr<0, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
        }
    }
#endif
}
