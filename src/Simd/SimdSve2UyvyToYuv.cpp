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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void Uyvy422ToYuv420p(const uint8_t* uyvy0, size_t uyvyStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, const svbool_t& mask)
        {
            size_t A = svcntb();
            const uint8_t* uyvy1 = uyvy0 + uyvyStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x4_t uyvy00 = svld4_u8(mask, uyvy0);
            svuint8x4_t uyvy10 = svld4_u8(mask, uyvy1);
            svuint8_t y00 = svget4(uyvy00, 1);
            svuint8_t y01 = svget4(uyvy00, 3);
            svuint8_t y10 = svget4(uyvy10, 1);
            svuint8_t y11 = svget4(uyvy10, 3);

            svst1_u8(mask, y0 + 0 * A, svzip1_u8(y00, y01));
            svst1_u8(mask, y0 + 1 * A, svzip2_u8(y00, y01));
            svst1_u8(mask, y1 + 0 * A, svzip1_u8(y10, y11));
            svst1_u8(mask, y1 + 1 * A, svzip2_u8(y10, y11));
            svst1_u8(mask, u, svrhadd_u8_x(mask, svget4(uyvy00, 0), svget4(uyvy10, 0)));
            svst1_u8(mask, v, svrhadd_u8_x(mask, svget4(uyvy00, 2), svget4(uyvy10, 2)));
        }

        SIMD_INLINE void Uyvy422ToYuv420p(const uint8_t* uyvy0, size_t uyvyStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* uyvy1 = uyvy0 + uyvyStride;
            uint8_t* y1 = y0 + yStride;
            y0[0] = uyvy0[1];
            y0[1] = uyvy0[3];
            y1[0] = uyvy1[1];
            y1[1] = uyvy1[3];
            u[0] = (uyvy0[0] + uyvy1[0] + 1) / 2;
            v[0] = (uyvy0[2] + uyvy1[2] + 1) / 2;
        }

        void Uyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            size_t A = svcntb(), DA = 2 * A, A4 = 4 * A, widthDA = AlignLo(width, DA);
            const svbool_t mask = svptrue_b8();
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0, colUyvy = 0, colY = 0, colUV = 0;
                for (; col < widthDA; col += DA, colUyvy += A4, colY += DA, colUV += A)
                    Uyvy422ToYuv420p(uyvy + colUyvy, uyvyStride, y + colY, yStride, u + colUV, v + colUV, mask);
                for (; col < width; col += 2, colUyvy += 4, colY += 2, colUV += 1)
                    Uyvy422ToYuv420p(uyvy + colUyvy, uyvyStride, y + colY, yStride, u + colUV, v + colUV);
                uyvy += 2 * uyvyStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }
    }
#endif
}
