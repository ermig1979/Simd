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
        SIMD_INLINE void Yuv420pToUyvy422(const uint8_t* y0, size_t yStride, const uint8_t* u, const uint8_t* v, uint8_t* uyvy0, size_t uyvyStride, const svbool_t& body)
        {
            const size_t A = svcntb();
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svuint8_t uv0 = svzip1_u8(_u, _v);
            svuint8_t uv1 = svzip2_u8(_u, _v);

            svuint8_t y00 = svld1_u8(body, y0 + 0 * A);
            svuint8_t y01 = svld1_u8(body, y0 + 1 * A);
            svst1_u8(body, uyvy0 + 0 * A, svzip1_u8(uv0, y00));
            svst1_u8(body, uyvy0 + 1 * A, svzip2_u8(uv0, y00));
            svst1_u8(body, uyvy0 + 2 * A, svzip1_u8(uv1, y01));
            svst1_u8(body, uyvy0 + 3 * A, svzip2_u8(uv1, y01));

            const uint8_t* y1 = y0 + yStride;
            uint8_t* uyvy1 = uyvy0 + uyvyStride;
            svuint8_t y10 = svld1_u8(body, y1 + 0 * A);
            svuint8_t y11 = svld1_u8(body, y1 + 1 * A);
            svst1_u8(body, uyvy1 + 0 * A, svzip1_u8(uv0, y10));
            svst1_u8(body, uyvy1 + 1 * A, svzip2_u8(uv0, y10));
            svst1_u8(body, uyvy1 + 2 * A, svzip1_u8(uv1, y11));
            svst1_u8(body, uyvy1 + 3 * A, svzip2_u8(uv1, y11));
        }

        SIMD_INLINE void Yuv420pToUyvy422(const uint8_t* y0, size_t yStride, const uint8_t* u, const uint8_t* v, uint8_t* uyvy0, size_t uyvyStride)
        {
            const uint8_t* y1 = y0 + yStride;
            uint8_t* uyvy1 = uyvy0 + uyvyStride;
            uyvy0[1] = y0[0];
            uyvy0[3] = y0[1];
            uyvy1[1] = y1[0];
            uyvy1[3] = y1[1];
            uyvy0[0] = u[0];
            uyvy1[0] = u[0];
            uyvy0[2] = v[0];
            uyvy1[2] = v[0];
        }

        void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 && height >= 2);

            const size_t A = svcntb(), DA = 2 * A, QA = 4 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colY = 0, colUV = 0, colUyvy = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colUyvy += QA)
                    Yuv420pToUyvy422(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride, body);
                for (; colY < width; colY += 2, colUV += 1, colUyvy += 4)
                    Yuv420pToUyvy422(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                uyvy += 2 * uyvyStride;
            }
        }
    }
#endif
}
