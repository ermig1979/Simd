/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2022-2022 Fabien Spindler,
*               2022-2022 Souriya Trinh.
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
#include "Simd/SimdSve1.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE
    namespace Sve
    {
        void AbsDifference(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, uint8_t* c, size_t cStride, size_t width, size_t height)
        {
            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                {
                    svuint8_t _a = svld1_u8(body, a + col);
                    svuint8_t _b = svld1_u8(body, b + col);
                    svuint8_t _c = svabd_x(body, _a, _b);
                    svst1_u8(body, c + col, _c);
                }
                if (widthA < width)
                {
                    svuint8_t _a = svld1_u8(tail, a + col);
                    svuint8_t _b = svld1_u8(tail, b + col);
                    svuint8_t _c = svabd_x(tail, _a, _b);
                    svst1_u8(tail, c + col, _c);
                }
                a += aStride;
                b += bStride;
                c += cStride;
            }
        }
    }
#endif
}
