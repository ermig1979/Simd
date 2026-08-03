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
        SIMD_INLINE void InterleaveUv(const uint8_t* u, const uint8_t* v, uint8_t* uv,
            const svbool_t& load, const svbool_t& store0, const svbool_t& store1)
        {
            svuint8_t _u = svld1_u8(load, u);
            svuint8_t _v = svld1_u8(load, v);
            svst1_u8(store0, uv, svzip1_u8(_u, _v));
            svst1_u8(store1, uv + svcntb(), svzip2_u8(_u, _v));
        }

        void InterleaveUv(const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* uv, size_t uvStride)
        {
            size_t A = svcntb(), A2 = A * 2;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 2;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A2)
                    InterleaveUv(u + col, v + col, uv + offset, body, body, body);
                if (widthA < width)
                    InterleaveUv(u + col, v + col, uv + offset, tail, tail0, tail1);
                u += uStride;
                v += vStride;
                uv += uvStride;
            }
        }
    }
#endif
}
