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
        template <int U, int V> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            size_t A = svlen(svuint8_t()), A2 = A * 2;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A2)
                {
                    svuint8x2_t _uv = svld2_u8(body, uv + offset);
                    if (U) svst1_u8(body, u + col, svget2(_uv, 0));
                    if (V) svst1_u8(body, v + col, svget2(_uv, 1));
                }
                if (widthA < width)
                {
                    svuint8x2_t _uv = svld2_u8(tail, uv + offset);
                    if (U) svst1_u8(tail, u + col, svget2(_uv, 0));
                    if (V) svst1_u8(tail, v + col, svget2(_uv, 1));
                }
                uv += uvStride;
                if (U) u += uStride;
                if (V) v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t* uv, size_t uvStride, size_t width, size_t height, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            if (u && v)
                DeinterleaveUv<1, 1>(uv, uvStride, width, height, u, uStride, v, vStride);
            else if (u)
                DeinterleaveUv<1, 0>(uv, uvStride, width, height, u, uStride, v, vStride);
            else if (v)
                DeinterleaveUv<0, 1>(uv, uvStride, width, height, u, uStride, v, vStride);
        }
    }
#endif
}
