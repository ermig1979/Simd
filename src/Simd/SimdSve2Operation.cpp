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
        SIMD_INLINE svuint8_t DivideI16By255(const svuint16_t & value)
        {
            const svbool_t full = svptrue_b16();
            svuint16_t sum = svadd_n_u16_x(full, value, 1);
            sum = svadd_u16_x(full, sum, svlsr_n_u16_x(full, sum, 8));
            return svqxtnb_u16(svlsr_n_u16_x(full, sum, 8));
        }

        SIMD_INLINE svuint8_t VectorProduct(const svuint16_t & vertical, const svuint8_t & horizontal)
        {
            const svbool_t full = svptrue_b16();
            svuint8_t lo = DivideI16By255(svmul_u16_x(full, vertical, svunpklo_u16(horizontal)));
            svuint8_t hi = DivideI16By255(svmul_u16_x(full, vertical, svunpkhi_u16(horizontal)));
            return svuzp1_u8(lo, hi);
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            size_t A = svcntb();
            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                const svuint16_t _vertical = svdup_n_u16(vertical[row]);
                size_t col = 0;
                for (; col < widthA; col += A)
                {
                    svuint8_t _horizontal = svld1_u8(svptrue_b8(), horizontal + col);
                    svst1_u8(svptrue_b8(), dst + col, VectorProduct(_vertical, _horizontal));
                }
                if (widthA < width)
                {
                    svbool_t tail = svwhilelt_b8(col, width);
                    svuint8_t _horizontal = svld1_u8(tail, horizontal + col);
                    svst1_u8(tail, dst + col, VectorProduct(_vertical, _horizontal));
                }
                dst += stride;
            }
        }
    }
#endif
}
