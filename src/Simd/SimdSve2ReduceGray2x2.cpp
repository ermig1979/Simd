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
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint16_t Average16(const uint8_t* src0, const uint8_t* src1, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t s0 = svld1_u8(mask8, src0);
            svuint8_t s1 = svld1_u8(mask8, src1);
            svuint16_t sum0 = svaddlb_u16(s0, svext_u8(s0, s0, 1));
            svuint16_t sum1 = svaddlb_u16(s1, svext_u8(s1, s1, 1));
            return svlsr_n_u16_x(mask16, svadd_n_u16_x(mask16, svadd_u16_x(mask16, sum0, sum1), 2), 2);
        }

        SIMD_INLINE svuint8_t Average8(const uint8_t* src0, const uint8_t* src1, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t even = svqxtnb_u16(Average16(src0, src1, mask8, mask16));
            return svuzp1_u8(even, even);
        }

        void ReduceGray2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            const size_t A = svcntb(), HA = svcnth();
            const size_t evenWidth = AlignLo(srcWidth, 2), dstEvenWidth = evenWidth / 2;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                for (size_t dstCol = 0, srcCol = 0; dstCol < dstEvenWidth; dstCol += HA, srcCol += A)
                {
                    svbool_t mask16 = svwhilelt_b16(dstCol, dstEvenWidth);
                    svst1_u8(svwhilelt_b8(dstCol, dstEvenWidth), dst + dstCol,
                        Average8(src0 + srcCol, src1 + srcCol, svwhilelt_b8(srcCol, evenWidth), mask16));
                }
                if (evenWidth != srcWidth)
                    dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                src += 2 * srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
