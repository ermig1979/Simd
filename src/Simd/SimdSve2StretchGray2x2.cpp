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
        SIMD_INLINE void StretchGray2x2(const uint8_t* src, size_t srcCol, size_t srcWidth, uint8_t* dst0, uint8_t* dst1, size_t dstWidth)
        {
            const size_t A = svcntb();
            svbool_t srcMask = svwhilelt_b8(srcCol, srcWidth);
            svuint8_t value = svld1_u8(srcMask, src + srcCol);
            svuint8_t lo = svzip1_u8(value, value);
            svuint8_t hi = svzip2_u8(value, value);
            size_t dstCol = 2 * srcCol;
            svbool_t loMask = svwhilelt_b8(dstCol, dstWidth);
            svbool_t hiMask = svwhilelt_b8(dstCol + A, dstWidth);
            svst1_u8(loMask, dst0 + dstCol, lo);
            svst1_u8(hiMask, dst0 + dstCol + A, hi);
            svst1_u8(loMask, dst1 + dstCol, lo);
            svst1_u8(hiMask, dst1 + dstCol + A, hi);
        }

        void StretchGray2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth * 2 == dstWidth && srcHeight * 2 == dstHeight);

            const size_t A = svcntb();
            for (size_t row = 0; row < srcHeight; ++row)
            {
                uint8_t* dst0 = dst;
                uint8_t* dst1 = dst + dstStride;
                for (size_t srcCol = 0; srcCol < srcWidth; srcCol += A)
                    StretchGray2x2(src, srcCol, srcWidth, dst0, dst1, dstWidth);
                src += srcStride;
                dst += 2 * dstStride;
            }
        }
    }
#endif
}
