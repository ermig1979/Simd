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
#include "Simd/SimdPack.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void Int16ToGray(const int16_t* src, uint8_t* dst, const svbool_t& srcLo, const svbool_t& srcHi, const svbool_t& dstMask)
        {
            svint16_t lo = svmin_n_s16_x(srcLo, svmax_n_s16_x(srcLo, svld1_s16(srcLo, src), 0), 255);
            svint16_t hi = svmin_n_s16_x(srcHi, svmax_n_s16_x(srcHi, svld1_s16(srcHi, src + svcnth()), 0), 255);
            svst1_u8(dstMask, dst, PackSeqI16ToU8(lo, hi));
        }

        void Int16ToGray(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const int16_t* s = (const int16_t*)src;
            srcStride /= sizeof(int16_t);
            size_t A = svcntb();
            size_t widthA = AlignLo(width, A);
            const svbool_t body16 = svptrue_b16();
            const svbool_t body8 = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    Int16ToGray(s + col, dst + col, body16, body16, body8);
                if (col < width)
                    Int16ToGray(s + col, dst + col, svwhilelt_b16(col, width), svwhilelt_b16(col + svcnth(), width), svwhilelt_b8(col, width));
                s += srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
