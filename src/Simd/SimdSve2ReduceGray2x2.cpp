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
        SIMD_INLINE svuint16_t Average16(const svuint8_t& s0, const svuint8_t& s1)
        {
            const svbool_t mask = svptrue_b16();
            return svlsr_n_u16_x(mask, svadalp_u16_x(mask, svadalp_u16_x(mask, svdup_n_u16(2), s0), s1), 2);
        }

        SIMD_INLINE svuint8_t Average8(const svuint8_t& s00, const svuint8_t& s01, const svuint8_t& s10, const svuint8_t& s11)
        {
            return svuzp1_u8(svreinterpret_u8_u16(Average16(s00, s10)), svreinterpret_u8_u16(Average16(s01, s11)));
        }

        SIMD_INLINE void ReduceGray2x2(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t A)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t s00 = svld1_u8(all, src0 + 0);
            svuint8_t s01 = svld1_u8(all, src0 + A);
            svuint8_t s10 = svld1_u8(all, src1 + 0);
            svuint8_t s11 = svld1_u8(all, src1 + A);
            svst1_u8(all, dst, Average8(s00, s01, s10, s11));
        }

        SIMD_INLINE void ReduceGray2x2(const uint8_t* src0, const uint8_t* src1, uint8_t* dst)
        {
            const svbool_t all = svptrue_b8();
            svst1b_u16(svptrue_b16(), dst, Average16(svld1_u8(all, src0), svld1_u8(all, src1)));
        }

        void ReduceGray2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= svcntb());

            const size_t A = svcntb(), DA = A * 2, QA = A * 4;
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t alignedQa = AlignLo(evenWidth, QA);
            const size_t alignedDa = AlignLo(evenWidth, DA);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedQa; srcOffset += QA, dstOffset += DA)
                {
                    ReduceGray2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A);
                    ReduceGray2x2(src0 + srcOffset + DA, src1 + srcOffset + DA, dst + dstOffset + A, A);
                }
                for (; srcOffset < alignedDa; srcOffset += DA, dstOffset += A)
                    ReduceGray2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A);
                if (alignedDa != evenWidth)
                {
                    if (evenWidth >= DA)
                    {
                        srcOffset = evenWidth - DA;
                        dstOffset = srcOffset / 2;
                        ReduceGray2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A);
                    }
                    else
                    {
                        ReduceGray2x2(src0, src1, dst);
                        if (evenWidth != A)
                        {
                            srcOffset = evenWidth - A;
                            dstOffset = srcOffset / 2;
                            ReduceGray2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                        }
                    }
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
