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
        SIMD_INLINE uint8_t ReduceGray4x4(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
            ptrdiff_t x0, ptrdiff_t x1, ptrdiff_t x2, ptrdiff_t x3)
        {
            int c0 = s0[x0] + 3 * (s0[x1] + s0[x2]) + s0[x3];
            int c1 = s1[x0] + 3 * (s1[x1] + s1[x2]) + s1[x3];
            int c2 = s2[x0] + 3 * (s2[x1] + s2[x2]) + s2[x3];
            int c3 = s3[x0] + 3 * (s3[x1] + s3[x2]) + s3[x3];
            return (uint8_t)((c0 + 3 * (c1 + c2) + c3 + 32) >> 6);
        }

        SIMD_INLINE svuint16_t BinomialSum8(const svuint8_t& ab, const svuint8_t& cd)
        {
            svuint16_t abSum = svmlalb_n_u16(svmovlb_u16(ab), svext_u8(ab, ab, 1), 3);
            svuint16_t cdSum = svmlalb_n_u16(svmovlb_u16(svext_u8(cd, cd, 1)), cd, 3);
            return svadd_u16_x(svptrue_b16(), abSum, cdSum);
        }

        SIMD_INLINE svuint16_t ReduceColNose(const uint8_t* src)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t t1 = svld1_u8(all, src);
            return BinomialSum8(svinsr_n_u8(t1, src[0]), svld1_u8(all, src + 1));
        }

        SIMD_INLINE svuint16_t ReduceColBody(const uint8_t* src)
        {
            const svbool_t all = svptrue_b8();
            return BinomialSum8(svld1_u8(all, src - 1), svld1_u8(all, src + 1));
        }

        SIMD_INLINE svuint16_t ReduceColTail(const uint8_t* src)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t t1 = svld1_u8(all, src);
            svuint8_t last = svdup_n_u8(svlastb_u8(all, t1));
            return BinomialSum8(svld1_u8(all, src - 1), svext_u8(t1, last, 1));
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c, const svuint16_t& d)
        {
            const svbool_t mask = svptrue_b16();
            return svmla_n_u16_x(mask, svadd_u16_x(mask, a, d), svadd_u16_x(mask, b, c), 3);
        }

        SIMD_INLINE svuint16_t ReduceRow(const svuint16_t& r0, const svuint16_t& r1, const svuint16_t& r2, const svuint16_t& r3)
        {
            return svrshr_n_u16_x(svptrue_b16(), BinomialSum16(r0, r1, r2, r3), 6);
        }

        SIMD_INLINE svuint8_t ReduceRow8(
            const svuint16_t& r00, const svuint16_t& r01,
            const svuint16_t& r10, const svuint16_t& r11,
            const svuint16_t& r20, const svuint16_t& r21,
            const svuint16_t& r30, const svuint16_t& r31)
        {
            return svuzp1_u8(
                svreinterpret_u8_u16(ReduceRow(r00, r10, r20, r30)),
                svreinterpret_u8_u16(ReduceRow(r01, r11, r21, r31)));
        }

        SIMD_INLINE void ReduceGray4x4Nose(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8(
                ReduceColNose(s0), ReduceColBody(s0 + A),
                ReduceColNose(s1), ReduceColBody(s1 + A),
                ReduceColNose(s2), ReduceColBody(s2 + A),
                ReduceColNose(s3), ReduceColBody(s3 + A)));
        }

        SIMD_INLINE void ReduceGray4x4NoseTail(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8(
                ReduceColNose(s0), ReduceColTail(s0 + A),
                ReduceColNose(s1), ReduceColTail(s1 + A),
                ReduceColNose(s2), ReduceColTail(s2 + A),
                ReduceColNose(s3), ReduceColTail(s3 + A)));
        }

        SIMD_INLINE void ReduceGray4x4(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8(
                ReduceColBody(s0), ReduceColBody(s0 + A),
                ReduceColBody(s1), ReduceColBody(s1 + A),
                ReduceColBody(s2), ReduceColBody(s2 + A),
                ReduceColBody(s3), ReduceColBody(s3 + A)));
        }

        SIMD_INLINE void ReduceGray4x4Tail(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8(
                ReduceColBody(s0), ReduceColTail(s0 + A),
                ReduceColBody(s1), ReduceColTail(s1 + A),
                ReduceColBody(s2), ReduceColTail(s2 + A),
                ReduceColBody(s3), ReduceColTail(s3 + A)));
        }

        SIMD_INLINE void ReduceGray4x4Nose(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst)
        {
            svst1b_u16(svptrue_b16(), dst, ReduceRow(
                ReduceColNose(s0), ReduceColNose(s1), ReduceColNose(s2), ReduceColNose(s3)));
        }

        SIMD_INLINE void ReduceGray4x4(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst)
        {
            svst1b_u16(svptrue_b16(), dst, ReduceRow(
                ReduceColBody(s0), ReduceColBody(s1), ReduceColBody(s2), ReduceColBody(s3)));
        }

        SIMD_INLINE void ReduceGray4x4Tail(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, uint8_t* dst)
        {
            svst1b_u16(svptrue_b16(), dst, ReduceRow(
                ReduceColTail(s0), ReduceColTail(s1), ReduceColTail(s2), ReduceColTail(s3)));
        }

        void ReduceGray4x4(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const size_t A = svcntb(), DA = A * 2, QA = A * 4;
            const size_t evenWidth = AlignLo(srcWidth, 2);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* s1 = src;
                const uint8_t* s0 = s1 - (row ? srcStride : 0);
                const uint8_t* s2 = s1 + (row < srcHeight - 1 ? srcStride : 0);
                const uint8_t* s3 = s2 + (row < srcHeight - 2 ? srcStride : 0);

                if (evenWidth > DA)
                {
                    ReduceGray4x4Nose(s0, s1, s2, s3, dst, A);
                    size_t srcCol = DA, dstCol = A;
                    const size_t bodyLimit = evenWidth - A;
                    for (; srcCol + QA <= bodyLimit; srcCol += QA, dstCol += DA)
                    {
                        ReduceGray4x4(s0 + srcCol, s1 + srcCol, s2 + srcCol, s3 + srcCol, dst + dstCol, A);
                        ReduceGray4x4(s0 + srcCol + DA, s1 + srcCol + DA, s2 + srcCol + DA, s3 + srcCol + DA, dst + dstCol + A, A);
                    }
                    for (; srcCol + DA <= bodyLimit; srcCol += DA, dstCol += A)
                        ReduceGray4x4(s0 + srcCol, s1 + srcCol, s2 + srcCol, s3 + srcCol, dst + dstCol, A);
                    srcCol = evenWidth - DA;
                    dstCol = srcCol / 2;
                    ReduceGray4x4Tail(s0 + srcCol, s1 + srcCol, s2 + srcCol, s3 + srcCol, dst + dstCol, A);
                }
                else if (evenWidth == DA)
                {
                    ReduceGray4x4NoseTail(s0, s1, s2, s3, dst, A);
                }
                else
                {
                    ReduceGray4x4Nose(s0, s1, s2, s3, dst);
                    if (evenWidth != A)
                    {
                        size_t srcCol = evenWidth - A;
                        size_t dstCol = srcCol / 2;
                        ReduceGray4x4Tail(s0 + srcCol, s1 + srcCol, s2 + srcCol, s3 + srcCol, dst + dstCol);
                    }
                }
                if (evenWidth != srcWidth)
                    dst[dstWidth - 1] = ReduceGray4x4(s0 + srcWidth, s1 + srcWidth, s2 + srcWidth, s3 + srcWidth, -2, -1, -1, -1);
            }
        }
    }
#endif
}
