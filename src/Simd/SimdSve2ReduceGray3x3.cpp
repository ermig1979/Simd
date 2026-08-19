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
        template <bool compensation> SIMD_INLINE svuint16_t DivideBy16(const svuint16_t& value);

        template <> SIMD_INLINE svuint16_t DivideBy16<true>(const svuint16_t& value)
        {
            return svrshr_n_u16_x(svptrue_b16(), value, 4);
        }

        template <> SIMD_INLINE svuint16_t DivideBy16<false>(const svuint16_t& value)
        {
            return svlsr_n_u16_x(svptrue_b16(), value, 4);
        }

        SIMD_INLINE svuint16_t ReduceCol(const svuint8_t& t01, const svuint8_t& t12)
        {
            return svadalp_u16_x(svptrue_b16(), svaddlp_u16(t01), t12);
        }

        SIMD_INLINE svuint16_t ReduceColNose(const uint8_t* src)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t t12 = svld1_u8(all, src);
            return ReduceCol(svinsr_n_u8(t12, src[0]), t12);
        }

        SIMD_INLINE svuint16_t ReduceColBody(const uint8_t* src)
        {
            const svbool_t all = svptrue_b8();
            return ReduceCol(svld1_u8(all, src - 1), svld1_u8(all, src));
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c)
        {
            const svbool_t mask = svptrue_b16();
            return svmla_n_u16_x(mask, svadd_u16_x(mask, a, c), b, 2);
        }

        template <bool compensation> SIMD_INLINE svuint16_t ReduceRow(const svuint16_t& r0, const svuint16_t& r1, const svuint16_t& r2)
        {
            return DivideBy16<compensation>(BinomialSum16(r0, r1, r2));
        }

        template <bool compensation> SIMD_INLINE svuint8_t ReduceRow8(
            const svuint16_t& r00, const svuint16_t& r01,
            const svuint16_t& r10, const svuint16_t& r11,
            const svuint16_t& r20, const svuint16_t& r21)
        {
            return svuzp1_u8(
                svreinterpret_u8_u16(ReduceRow<compensation>(r00, r10, r20)),
                svreinterpret_u8_u16(ReduceRow<compensation>(r01, r11, r21)));
        }

        template <bool compensation> SIMD_INLINE void ReduceGray3x3Nose(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8<compensation>(
                ReduceColNose(s0), ReduceColBody(s0 + A),
                ReduceColNose(s1), ReduceColBody(s1 + A),
                ReduceColNose(s2), ReduceColBody(s2 + A)));
        }

        template <bool compensation> SIMD_INLINE void ReduceGray3x3(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, uint8_t* dst, size_t A)
        {
            svst1_u8(svptrue_b8(), dst, ReduceRow8<compensation>(
                ReduceColBody(s0), ReduceColBody(s0 + A),
                ReduceColBody(s1), ReduceColBody(s1 + A),
                ReduceColBody(s2), ReduceColBody(s2 + A)));
        }

        template <bool compensation> SIMD_INLINE void ReduceGray3x3Nose(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, uint8_t* dst)
        {
            svst1b_u16(svptrue_b16(), dst, ReduceRow<compensation>(
                ReduceColNose(s0), ReduceColNose(s1), ReduceColNose(s2)));
        }

        template <bool compensation> SIMD_INLINE void ReduceGray3x3(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, uint8_t* dst)
        {
            svst1b_u16(svptrue_b16(), dst, ReduceRow<compensation>(
                ReduceColBody(s0), ReduceColBody(s1), ReduceColBody(s2)));
        }

        template <bool compensation> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth >= svcntb() && (srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            const size_t A = svcntb(), DA = A * 2, QA = A * 4;
            const size_t evenWidth = AlignLo(srcWidth, 2);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* s1 = src;
                const uint8_t* s0 = s1 - (row ? srcStride : 0);
                const uint8_t* s2 = s1 + (row != srcHeight - 1 ? srcStride : 0);

                if (evenWidth >= DA)
                {
                    ReduceGray3x3Nose<compensation>(s0, s1, s2, dst, A);
                    size_t srcCol = DA, dstCol = A;
                    for (; srcCol + QA <= evenWidth; srcCol += QA, dstCol += DA)
                    {
                        ReduceGray3x3<compensation>(s0 + srcCol, s1 + srcCol, s2 + srcCol, dst + dstCol, A);
                        ReduceGray3x3<compensation>(s0 + srcCol + DA, s1 + srcCol + DA, s2 + srcCol + DA, dst + dstCol + A, A);
                    }
                    for (; srcCol + DA <= evenWidth; srcCol += DA, dstCol += A)
                        ReduceGray3x3<compensation>(s0 + srcCol, s1 + srcCol, s2 + srcCol, dst + dstCol, A);
                    if (srcCol != evenWidth)
                    {
                        srcCol = evenWidth - DA;
                        dstCol = srcCol / 2;
                        ReduceGray3x3<compensation>(s0 + srcCol, s1 + srcCol, s2 + srcCol, dst + dstCol, A);
                    }
                }
                else
                {
                    ReduceGray3x3Nose<compensation>(s0, s1, s2, dst);
                    if (evenWidth != A)
                    {
                        size_t srcCol = evenWidth - A;
                        size_t dstCol = srcCol / 2;
                        ReduceGray3x3<compensation>(s0 + srcCol, s1 + srcCol, s2 + srcCol, dst + dstCol);
                    }
                }
                if (evenWidth != srcWidth)
                    dst[dstWidth - 1] = Base::GaussianBlur3x3<compensation>(s0 + srcWidth, s1 + srcWidth, s2 + srcWidth, -2, -1, -1);
            }
        }

        void ReduceGray3x3(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray3x3<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray3x3<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif
}
