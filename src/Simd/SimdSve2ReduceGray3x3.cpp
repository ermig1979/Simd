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
        SIMD_INLINE svuint8_t PrependFirst(const svuint8_t& value)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t iota = svindex_u8(0, 1);
            svuint8_t idx = svqsub_u8_x(mask, iota, svdup_n_u8(1));
            idx = svsel_u8(svcmpeq_n_u8(mask, idx, 255), svdup_n_u8(0), idx);
            return svtbl_u8(value, idx);
        }

        template <bool compensation> SIMD_INLINE svuint16_t DivideBy16(const svuint16_t& value, const svbool_t& mask);

        template <> SIMD_INLINE svuint16_t DivideBy16<true>(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 8), 4);
        }

        template <> SIMD_INLINE svuint16_t DivideBy16<false>(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, value, 4);
        }

        SIMD_INLINE svuint16_t PairSum(const svuint8_t& value)
        {
            return svaddlb_u16(value, svext_u8(value, value, 1));
        }

        SIMD_INLINE svuint16_t ReduceColNose(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t t12 = svld1_u8(mask8, src);
            svuint8_t t01 = PrependFirst(t12);
            return svadd_u16_x(mask16, PairSum(t01), PairSum(t12));
        }

        SIMD_INLINE svuint16_t ReduceColBody(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t t01 = svld1_u8(mask8, src - 1);
            svuint8_t t12 = svld1_u8(mask8, src);
            return svadd_u16_x(mask16, PairSum(t01), PairSum(t12));
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c, const svbool_t& mask)
        {
            return svadd_u16_x(mask, svadd_u16_x(mask, a, c), svlsl_n_u16_x(mask, b, 1));
        }

        template <bool compensation> SIMD_INLINE svuint8_t ReduceRow(const svuint16_t& r0, const svuint16_t& r1, const svuint16_t& r2, const svbool_t& mask16)
        {
            svuint8_t value = svqxtnb_u16(DivideBy16<compensation>(BinomialSum16(r0, r1, r2, mask16), mask16));
            return svuzp1_u8(value, value);
        }

        template <bool compensation> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth >= svcntb() && (srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            const size_t A = svcntb(), HA = svcnth();
            size_t lastOddCol = srcWidth - AlignLo(srcWidth, 2);
            size_t bodyWidth = AlignLo(srcWidth, A);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* s1 = src;
                const uint8_t* s0 = s1 - (row ? srcStride : 0);
                const uint8_t* s2 = s1 + (row != srcHeight - 1 ? srcStride : 0);

                svbool_t noseMask8 = svwhilelt_b8(size_t(0), Simd::Min(srcWidth, A));
                svbool_t noseMask16 = svwhilelt_b16(size_t(0), Simd::Min(dstWidth, HA));
                svbool_t noseStore = svwhilelt_b8(size_t(0), Simd::Min(dstWidth, HA));
                svst1_u8(noseStore, dst, ReduceRow<compensation>(
                    ReduceColNose(s0, noseMask8, noseMask16),
                    ReduceColNose(s1, noseMask8, noseMask16),
                    ReduceColNose(s2, noseMask8, noseMask16), noseMask16));

                for (size_t srcCol = A, dstCol = HA; srcCol < bodyWidth; srcCol += A, dstCol += HA)
                {
                    svbool_t bodyMask8 = svwhilelt_b8(srcCol, srcWidth);
                    svbool_t bodyMask16 = svwhilelt_b16(dstCol, dstWidth);
                    svbool_t bodyStore = svwhilelt_b8(dstCol, dstWidth);
                    svst1_u8(bodyStore, dst + dstCol, ReduceRow<compensation>(
                        ReduceColBody(s0 + srcCol, bodyMask8, bodyMask16),
                        ReduceColBody(s1 + srcCol, bodyMask8, bodyMask16),
                        ReduceColBody(s2 + srcCol, bodyMask8, bodyMask16), bodyMask16));
                }

                if (bodyWidth != srcWidth)
                {
                    size_t srcCol = srcWidth - A - lastOddCol;
                    size_t dstCol = dstWidth - HA - lastOddCol;
                    svbool_t tailMask8 = svwhilelt_b8(srcCol, srcWidth);
                    svbool_t tailMask16 = svwhilelt_b16(dstCol, dstWidth);
                    svbool_t tailStore = svwhilelt_b8(dstCol, dstWidth);
                    svst1_u8(tailStore, dst + dstCol, ReduceRow<compensation>(
                        ReduceColBody(s0 + srcCol, tailMask8, tailMask16),
                        ReduceColBody(s1 + srcCol, tailMask8, tailMask16),
                        ReduceColBody(s2 + srcCol, tailMask8, tailMask16), tailMask16));
                    if (lastOddCol)
                        dst[dstWidth - 1] = Base::GaussianBlur3x3<compensation>(s0 + srcWidth, s1 + srcWidth, s2 + srcWidth, -2, -1, -1);
                }
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
