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
        SIMD_INLINE svuint8_t PrependFirst(const svuint8_t& value)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t iota = svindex_u8(0, 1);
            svuint8_t idx = svqsub_u8_x(mask, iota, svdup_n_u8(1));
            idx = svsel_u8(svcmpeq_n_u8(mask, idx, 255), svdup_n_u8(0), idx);
            return svtbl_u8(value, idx);
        }

        SIMD_INLINE svuint8_t AppendLast1(const svuint8_t& value, const svbool_t& mask8)
        {
            svuint8_t last = svdup_u8(svlastb_u8(mask8, value));
            return svext_u8(value, last, 1);
        }

        SIMD_INLINE svuint8_t AppendLast2(const svuint8_t& value, const svbool_t& mask8)
        {
            svuint8_t last = svdup_u8(svlastb_u8(mask8, value));
            return svext_u8(svext_u8(value, last, 1), last, 1);
        }

        SIMD_INLINE void EvenOdd(const svuint8_t& value, const svbool_t& mask8, svuint8_t& even, svuint8_t& odd)
        {
            svuint8_t iota = svindex_u8(0, 1);
            svuint8_t idxEven = svlsl_n_u8_x(mask8, iota, 1);
            svuint8_t idxOdd = svadd_n_u8_x(mask8, idxEven, 1);
            even = svtbl_u8(value, idxEven);
            odd = svtbl_u8(value, idxOdd);
        }

        SIMD_INLINE svuint16_t BinomialSum8(const svuint8_t& t01, const svuint8_t& t23, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t t01e, t01o, t23e, t23o;
            EvenOdd(t01, mask8, t01e, t01o);
            EvenOdd(t23, mask8, t23e, t23o);
            svuint16_t lo = svadd_u16_x(mask16, svmovlb_u16(t01e), svmovlb_u16(t23o));
            svuint16_t mid = svadd_u16_x(mask16, svmovlb_u16(t01o), svmovlb_u16(t23e));
            return svadd_u16_x(mask16, lo, svadd_u16_x(mask16, mid, svlsl_n_u16_x(mask16, mid, 1)));
        }

        SIMD_INLINE svuint16_t ReduceColNose(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum8(PrependFirst(svld1_u8(mask8, src)), svld1_u8(mask8, src + 1), mask8, mask16);
        }

        SIMD_INLINE svuint16_t ReduceColBody(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum8(svld1_u8(mask8, src - 1), svld1_u8(mask8, src + 1), mask8, mask16);
        }

        template <bool even> SIMD_INLINE svuint16_t ReduceColTail(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16);

        template <> SIMD_INLINE svuint16_t ReduceColTail<true>(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum8(svld1_u8(mask8, src - 1), AppendLast1(svld1_u8(mask8, src), mask8), mask8, mask16);
        }

        template <> SIMD_INLINE svuint16_t ReduceColTail<false>(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t t01 = svld1_u8(mask8, src - 1);
            return BinomialSum8(t01, AppendLast2(t01, mask8), mask8, mask16);
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c, const svuint16_t& d, const svbool_t& mask)
        {
            svuint16_t bc = svadd_u16_x(mask, b, c);
            return svadd_u16_x(mask, svadd_u16_x(mask, a, d), svadd_u16_x(mask, bc, svlsl_n_u16_x(mask, bc, 1)));
        }

        SIMD_INLINE svuint16_t DivideBy64(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 32), 6);
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE svuint8_t ReduceRow(
            const svuint16_t& lo0, const svuint16_t& lo1, const svuint16_t& lo2, const svuint16_t& lo3,
            const svuint16_t& hi0, const svuint16_t& hi1, const svuint16_t& hi2, const svuint16_t& hi3,
            const svbool_t& maskLo16, const svbool_t& maskHi16)
        {
            return PackU16ToU8(
                DivideBy64(BinomialSum16(lo0, lo1, lo2, lo3, maskLo16), maskLo16),
                DivideBy64(BinomialSum16(hi0, hi1, hi2, hi3, maskHi16), maskHi16));
        }

        template <bool even> void ReduceGray4x4(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const size_t A = svcntb();
            const size_t HA = svcnth();
            size_t bodyWidth = AlignLo(srcWidth, A);
            size_t srcTail = AlignHi(srcWidth - A, 2);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t* s0 = src + srcStride * row - (row ? srcStride : 0);
                const uint8_t* s1 = src + srcStride * row;
                const uint8_t* s2 = s1 + (row < srcHeight - 1 ? srcStride : 0);
                const uint8_t* s3 = s2 + (row < srcHeight - 2 ? srcStride : 0);

                svuint16_t lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
                {
                    svbool_t mask8 = svwhilelt_b8(size_t(0), Simd::Min(srcWidth, A));
                    svbool_t mask16 = svwhilelt_b16(size_t(0), Simd::Min(dstWidth, HA));
                    svbool_t mask8Hi = svwhilelt_b8(A, Simd::Min(srcWidth, A + A));
                    svbool_t mask16Hi = svwhilelt_b16(HA, Simd::Min(dstWidth, A));
                    svbool_t maskStore = svwhilelt_b8(size_t(0), Simd::Min(dstWidth, A));
                    lo0 = ReduceColNose(s0, mask8, mask16);
                    lo1 = ReduceColNose(s1, mask8, mask16);
                    lo2 = ReduceColNose(s2, mask8, mask16);
                    lo3 = ReduceColNose(s3, mask8, mask16);
                    hi0 = ReduceColBody(s0 + A, mask8Hi, mask16Hi);
                    hi1 = ReduceColBody(s1 + A, mask8Hi, mask16Hi);
                    hi2 = ReduceColBody(s2 + A, mask8Hi, mask16Hi);
                    hi3 = ReduceColBody(s3 + A, mask8Hi, mask16Hi);
                    svst1_u8(maskStore, dst, ReduceRow(lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3, mask16, mask16Hi));
                }

                for (size_t srcCol = A, dstCol = A; srcCol < bodyWidth; srcCol += A, dstCol += A)
                {
                    svbool_t mask8Lo = svwhilelt_b8(srcCol, srcWidth);
                    svbool_t mask8Hi = svwhilelt_b8(srcCol + A, srcWidth);
                    svbool_t mask16Lo = svwhilelt_b16(dstCol, dstWidth);
                    svbool_t mask16Hi = svwhilelt_b16(dstCol + HA, dstWidth);
                    svbool_t maskStore = svwhilelt_b8(dstCol, dstWidth);
                    lo0 = ReduceColBody(s0 + srcCol, mask8Lo, mask16Lo);
                    lo1 = ReduceColBody(s1 + srcCol, mask8Lo, mask16Lo);
                    lo2 = ReduceColBody(s2 + srcCol, mask8Lo, mask16Lo);
                    lo3 = ReduceColBody(s3 + srcCol, mask8Lo, mask16Lo);
                    hi0 = ReduceColBody(s0 + srcCol + A, mask8Hi, mask16Hi);
                    hi1 = ReduceColBody(s1 + srcCol + A, mask8Hi, mask16Hi);
                    hi2 = ReduceColBody(s2 + srcCol + A, mask8Hi, mask16Hi);
                    hi3 = ReduceColBody(s3 + srcCol + A, mask8Hi, mask16Hi);
                    svst1_u8(maskStore, dst + dstCol, ReduceRow(lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3, mask16Lo, mask16Hi));
                }

                if (bodyWidth != srcWidth)
                {
                    size_t dstCol = dstWidth - A;
                    svbool_t mask8Lo = svwhilelt_b8(srcTail, srcWidth);
                    svbool_t mask8Hi = svwhilelt_b8(srcTail + A, srcWidth);
                    svbool_t mask16Lo = svwhilelt_b16(dstCol, dstWidth);
                    svbool_t mask16Hi = svwhilelt_b16(dstCol + HA, dstWidth);
                    svbool_t maskStore = svwhilelt_b8(dstCol, dstWidth);
                    lo0 = ReduceColBody(s0 + srcTail, mask8Lo, mask16Lo);
                    lo1 = ReduceColBody(s1 + srcTail, mask8Lo, mask16Lo);
                    lo2 = ReduceColBody(s2 + srcTail, mask8Lo, mask16Lo);
                    lo3 = ReduceColBody(s3 + srcTail, mask8Lo, mask16Lo);
                    hi0 = ReduceColTail<even>(s0 + srcTail + A, mask8Hi, mask16Hi);
                    hi1 = ReduceColTail<even>(s1 + srcTail + A, mask8Hi, mask16Hi);
                    hi2 = ReduceColTail<even>(s2 + srcTail + A, mask8Hi, mask16Hi);
                    hi3 = ReduceColTail<even>(s3 + srcTail + A, mask8Hi, mask16Hi);
                    svst1_u8(maskStore, dst + dstCol, ReduceRow(lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3, mask16Lo, mask16Hi));
                }
            }
        }

        void ReduceGray4x4(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(srcWidth, 2))
                ReduceGray4x4<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray4x4<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif
}
