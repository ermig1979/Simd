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
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 4 * width);
                    src0 = (uint16_t*)_p;
                    src1 = src0 + width;
                    src2 = src1 + width;
                    src3 = src2 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* src0;
                uint16_t* src1;
                uint16_t* src2;
                uint16_t* src3;
            private:
                void* _p;
            };
        }

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

        SIMD_INLINE svuint16_t BinomialSum16Pairs(const svuint8_t& ab, const svuint8_t& cd, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t abe, abo, cde, cdo;
            EvenOdd(ab, mask8, abe, abo);
            EvenOdd(cd, mask8, cde, cdo);
            svuint16_t s0 = svadd_u16_x(mask16, svunpklo_u16(abe), svmul_n_u16_x(mask16, svunpklo_u16(abo), 3));
            svuint16_t s1 = svadd_u16_x(mask16, svmul_n_u16_x(mask16, svunpklo_u16(cde), 3), svunpklo_u16(cdo));
            return svadd_u16_x(mask16, s0, s1);
        }

        SIMD_INLINE svuint16_t ReduceColNose(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum16Pairs(PrependFirst(svld1_u8(mask8, src)), svld1_u8(mask8, src + 1), mask8, mask16);
        }

        SIMD_INLINE svuint16_t ReduceColBody(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum16Pairs(svld1_u8(mask8, src - 1), svld1_u8(mask8, src + 1), mask8, mask16);
        }

        template <bool even> SIMD_INLINE svuint16_t ReduceColTail(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16);

        template <> SIMD_INLINE svuint16_t ReduceColTail<true>(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            return BinomialSum16Pairs(svld1_u8(mask8, src - 1), AppendLast1(svld1_u8(mask8, src), mask8), mask8, mask16);
        }

        template <> SIMD_INLINE svuint16_t ReduceColTail<false>(const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t ab = svld1_u8(mask8, src - 1);
            return BinomialSum16Pairs(ab, AppendLast2(ab, mask8), mask8, mask16);
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

        SIMD_INLINE svuint16_t ReduceRow16(const Buffer& buffer, size_t offset, const svbool_t& mask16)
        {
            return DivideBy64(BinomialSum16(
                svld1_u16(mask16, buffer.src0 + offset),
                svld1_u16(mask16, buffer.src1 + offset),
                svld1_u16(mask16, buffer.src2 + offset),
                svld1_u16(mask16, buffer.src3 + offset), mask16), mask16);
        }

        SIMD_INLINE svuint8_t ReduceRow(const Buffer& buffer, size_t offset, const svbool_t& mask16)
        {
            svuint8_t value = svqxtnb_u16(ReduceRow16(buffer, offset, mask16));
            return svuzp1_u8(value, value);
        }

        SIMD_INLINE void StoreColNose(uint16_t* dst, const uint8_t* src, const svbool_t& mask8, const svbool_t& mask16)
        {
            svst1_u16(mask16, dst, ReduceColNose(src, mask8, mask16));
        }

        SIMD_INLINE void StoreColBody(uint16_t* dst, const uint8_t* src, size_t srcCol, const svbool_t& mask8, const svbool_t& mask16)
        {
            svst1_u16(mask16, dst, ReduceColBody(src + srcCol, mask8, mask16));
        }

        template <bool even> SIMD_INLINE void StoreColTail(uint16_t* dst, const uint8_t* src, size_t srcCol, const svbool_t& mask8, const svbool_t& mask16)
        {
            svst1_u16(mask16, dst, ReduceColTail<even>(src + srcCol, mask8, mask16));
        }

        template <bool even> void ReduceGray4x4(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const size_t A = svcntb();
            const size_t HA = svcnth();
            size_t alignedDstWidth = AlignLo(dstWidth, HA);
            size_t srcTail = AlignHi(srcWidth - A, 2);

            Buffer buffer(AlignHi(dstWidth, A));

            {
                svbool_t mask8 = svwhilelt_b8(size_t(0), Simd::Min(srcWidth, A));
                svbool_t mask16 = svwhilelt_b16(size_t(0), Simd::Min(dstWidth, HA));
                StoreColNose(buffer.src0, src, mask8, mask16);
                StoreColNose(buffer.src1, src, mask8, mask16);
                for (size_t srcCol = A, dstCol = HA; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                {
                    mask8 = svwhilelt_b8(srcCol, srcWidth);
                    mask16 = svwhilelt_b16(dstCol, dstWidth);
                    StoreColBody(buffer.src0 + dstCol, src, srcCol, mask8, mask16);
                    StoreColBody(buffer.src1 + dstCol, src, srcCol, mask8, mask16);
                }
                mask8 = svwhilelt_b8(srcTail, srcWidth);
                mask16 = svwhilelt_b16(dstWidth - HA, dstWidth);
                StoreColTail<even>(buffer.src0 + dstWidth - HA, src, srcTail, mask8, mask16);
                StoreColTail<even>(buffer.src1 + dstWidth - HA, src, srcTail, mask8, mask16);
            }

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t* src2 = src + srcStride * (row + 1);
                const uint8_t* src3 = src2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    src2 = src + srcStride * (srcHeight - 1);
                    src3 = src2;
                }

                {
                    svbool_t mask8 = svwhilelt_b8(size_t(0), Simd::Min(srcWidth, A));
                    svbool_t mask16 = svwhilelt_b16(size_t(0), Simd::Min(dstWidth, HA));
                    StoreColNose(buffer.src2, src2, mask8, mask16);
                    StoreColNose(buffer.src3, src3, mask8, mask16);
                    for (size_t srcCol = A, dstCol = HA; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                    {
                        mask8 = svwhilelt_b8(srcCol, srcWidth);
                        mask16 = svwhilelt_b16(dstCol, dstWidth);
                        StoreColBody(buffer.src2 + dstCol, src2, srcCol, mask8, mask16);
                        StoreColBody(buffer.src3 + dstCol, src3, srcCol, mask8, mask16);
                    }
                    mask8 = svwhilelt_b8(srcTail, srcWidth);
                    mask16 = svwhilelt_b16(dstWidth - HA, dstWidth);
                    StoreColTail<even>(buffer.src2 + dstWidth - HA, src2, srcTail, mask8, mask16);
                    StoreColTail<even>(buffer.src3 + dstWidth - HA, src3, srcTail, mask8, mask16);
                }

                for (size_t col = 0; col < alignedDstWidth; col += HA)
                {
                    svbool_t mask16 = svwhilelt_b16(col, dstWidth);
                    svbool_t mask8 = svwhilelt_b8(col, Simd::Min(col + HA, dstWidth));
                    svst1_u8(mask8, dst + col, ReduceRow(buffer, col, mask16));
                }

                if (alignedDstWidth != dstWidth)
                {
                    size_t col = dstWidth - HA;
                    svbool_t mask16 = svwhilelt_b16(col, dstWidth);
                    svbool_t mask8 = svwhilelt_b8(col, Simd::Min(col + HA, dstWidth));
                    svst1_u8(mask8, dst + col, ReduceRow(buffer, col, mask16));
                }

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src1, buffer.src3);
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
