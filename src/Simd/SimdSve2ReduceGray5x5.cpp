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
            const size_t BA = 16;
            const size_t BHA = BA / 2;

            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * (5 * width + BA));
                    in0 = (uint16_t*)_p;
                    in1 = in0 + width;
                    out0 = in1 + width;
                    out1 = out0 + width;
                    dst = out1 + width + BHA;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* in0;
                uint16_t* in1;
                uint16_t* out0;
                uint16_t* out1;
                uint16_t* dst;
            private:
                void* _p;
            };
        }

        template <bool compensation> SIMD_INLINE svuint16_t DivideBy256(const svuint16_t& value, const svbool_t& mask);

        template <> SIMD_INLINE svuint16_t DivideBy256<true>(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 128), 8);
        }

        template <> SIMD_INLINE svuint16_t DivideBy256<false>(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, value, 8);
        }

        SIMD_INLINE void FirstRow5x5Part(svuint16_t src, Buffer& buffer, size_t offset, const svbool_t& mask16)
        {
            svst1_u16(mask16, buffer.in0 + offset, src);
            svst1_u16(mask16, buffer.in1 + offset, svmul_n_u16_x(mask16, src, 5));
        }

        SIMD_INLINE void FirstRow5x5Part(const uint8_t* src, Buffer& buffer, size_t offset,
            const svbool_t& mask8, const svbool_t& mask16)
        {
            FirstRow5x5Part(svld1ub_u16(mask16, src + offset), buffer, offset, mask16);
        }

        SIMD_INLINE void FirstRow5x5(const uint8_t* src, Buffer& buffer, size_t offset,
            const svbool_t& mask8, const svbool_t& mask16)
        {
            FirstRow5x5Part(src, buffer, offset, mask8, mask16);
            FirstRow5x5Part(src, buffer, offset + BHA, mask8, mask16);
        }

        SIMD_INLINE void MainRowY5x5Part(svuint16_t odd, svuint16_t even, Buffer& buffer, size_t offset, const svbool_t& mask16)
        {
            svuint16_t cp = svmul_n_u16_x(mask16, odd, 4);
            svuint16_t c0 = svld1_u16(mask16, buffer.in0 + offset);
            svuint16_t c1 = svld1_u16(mask16, buffer.in1 + offset);
            svst1_u16(mask16, buffer.dst + offset, svadd_u16_x(mask16, even,
                svadd_u16_x(mask16, c1, svadd_u16_x(mask16, cp, svmul_n_u16_x(mask16, c0, 6)))));
            svst1_u16(mask16, buffer.out1 + offset, svadd_u16_x(mask16, c0, cp));
            svst1_u16(mask16, buffer.out0 + offset, even);
        }

        SIMD_INLINE void MainRowY5x5(const uint8_t* odd, const uint8_t* even, Buffer& buffer, size_t offset,
            const svbool_t& mask8, const svbool_t& mask16)
        {
            MainRowY5x5Part(svld1ub_u16(mask16, odd + offset), svld1ub_u16(mask16, even + offset), buffer, offset, mask16);
            MainRowY5x5Part(svld1ub_u16(mask16, odd + offset + BHA), svld1ub_u16(mask16, even + offset + BHA), buffer, offset + BHA, mask16);
        }

        template <bool compensation> SIMD_INLINE svuint16_t MainRowX5x5(uint16_t* row, const svbool_t& mask16)
        {
            svuint16_t t0 = svld1_u16(mask16, row - 2);
            svuint16_t t1 = svld1_u16(mask16, row - 1);
            svuint16_t t2 = svld1_u16(mask16, row);
            svuint16_t t3 = svld1_u16(mask16, row + 1);
            svuint16_t t4 = svld1_u16(mask16, row + 2);
            t2 = svadd_u16_x(mask16, svadd_u16_x(mask16, svmul_n_u16_x(mask16, t2, 6),
                svmul_n_u16_x(mask16, svadd_u16_x(mask16, t1, t3), 4)), svadd_u16_x(mask16, t0, t4));
            return DivideBy256<compensation>(t2, mask16);
        }

        template <bool compensation> SIMD_INLINE svuint8_t PackReduceGray5x5(const svuint16_t& lo, const svuint16_t& hi)
        {
            const svbool_t mask16 = svptrue_pat_b16(SV_VL8);
            const svbool_t mask8 = svptrue_pat_b8(SV_VL8);
            uint16_t buf16[BA];
            uint8_t out[BHA];
            svst1_u16(mask16, buf16, lo);
            svst1_u16(mask16, buf16 + BHA, hi);
            for (size_t i = 0; i < BHA / 2; ++i)
            {
                out[i] = (uint8_t)buf16[i * 2];
                out[BHA / 2 + i] = (uint8_t)buf16[BHA + i * 2];
            }
            return svld1_u8(mask8, out);
        }

        template <bool compensation> SIMD_INLINE void MainRowX5x5(Buffer& buffer, size_t offset, uint8_t* dst,
            const svbool_t& mask16Lo, const svbool_t& mask16Hi, const svbool_t& mask8)
        {
            svuint16_t lo = MainRowX5x5<compensation>(buffer.dst + offset, mask16Lo);
            svuint16_t hi = MainRowX5x5<compensation>(buffer.dst + offset + BHA, mask16Hi);
            svst1_u8(mask8, dst, PackReduceGray5x5<compensation>(lo, hi));
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const svbool_t body8 = svptrue_pat_b8(SV_VL16);
            const svbool_t body16 = svptrue_pat_b16(SV_VL8);
            size_t alignedWidth = AlignLo(srcWidth, BA);
            size_t bufferDstTail = AlignHi(srcWidth - BA, 2);

            Buffer buffer(AlignHi(srcWidth, BA));

            for (size_t col = 0; col < alignedWidth; col += BA)
                FirstRow5x5(src, buffer, col, body8, body16);
            if (alignedWidth != srcWidth)
            {
                size_t col = srcWidth - BA;
                FirstRow5x5Part(src, buffer, col, svwhilelt_b8(col, srcWidth), svwhilelt_b16(col, srcWidth));
                FirstRow5x5Part(src, buffer, col + BHA, svwhilelt_b8(col + BHA, srcWidth), svwhilelt_b16(col + BHA, srcWidth));
            }
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t* even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t col = 0; col < alignedWidth; col += BA)
                    MainRowY5x5(odd, even, buffer, col, body8, body16);
                if (alignedWidth != srcWidth)
                {
                    size_t col = srcWidth - BA;
                    MainRowY5x5Part(svld1ub_u16(svwhilelt_b16(col, srcWidth), odd + col),
                        svld1ub_u16(svwhilelt_b16(col, srcWidth), even + col), buffer, col, svwhilelt_b16(col, srcWidth));
                    MainRowY5x5Part(svld1ub_u16(svwhilelt_b16(col + BHA, srcWidth), odd + col + BHA),
                        svld1ub_u16(svwhilelt_b16(col + BHA, srcWidth), even + col + BHA), buffer, col + BHA, svwhilelt_b16(col + BHA, srcWidth));
                }

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += BA, dstCol += BHA)
                    MainRowX5x5<compensation>(buffer, srcCol, dst + dstCol, body16, body16, svptrue_pat_b8(SV_VL8));
                if (alignedWidth != srcWidth)
                    MainRowX5x5<compensation>(buffer, bufferDstTail, dst + dstWidth - BHA,
                        svwhilelt_b16(bufferDstTail, srcWidth),
                        svwhilelt_b16(bufferDstTail + BHA, srcWidth),
                        svwhilelt_b8(size_t(0), dstWidth - (dstWidth - BHA)));
            }
        }

        void ReduceGray5x5(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif
}
