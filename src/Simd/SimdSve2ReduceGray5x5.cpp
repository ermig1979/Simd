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
                    const size_t A = svcntb();
                    _p = Allocate(sizeof(uint16_t) * (5 * width + A));
                    in0 = (uint16_t*)_p;
                    in1 = in0 + width;
                    out0 = in1 + width;
                    out1 = out0 + width;
                    dst = out1 + width + A / 2;
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

        template <bool compensation> SIMD_INLINE svuint16_t DivideBy256(const svuint16_t& value);

        template <> SIMD_INLINE svuint16_t DivideBy256<true>(const svuint16_t& value)
        {
            return svrshr_n_u16_x(svptrue_b16(), value, 8);
        }

        template <> SIMD_INLINE svuint16_t DivideBy256<false>(const svuint16_t& value)
        {
            return svlsr_n_u16_x(svptrue_b16(), value, 8);
        }

        SIMD_INLINE void FirstRow5x5(const svuint16_t& src, Buffer& buffer, size_t offset)
        {
            const svbool_t mask = svptrue_b16();
            svst1_u16(mask, buffer.in0 + offset, src);
            svst1_u16(mask, buffer.in1 + offset, svmla_n_u16_x(mask, src, src, 4));
        }

        SIMD_INLINE void FirstRow5x5(const uint8_t* src, Buffer& buffer, size_t offset, size_t HA)
        {
            svuint8_t s = svld1_u8(svptrue_b8(), src + offset);
            FirstRow5x5(svunpklo_u16(s), buffer, offset);
            FirstRow5x5(svunpkhi_u16(s), buffer, offset + HA);
        }

        SIMD_INLINE void MainRowY5x5(const svuint16_t& odd, const svuint16_t& even, Buffer& buffer, size_t offset)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t cp = svlsl_n_u16_x(mask, odd, 2);
            svuint16_t c0 = svld1_u16(mask, buffer.in0 + offset);
            svuint16_t c1 = svld1_u16(mask, buffer.in1 + offset);
            svst1_u16(mask, buffer.dst + offset, svmla_n_u16_x(mask, svadd_u16_x(mask, even, svadd_u16_x(mask, c1, cp)), c0, 6));
            svst1_u16(mask, buffer.out1 + offset, svadd_u16_x(mask, c0, cp));
            svst1_u16(mask, buffer.out0 + offset, even);
        }

        SIMD_INLINE void MainRowY5x5(const uint8_t* odd, const uint8_t* even, Buffer& buffer, size_t offset, size_t HA)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t o = svld1_u8(all, odd + offset);
            svuint8_t e = svld1_u8(all, even + offset);
            MainRowY5x5(svunpklo_u16(o), svunpklo_u16(e), buffer, offset);
            MainRowY5x5(svunpkhi_u16(o), svunpkhi_u16(e), buffer, offset + HA);
        }

        template <bool compensation> SIMD_INLINE svuint16_t MainRowX5x5(const uint16_t* row)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t t0 = svld1_u16(mask, row - 2);
            svuint16_t t1 = svld1_u16(mask, row - 1);
            svuint16_t t2 = svld1_u16(mask, row);
            svuint16_t t3 = svld1_u16(mask, row + 1);
            svuint16_t t4 = svld1_u16(mask, row + 2);
            svuint16_t sum = svadd_u16_x(mask, t0, t4);
            sum = svmla_n_u16_x(mask, sum, svadd_u16_x(mask, t1, t3), 4);
            sum = svmla_n_u16_x(mask, sum, t2, 6);
            return DivideBy256<compensation>(sum);
        }

        template <bool compensation> SIMD_INLINE svuint16_t MainRowXEven(const uint16_t* row, size_t HA)
        {
            return svuzp1_u16(MainRowX5x5<compensation>(row), MainRowX5x5<compensation>(row + HA));
        }

        template <bool compensation> SIMD_INLINE void MainRowX5x5(const uint16_t* row, uint8_t* dst, size_t HA)
        {
            svst1b_u16(svptrue_b16(), dst, MainRowXEven<compensation>(row, HA));
        }

        template <bool compensation> SIMD_INLINE void MainRowX5x5(const uint16_t* row, uint8_t* dst, size_t A, size_t HA)
        {
            svst1_u8(svptrue_b8(), dst, svuzp1_u8(
                svreinterpret_u8_u16(MainRowXEven<compensation>(row, HA)),
                svreinterpret_u8_u16(MainRowXEven<compensation>(row + A, HA))));
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const size_t A = svcntb(), HA = A / 2, DA = A * 2, QA = A * 4;
            const size_t alignedA = AlignLo(srcWidth, A);
            const size_t alignedDa = AlignLo(srcWidth, DA);
            const size_t alignedQa = AlignLo(srcWidth, QA);

            Buffer buffer(AlignHi(srcWidth, A));

            size_t col = 0;
            for (; col < alignedQa; col += QA)
            {
                FirstRow5x5(src, buffer, col, HA);
                FirstRow5x5(src, buffer, col + A, HA);
                FirstRow5x5(src, buffer, col + DA, HA);
                FirstRow5x5(src, buffer, col + DA + A, HA);
            }
            for (; col < alignedA; col += A)
                FirstRow5x5(src, buffer, col, HA);
            if (alignedA != srcWidth)
                FirstRow5x5(src, buffer, srcWidth - A, HA);
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t* even = odd + (row < srcHeight - 1 ? srcStride : 0);

                col = 0;
                for (; col < alignedQa; col += QA)
                {
                    MainRowY5x5(odd, even, buffer, col, HA);
                    MainRowY5x5(odd, even, buffer, col + A, HA);
                    MainRowY5x5(odd, even, buffer, col + DA, HA);
                    MainRowY5x5(odd, even, buffer, col + DA + A, HA);
                }
                for (; col < alignedA; col += A)
                    MainRowY5x5(odd, even, buffer, col, HA);
                if (alignedA != srcWidth)
                    MainRowY5x5(odd, even, buffer, srcWidth - A, HA);

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                if (srcWidth >= DA)
                {
                    size_t srcCol = 0, dstCol = 0;
                    for (; srcCol < alignedQa; srcCol += QA, dstCol += DA)
                    {
                        MainRowX5x5<compensation>(buffer.dst + srcCol, dst + dstCol, A, HA);
                        MainRowX5x5<compensation>(buffer.dst + srcCol + DA, dst + dstCol + A, A, HA);
                    }
                    for (; srcCol < alignedDa; srcCol += DA, dstCol += A)
                        MainRowX5x5<compensation>(buffer.dst + srcCol, dst + dstCol, A, HA);
                    if (alignedDa != srcWidth)
                    {
                        srcCol = AlignHi(srcWidth - DA, 2);
                        dstCol = dstWidth - A;
                        MainRowX5x5<compensation>(buffer.dst + srcCol, dst + dstCol, A, HA);
                    }
                }
                else
                {
                    size_t srcCol = 0, dstCol = 0;
                    for (; srcCol < alignedA; srcCol += A, dstCol += HA)
                        MainRowX5x5<compensation>(buffer.dst + srcCol, dst + dstCol, HA);
                    if (alignedA != srcWidth)
                    {
                        srcCol = AlignHi(srcWidth - A, 2);
                        dstCol = dstWidth - HA;
                        MainRowX5x5<compensation>(buffer.dst + srcCol, dst + dstCol, HA);
                    }
                }
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
