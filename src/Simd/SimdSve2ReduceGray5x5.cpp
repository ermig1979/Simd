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
                Buffer(size_t width, size_t tail)
                {
                    _p = Allocate(sizeof(uint16_t) * (5 * width + tail));
                    in0 = (uint16_t*)_p;
                    in1 = in0 + width;
                    out0 = in1 + width;
                    out1 = out0 + width;
                    dst = out1 + width + tail / 2;
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

        SIMD_INLINE void FirstRow5x5(const uint8_t* src, Buffer& buffer, size_t col,
            const svbool_t& mask8, const svbool_t& mask16Lo, const svbool_t& mask16Hi)
        {
            const size_t HA = svcnth();
            svuint8_t s = svld1_u8(mask8, src + col);
            svuint16_t lo = svunpklo_u16(s);
            svuint16_t hi = svunpkhi_u16(s);
            svst1_u16(mask16Lo, buffer.in0 + col, lo);
            svst1_u16(mask16Hi, buffer.in0 + col + HA, hi);
            svst1_u16(mask16Lo, buffer.in1 + col, svmul_n_u16_x(mask16Lo, lo, 5));
            svst1_u16(mask16Hi, buffer.in1 + col + HA, svmul_n_u16_x(mask16Hi, hi, 5));
        }

        SIMD_INLINE void MainRowY5x5(const uint8_t* odd, const uint8_t* even, Buffer& buffer, size_t col,
            const svbool_t& mask8, const svbool_t& mask16Lo, const svbool_t& mask16Hi)
        {
            const size_t HA = svcnth();
            svuint8_t sOdd = svld1_u8(mask8, odd + col);
            svuint8_t sEven = svld1_u8(mask8, even + col);
            svuint16_t oddLo = svunpklo_u16(sOdd);
            svuint16_t oddHi = svunpkhi_u16(sOdd);
            svuint16_t evenLo = svunpklo_u16(sEven);
            svuint16_t evenHi = svunpkhi_u16(sEven);

            svuint16_t cpLo = svmul_n_u16_x(mask16Lo, oddLo, 4);
            svuint16_t c0Lo = svld1_u16(mask16Lo, buffer.in0 + col);
            svuint16_t c1Lo = svld1_u16(mask16Lo, buffer.in1 + col);
            svst1_u16(mask16Lo, buffer.dst + col, svadd_u16_x(mask16Lo, evenLo,
                svadd_u16_x(mask16Lo, c1Lo, svadd_u16_x(mask16Lo, cpLo, svmul_n_u16_x(mask16Lo, c0Lo, 6)))));
            svst1_u16(mask16Lo, buffer.out1 + col, svadd_u16_x(mask16Lo, c0Lo, cpLo));
            svst1_u16(mask16Lo, buffer.out0 + col, evenLo);

            svuint16_t cpHi = svmul_n_u16_x(mask16Hi, oddHi, 4);
            svuint16_t c0Hi = svld1_u16(mask16Hi, buffer.in0 + col + HA);
            svuint16_t c1Hi = svld1_u16(mask16Hi, buffer.in1 + col + HA);
            svst1_u16(mask16Hi, buffer.dst + col + HA, svadd_u16_x(mask16Hi, evenHi,
                svadd_u16_x(mask16Hi, c1Hi, svadd_u16_x(mask16Hi, cpHi, svmul_n_u16_x(mask16Hi, c0Hi, 6)))));
            svst1_u16(mask16Hi, buffer.out1 + col + HA, svadd_u16_x(mask16Hi, c0Hi, cpHi));
            svst1_u16(mask16Hi, buffer.out0 + col + HA, evenHi);
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

        template <bool compensation> SIMD_INLINE void MainRowX5x5(Buffer& buffer, size_t col, uint8_t* dst,
            const svbool_t& mask16, const svbool_t& mask8)
        {
            const size_t HA = svcnth();
            svuint16_t lo = MainRowX5x5<compensation>(buffer.dst + col, mask16);
            svuint16_t hi = MainRowX5x5<compensation>(buffer.dst + col + HA, mask16);
            svst1_u8(mask8, dst, svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi)));
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > svcntb());

            const size_t A = svcntb(), HA = svcnth();
            const svbool_t body8 = svptrue_b8(), body16 = svptrue_b16();
            size_t alignedWidth = AlignLo(srcWidth, A);
            size_t bufferDstTail = AlignHi(srcWidth - A, 2);

            Buffer buffer(AlignHi(srcWidth, A), A);

            for (size_t col = 0; col < alignedWidth; col += A)
                FirstRow5x5(src, buffer, col, body8, body16, body16);
            if (alignedWidth != srcWidth)
            {
                size_t col = srcWidth - A;
                FirstRow5x5(src, buffer, col, svwhilelt_b8(col, srcWidth),
                    svwhilelt_b16(col, srcWidth), svwhilelt_b16(col + HA, srcWidth));
            }
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t* odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t* even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t col = 0; col < alignedWidth; col += A)
                    MainRowY5x5(odd, even, buffer, col, body8, body16, body16);
                if (alignedWidth != srcWidth)
                {
                    size_t col = srcWidth - A;
                    MainRowY5x5(odd, even, buffer, col, svwhilelt_b8(col, srcWidth),
                        svwhilelt_b16(col, srcWidth), svwhilelt_b16(col + HA, srcWidth));
                }

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += A, dstCol += HA)
                    MainRowX5x5<compensation>(buffer, srcCol, dst + dstCol, body16, svwhilelt_b8(dstCol, dstWidth));
                if (alignedWidth != srcWidth)
                {
                    MainRowX5x5<compensation>(buffer, bufferDstTail, dst + dstWidth - HA,
                        svwhilelt_b16(bufferDstTail, srcWidth), svwhilelt_b8(dstWidth - HA, dstWidth));
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
