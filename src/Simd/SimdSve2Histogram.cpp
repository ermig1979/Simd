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
            template<class T> struct Buffer
            {
                Buffer(size_t rowSize, size_t histogramSize)
                {
                    _p = Allocate(sizeof(T) * rowSize + 4 * sizeof(uint32_t) * histogramSize);
                    v = (T*)_p;
                    h[0] = (uint32_t*)(v + rowSize);
                    h[1] = h[0] + histogramSize;
                    h[2] = h[1] + histogramSize;
                    h[3] = h[2] + histogramSize;
                    memset(h[0], 0, 4 * sizeof(uint32_t) * histogramSize);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                T* v;
                uint32_t* h[4];
            private:
                void* _p;
            };
        }

        SIMD_INLINE svuint8_t Average2(const svuint8_t& a, const svuint8_t& b)
        {
            svuint16_t lo = svadd_n_u16_x(svptrue_b16(), svaddlb_u16(a, b), 1);
            svuint16_t hi = svadd_n_u16_x(svptrue_b16(), svaddlt_u16(a, b), 1);
            return svshrnt_n_u16(svshrnb_n_u16(lo, 1), hi, 1);
        }

        SIMD_INLINE svuint8_t AbsSecondDerivative(const uint8_t* src, ptrdiff_t step, const svbool_t& mask)
        {
            const svuint8_t s0 = svld1_u8(mask, src - step);
            const svuint8_t s1 = svld1_u8(mask, src);
            const svuint8_t s2 = svld1_u8(mask, src + step);
            return svabd_u8_x(mask, Average2(s0, s2), s1);
        }

        SIMD_INLINE void AbsSecondDerivative(const uint8_t* src, ptrdiff_t colStep, ptrdiff_t rowStep, uint8_t* dst, const svbool_t& mask)
        {
            const svuint8_t sdX = AbsSecondDerivative(src, colStep, mask);
            const svuint8_t sdY = AbsSecondDerivative(src, rowStep, mask);
            svst1_u8(mask, dst, svmax_u8_x(mask, sdY, sdX));
        }

        SIMD_INLINE void SumHistograms(uint32_t* src, size_t start, uint32_t* dst)
        {
            uint32_t* src0 = src + start;
            uint32_t* src1 = src0 + start + HISTOGRAM_SIZE;
            uint32_t* src2 = src1 + start + HISTOGRAM_SIZE;
            uint32_t* src3 = src2 + start + HISTOGRAM_SIZE;
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                dst[i] = src0[i] + src1[i] + src2[i] + src3[i];
        }

        void AbsSecondDerivativeHistogram(const uint8_t* src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t* histogram)
        {
            assert(width > 2 * indent && height > 2 * indent && indent >= step);

            size_t A = svlen(svuint8_t());
            src += indent * (stride + 1);
            height -= 2 * indent;
            width -= 2 * indent;

            Buffer<uint8_t> buffer(AlignHiAny(width, A), HISTOGRAM_SIZE);
            ptrdiff_t rowStep = step * stride;
            size_t widthA = AlignLoAny(width, A);
            size_t alignedWidth4 = Simd::AlignLo(width, 4);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsSecondDerivative(src + col, step, rowStep, buffer.v + col, body);
                if (widthA < width)
                    AbsSecondDerivative(src + col, step, rowStep, buffer.v + col, tail);

                for (col = 0; col < alignedWidth4; col += 4)
                {
                    ++buffer.h[0][buffer.v[col + 0]];
                    ++buffer.h[1][buffer.v[col + 1]];
                    ++buffer.h[2][buffer.v[col + 2]];
                    ++buffer.h[3][buffer.v[col + 3]];
                }
                for (; col < width; ++col)
                    ++buffer.h[0][buffer.v[col + 0]];
                src += stride;
            }

            SumHistograms(buffer.h[0], 0, histogram);
        }
    }
#endif
}
