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
        SIMD_INLINE uint8_t Max9(const uint8_t* y[3], size_t x[3], int threshold)
        {
            uint8_t a[9];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]];
            a[3] = y[1][x[0]]; a[4] = y[1][x[1]]; a[5] = y[1][x[2]];
            a[6] = y[2][x[0]]; a[7] = y[2][x[1]]; a[8] = y[2][x[2]];

            uint8_t max = a[0];
            for (int i = 1; i < 9; ++i)
                max = max > a[i] ? max : a[i];

            if (1 >= threshold)
                return max;

            int num = 0;
            for (int i = 0; i < 9; ++i)
            {
                if (a[i] == max)
                    ++num;
            }
            return num >= threshold ? max : a[4];
        }

        template <size_t step> SIMD_INLINE void LoadSquare3x3(const uint8_t* y[3], size_t offset,
            svuint8_t& a0, svuint8_t& a1, svuint8_t& a2, svuint8_t& a3, svuint8_t& a4, svuint8_t& a5, svuint8_t& a6, svuint8_t& a7, svuint8_t& a8,
            const svbool_t& mask)
        {
            a0 = svld1_u8(mask, y[0] + offset - step);
            a1 = svld1_u8(mask, y[0] + offset);
            a2 = svld1_u8(mask, y[0] + offset + step);
            a3 = svld1_u8(mask, y[1] + offset - step);
            a4 = svld1_u8(mask, y[1] + offset);
            a5 = svld1_u8(mask, y[1] + offset + step);
            a6 = svld1_u8(mask, y[2] + offset - step);
            a7 = svld1_u8(mask, y[2] + offset);
            a8 = svld1_u8(mask, y[2] + offset + step);
        }

        SIMD_INLINE svuint8_t Max9(const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2, const svuint8_t& a3, const svuint8_t& a4,
            const svuint8_t& a5, const svuint8_t& a6, const svuint8_t& a7, const svuint8_t& a8, int threshold, const svbool_t& mask)
        {
            svuint8_t max = a0;
            max = svmax_u8_x(mask, max, a1);
            max = svmax_u8_x(mask, max, a2);
            max = svmax_u8_x(mask, max, a3);
            max = svmax_u8_x(mask, max, a4);
            max = svmax_u8_x(mask, max, a5);
            max = svmax_u8_x(mask, max, a6);
            max = svmax_u8_x(mask, max, a7);
            max = svmax_u8_x(mask, max, a8);

            if (1 >= threshold)
                return max;

            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a0), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a1), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a2), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a3), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a4), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a5), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a6), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a7), one, zero));
            count = svadd_u8_x(mask, count, svsel_u8(svcmpeq_u8(mask, max, a8), one, zero));

            return svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), max, a4);
        }

        template <size_t step> void MaxFilterSquare3x3(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(width > 2 && step * (width - 2) >= svcntb());

            const size_t A = svcntb();
            const size_t size = step * width;
            const size_t end = size - step;
            const uint8_t* y[3];
            size_t x[3];
            svuint8_t a0, a1, a2, a3, a4, a5, a6, a7, a8;

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                y[1] = src + srcStride * row;
                y[0] = row ? y[1] - srcStride : y[1];
                y[2] = row + 1 < height ? y[1] + srcStride : y[1];

                for (size_t col = 0; col < step; ++col)
                {
                    x[0] = col;
                    x[1] = col;
                    x[2] = col + step;
                    dst[col] = Max9(y, x, threshold);
                }

                for (size_t col = step; col < end; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, end);
                    LoadSquare3x3<step>(y, col, a0, a1, a2, a3, a4, a5, a6, a7, a8, mask);
                    svst1_u8(mask, dst + col, Max9(a0, a1, a2, a3, a4, a5, a6, a7, a8, threshold, mask));
                }

                for (size_t col = end; col < size; ++col)
                {
                    x[0] = col - step;
                    x[1] = col;
                    x[2] = col;
                    dst[col] = Max9(y, x, threshold);
                }
            }
        }

        void MaxFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare3x3<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare3x3<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare3x3<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare3x3<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }

        SIMD_INLINE uint8_t Max25(const uint8_t* y[5], size_t x[5], int threshold)
        {
            uint8_t a[25];
            a[0] = y[0][x[0]]; a[1] = y[0][x[1]]; a[2] = y[0][x[2]]; a[3] = y[0][x[3]]; a[4] = y[0][x[4]];
            a[5] = y[1][x[0]]; a[6] = y[1][x[1]]; a[7] = y[1][x[2]]; a[8] = y[1][x[3]]; a[9] = y[1][x[4]];
            a[10] = y[2][x[0]]; a[11] = y[2][x[1]]; a[12] = y[2][x[2]]; a[13] = y[2][x[3]]; a[14] = y[2][x[4]];
            a[15] = y[3][x[0]]; a[16] = y[3][x[1]]; a[17] = y[3][x[2]]; a[18] = y[3][x[3]]; a[19] = y[3][x[4]];
            a[20] = y[4][x[0]]; a[21] = y[4][x[1]]; a[22] = y[4][x[2]]; a[23] = y[4][x[3]]; a[24] = y[4][x[4]];

            uint8_t max = a[0];
            for (int i = 1; i < 25; ++i)
                max = max > a[i] ? max : a[i];

            if (1 >= threshold)
                return max;

            int num = 0;
            for (int i = 0; i < 25; ++i)
            {
                if (a[i] == max)
                    ++num;
            }
            return num >= threshold ? max : a[12];
        }

        SIMD_INLINE const uint8_t* SrcRow(const uint8_t* src, size_t srcStride, size_t height, int row)
        {
            if (row < 0)
                row = 0;
            else if (row >= (int)height)
                row = (int)height - 1;
            return src + srcStride * row;
        }

        SIMD_INLINE void Edge25(const uint8_t* y0, const uint8_t* y1, const uint8_t* y2, const uint8_t* y3, const uint8_t* y4,
            uint8_t* dst, size_t x0, size_t x1, size_t x2, size_t x3, size_t x4, int threshold)
        {
            const uint8_t* y[5] = { y0, y1, y2, y3, y4 };
            size_t x[5] = { x0, x1, x2, x3, x4 };
            dst[x2] = Max25(y, x, threshold);
        }

        template <size_t step> SIMD_INLINE void Edges(const uint8_t* y0, const uint8_t* y1, const uint8_t* y2,
            const uint8_t* y3, const uint8_t* y4, uint8_t* dst, size_t size, size_t end, int threshold)
        {
            for (size_t col = 0; col < 2 * step; ++col)
            {
                size_t x0 = col < step ? col : col - step;
                Edge25(y0, y1, y2, y3, y4, dst, x0, x0, col, col + step, col + 2 * step, threshold);
            }
            for (size_t col = end; col < size; ++col)
            {
                size_t x3 = col + step < size ? col + step : col;
                size_t x4 = col + 2 * step < size ? col + 2 * step : x3;
                Edge25(y0, y1, y2, y3, y4, dst, col - 2 * step, col - step, col, x3, x4, threshold);
            }
        }

        template <size_t step> SIMD_INLINE void Load5(const uint8_t* src, const svbool_t& mask,
            svuint8_t& a0, svuint8_t& a1, svuint8_t& a2, svuint8_t& a3, svuint8_t& a4)
        {
            a0 = svld1_u8(mask, src - 2 * step);
            a1 = svld1_u8(mask, src - step);
            a2 = svld1_u8(mask, src);
            a3 = svld1_u8(mask, src + step);
            a4 = svld1_u8(mask, src + 2 * step);
        }

        SIMD_INLINE svuint8_t Max5(const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2,
            const svuint8_t& a3, const svuint8_t& a4, const svbool_t& mask)
        {
            return svmax_u8_x(mask, svmax_u8_x(mask, a0, a1), svmax_u8_x(mask, svmax_u8_x(mask, a2, a3), a4));
        }

        template <size_t step> SIMD_INLINE svuint8_t HorizMax(const uint8_t* src, const svbool_t& mask)
        {
            svuint8_t a0, a1, a2, a3, a4;
            Load5<step>(src, mask, a0, a1, a2, a3, a4);
            return Max5(a0, a1, a2, a3, a4, mask);
        }

        SIMD_INLINE void AddEq(svuint8_t& count, const svuint8_t& value, const svuint8_t& max, const svuint8_t& one, const svbool_t& mask)
        {
            count = svadd_u8_m(svcmpeq_u8(mask, max, value), count, one);
        }

        SIMD_INLINE void AddEq5(svuint8_t& count, const svuint8_t& a0, const svuint8_t& a1, const svuint8_t& a2,
            const svuint8_t& a3, const svuint8_t& a4, const svuint8_t& max, const svuint8_t& one, const svbool_t& mask)
        {
            AddEq(count, a0, max, one, mask);
            AddEq(count, a1, max, one, mask);
            AddEq(count, a2, max, one, mask);
            AddEq(count, a3, max, one, mask);
            AddEq(count, a4, max, one, mask);
        }

        template <size_t step> SIMD_INLINE void Max1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, uint8_t* dst, const svbool_t& mask)
        {
            svst1_u8(mask, dst, Max5(HorizMax<step>(src0, mask), HorizMax<step>(src1, mask), HorizMax<step>(src2, mask),
                HorizMax<step>(src3, mask), HorizMax<step>(src4, mask), mask));
        }

        template <size_t step> SIMD_INLINE void Max2(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, const svbool_t& mask)
        {
            svuint8_t h0 = HorizMax<step>(src0, mask);
            svuint8_t h1 = HorizMax<step>(src1, mask);
            svuint8_t h2 = HorizMax<step>(src2, mask);
            svuint8_t h3 = HorizMax<step>(src3, mask);
            svuint8_t h4 = HorizMax<step>(src4, mask);
            svuint8_t h5 = HorizMax<step>(src5, mask);
            svst1_u8(mask, dst0, Max5(h0, h1, h2, h3, h4, mask));
            svst1_u8(mask, dst1, Max5(h1, h2, h3, h4, h5, mask));
        }

        template <size_t step> SIMD_INLINE void Max4(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
            const uint8_t* src4, const uint8_t* src5, const uint8_t* src6, const uint8_t* src7,
            uint8_t* dst0, uint8_t* dst1, uint8_t* dst2, uint8_t* dst3, const svbool_t& mask)
        {
            svuint8_t h0 = HorizMax<step>(src0, mask);
            svuint8_t h1 = HorizMax<step>(src1, mask);
            svuint8_t h2 = HorizMax<step>(src2, mask);
            svuint8_t h3 = HorizMax<step>(src3, mask);
            svuint8_t h4 = HorizMax<step>(src4, mask);
            svuint8_t h5 = HorizMax<step>(src5, mask);
            svuint8_t h6 = HorizMax<step>(src6, mask);
            svuint8_t h7 = HorizMax<step>(src7, mask);
            svst1_u8(mask, dst0, Max5(h0, h1, h2, h3, h4, mask));
            svst1_u8(mask, dst1, Max5(h1, h2, h3, h4, h5, mask));
            svst1_u8(mask, dst2, Max5(h2, h3, h4, h5, h6, mask));
            svst1_u8(mask, dst3, Max5(h3, h4, h5, h6, h7, mask));
        }

        template <size_t step> SIMD_INLINE void MaxThresh1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, uint8_t* dst, int threshold, const svbool_t& mask)
        {
            svuint8_t a00, a01, a02, a03, a04;
            svuint8_t a10, a11, a12, a13, a14;
            svuint8_t a20, a21, a22, a23, a24;
            svuint8_t a30, a31, a32, a33, a34;
            svuint8_t a40, a41, a42, a43, a44;
            Load5<step>(src0, mask, a00, a01, a02, a03, a04);
            Load5<step>(src1, mask, a10, a11, a12, a13, a14);
            Load5<step>(src2, mask, a20, a21, a22, a23, a24);
            Load5<step>(src3, mask, a30, a31, a32, a33, a34);
            Load5<step>(src4, mask, a40, a41, a42, a43, a44);
            svuint8_t max = Max5(
                Max5(a00, a01, a02, a03, a04, mask),
                Max5(a10, a11, a12, a13, a14, mask),
                Max5(a20, a21, a22, a23, a24, mask),
                Max5(a30, a31, a32, a33, a34, mask),
                Max5(a40, a41, a42, a43, a44, mask), mask);
            svuint8_t count = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            AddEq5(count, a00, a01, a02, a03, a04, max, one, mask);
            AddEq5(count, a10, a11, a12, a13, a14, max, one, mask);
            AddEq5(count, a20, a21, a22, a23, a24, max, one, mask);
            AddEq5(count, a30, a31, a32, a33, a34, max, one, mask);
            AddEq5(count, a40, a41, a42, a43, a44, max, one, mask);
            svst1_u8(mask, dst, svsel_u8(svcmpge_n_u8(mask, count, (uint8_t)threshold), max, a22));
        }

        template <size_t step> SIMD_INLINE void MaxThresh2(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, int threshold, const svbool_t& mask)
        {
            svuint8_t a00, a01, a02, a03, a04;
            svuint8_t a10, a11, a12, a13, a14;
            svuint8_t a20, a21, a22, a23, a24;
            svuint8_t a30, a31, a32, a33, a34;
            svuint8_t a40, a41, a42, a43, a44;
            svuint8_t a50, a51, a52, a53, a54;
            Load5<step>(src0, mask, a00, a01, a02, a03, a04);
            Load5<step>(src1, mask, a10, a11, a12, a13, a14);
            Load5<step>(src2, mask, a20, a21, a22, a23, a24);
            Load5<step>(src3, mask, a30, a31, a32, a33, a34);
            Load5<step>(src4, mask, a40, a41, a42, a43, a44);
            Load5<step>(src5, mask, a50, a51, a52, a53, a54);
            svuint8_t h0 = Max5(a00, a01, a02, a03, a04, mask);
            svuint8_t h1 = Max5(a10, a11, a12, a13, a14, mask);
            svuint8_t h2 = Max5(a20, a21, a22, a23, a24, mask);
            svuint8_t h3 = Max5(a30, a31, a32, a33, a34, mask);
            svuint8_t h4 = Max5(a40, a41, a42, a43, a44, mask);
            svuint8_t h5 = Max5(a50, a51, a52, a53, a54, mask);
            svuint8_t max0 = Max5(h0, h1, h2, h3, h4, mask);
            svuint8_t max1 = Max5(h1, h2, h3, h4, h5, mask);
            svuint8_t count0 = svdup_n_u8(0);
            svuint8_t count1 = svdup_n_u8(0);
            const svuint8_t one = svdup_n_u8(1);
            AddEq5(count0, a00, a01, a02, a03, a04, max0, one, mask);
            AddEq5(count0, a10, a11, a12, a13, a14, max0, one, mask);
            AddEq5(count1, a10, a11, a12, a13, a14, max1, one, mask);
            AddEq5(count0, a20, a21, a22, a23, a24, max0, one, mask);
            AddEq5(count1, a20, a21, a22, a23, a24, max1, one, mask);
            AddEq5(count0, a30, a31, a32, a33, a34, max0, one, mask);
            AddEq5(count1, a30, a31, a32, a33, a34, max1, one, mask);
            AddEq5(count0, a40, a41, a42, a43, a44, max0, one, mask);
            AddEq5(count1, a40, a41, a42, a43, a44, max1, one, mask);
            AddEq5(count1, a50, a51, a52, a53, a54, max1, one, mask);
            svst1_u8(mask, dst0, svsel_u8(svcmpge_n_u8(mask, count0, (uint8_t)threshold), max0, a22));
            svst1_u8(mask, dst1, svsel_u8(svcmpge_n_u8(mask, count1, (uint8_t)threshold), max1, a32));
        }

        template <size_t step> void MaxBody1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, uint8_t* dst, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = 2 * step;
            for (; col + A2 <= end; col += A2)
            {
                Max1<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, dst + col, all);
                Max1<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, dst + col + A, all);
            }
            for (; col < end; col += A)
                Max1<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, dst + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void MaxBody2(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = 2 * step;
            for (; col + A2 <= end; col += A2)
            {
                Max2<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col, dst0 + col, dst1 + col, all);
                Max2<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, src5 + col + A, dst0 + col + A, dst1 + col + A, all);
            }
            for (; col < end; col += A)
                Max2<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col, dst0 + col, dst1 + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void MaxBody4(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
            const uint8_t* src4, const uint8_t* src5, const uint8_t* src6, const uint8_t* src7,
            uint8_t* dst0, uint8_t* dst1, uint8_t* dst2, uint8_t* dst3,
            size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = 2 * step;
            for (; col + A2 <= end; col += A2)
            {
                Max4<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col, src6 + col, src7 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, all);
                Max4<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, src5 + col + A, src6 + col + A, src7 + col + A,
                    dst0 + col + A, dst1 + col + A, dst2 + col + A, dst3 + col + A, all);
            }
            for (; col < end; col += A)
                Max4<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col, src6 + col, src7 + col,
                    dst0 + col, dst1 + col, dst2 + col, dst3 + col, svwhilelt_b8(col, end));
        }

        template <size_t step> void MaxThreshBody1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
            const uint8_t* src3, const uint8_t* src4, uint8_t* dst, int threshold, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = 2 * step;
            for (; col + A2 <= end; col += A2)
            {
                MaxThresh1<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, dst + col, threshold, all);
                MaxThresh1<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, dst + col + A, threshold, all);
            }
            for (; col < end; col += A)
                MaxThresh1<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, dst + col, threshold, svwhilelt_b8(col, end));
        }

        template <size_t step> void MaxThreshBody2(
            const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const uint8_t* src4, const uint8_t* src5,
            uint8_t* dst0, uint8_t* dst1, int threshold, size_t end, size_t A, size_t A2, const svbool_t& all)
        {
            size_t col = 2 * step;
            for (; col + A2 <= end; col += A2)
            {
                MaxThresh2<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col, dst0 + col, dst1 + col, threshold, all);
                MaxThresh2<step>(src0 + col + A, src1 + col + A, src2 + col + A, src3 + col + A, src4 + col + A, src5 + col + A,
                    dst0 + col + A, dst1 + col + A, threshold, all);
            }
            for (; col < end; col += A)
                MaxThresh2<step>(src0 + col, src1 + col, src2 + col, src3 + col, src4 + col, src5 + col,
                    dst0 + col, dst1 + col, threshold, svwhilelt_b8(col, end));
        }

        template <size_t step> void MaxFilterSquare5x5(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(width > 4 && step * (width - 4) >= svcntb());

            const size_t A = svcntb();
            const size_t A2 = A * 2;
            const size_t size = step * width;
            const size_t end = size - 2 * step;
            const svbool_t all = svptrue_b8();
            const bool fast = threshold <= 1;

            size_t row = 0;
            if (fast)
            {
                for (; row + 4 <= height; row += 4)
                {
                    const uint8_t* src0 = SrcRow(src, srcStride, height, (int)row - 2);
                    const uint8_t* src1 = SrcRow(src, srcStride, height, (int)row - 1);
                    const uint8_t* src2 = SrcRow(src, srcStride, height, (int)row);
                    const uint8_t* src3 = SrcRow(src, srcStride, height, (int)row + 1);
                    const uint8_t* src4 = SrcRow(src, srcStride, height, (int)row + 2);
                    const uint8_t* src5 = SrcRow(src, srcStride, height, (int)row + 3);
                    const uint8_t* src6 = SrcRow(src, srcStride, height, (int)row + 4);
                    const uint8_t* src7 = SrcRow(src, srcStride, height, (int)row + 5);
                    uint8_t* dst0 = dst + dstStride * row;
                    uint8_t* dst1 = dst0 + dstStride;
                    uint8_t* dst2 = dst1 + dstStride;
                    uint8_t* dst3 = dst2 + dstStride;
                    Edges<step>(src0, src1, src2, src3, src4, dst0, size, end, threshold);
                    Edges<step>(src1, src2, src3, src4, src5, dst1, size, end, threshold);
                    Edges<step>(src2, src3, src4, src5, src6, dst2, size, end, threshold);
                    Edges<step>(src3, src4, src5, src6, src7, dst3, size, end, threshold);
                    MaxBody4<step>(src0, src1, src2, src3, src4, src5, src6, src7, dst0, dst1, dst2, dst3, end, A, A2, all);
                }
            }

            for (; row + 2 <= height; row += 2)
            {
                const uint8_t* src0 = SrcRow(src, srcStride, height, (int)row - 2);
                const uint8_t* src1 = SrcRow(src, srcStride, height, (int)row - 1);
                const uint8_t* src2 = SrcRow(src, srcStride, height, (int)row);
                const uint8_t* src3 = SrcRow(src, srcStride, height, (int)row + 1);
                const uint8_t* src4 = SrcRow(src, srcStride, height, (int)row + 2);
                const uint8_t* src5 = SrcRow(src, srcStride, height, (int)row + 3);
                uint8_t* dst0 = dst + dstStride * row;
                uint8_t* dst1 = dst0 + dstStride;
                Edges<step>(src0, src1, src2, src3, src4, dst0, size, end, threshold);
                Edges<step>(src1, src2, src3, src4, src5, dst1, size, end, threshold);
                if (fast)
                    MaxBody2<step>(src0, src1, src2, src3, src4, src5, dst0, dst1, end, A, A2, all);
                else
                    MaxThreshBody2<step>(src0, src1, src2, src3, src4, src5, dst0, dst1, threshold, end, A, A2, all);
            }

            if (row < height)
            {
                const uint8_t* src0 = SrcRow(src, srcStride, height, (int)row - 2);
                const uint8_t* src1 = SrcRow(src, srcStride, height, (int)row - 1);
                const uint8_t* src2 = SrcRow(src, srcStride, height, (int)row);
                const uint8_t* src3 = SrcRow(src, srcStride, height, (int)row + 1);
                const uint8_t* src4 = SrcRow(src, srcStride, height, (int)row + 2);
                uint8_t* dst0 = dst + dstStride * row;
                Edges<step>(src0, src1, src2, src3, src4, dst0, size, end, threshold);
                if (fast)
                    MaxBody1<step>(src0, src1, src2, src3, src4, dst0, end, A, A2, all);
                else
                    MaxThreshBody1<step>(src0, src1, src2, src3, src4, dst0, threshold, end, A, A2, all);
            }
        }

        void MaxFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MaxFilterSquare5x5<1>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 2: MaxFilterSquare5x5<2>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 3: MaxFilterSquare5x5<3>(src, srcStride, width, height, dst, dstStride, threshold); break;
            case 4: MaxFilterSquare5x5<4>(src, srcStride, width, height, dst, dstStride, threshold); break;
            }
        }
    }
#endif
}
