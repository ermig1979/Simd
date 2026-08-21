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
        SIMD_INLINE void GetObjectMoments8(const svuint8_t& src, const svuint8_t& count, const svuint8_t& col,
            const svuint8_t& one, svuint32_t& n, svuint32_t& s, svuint32_t& sx, svuint32_t& sxx)
        {
            n = svdot_u32(n, count, one);
            s = svdot_u32(s, src, one);
            sx = svdot_u32(sx, src, col);

            svuint16_t srcEven = svmovlb_u16(src);
            svuint16_t srcOdd = svmovlt_u16(src);
            svuint16_t col2Even = svmullb_u16(col, col);
            svuint16_t col2Odd = svmullt_u16(col, col);
            sxx = svmlalb_u32(sxx, srcEven, col2Even);
            sxx = svmlalt_u32(sxx, srcEven, col2Even);
            sxx = svmlalb_u32(sxx, srcOdd, col2Odd);
            sxx = svmlalt_u32(sxx, srcOdd, col2Odd);
        }

        SIMD_INLINE void AddWide(svuint64_t& sum, const svuint32_t& value)
        {
            const svbool_t mask64 = svptrue_b64();
            sum = svadd_u64_x(mask64, sum, svunpklo_u64(value));
            sum = svadd_u64_x(mask64, sum, svunpkhi_u64(value));
        }

        SIMD_INLINE void AddWideMul(svuint64_t& sum, const svuint32_t& value, uint64_t factor)
        {
            const svbool_t mask64 = svptrue_b64();
            sum = svmla_n_u64_x(mask64, sum, svunpklo_u64(value), factor);
            sum = svmla_n_u64_x(mask64, sum, svunpkhi_u64(value), factor);
        }

        SIMD_INLINE void AddBlockMoments(svuint32_t n, svuint32_t s, svuint32_t sx, svuint32_t sxx,
            uint64_t x, uint64_t y, svuint64_t& n64, svuint64_t& s64, svuint64_t& sx64, svuint64_t& sy64,
            svuint64_t& sxx64, svuint64_t& sxy64, svuint64_t& syy64)
        {
            AddWide(n64, n);
            AddWide(s64, s);
            AddWide(sx64, sx);
            AddWideMul(sx64, s, x);
            AddWideMul(sy64, s, y);
            AddWide(sxx64, sxx);
            AddWideMul(sxx64, sx, x * 2);
            AddWideMul(sxx64, s, x * x);
            AddWideMul(sxy64, sx, y);
            AddWideMul(sxy64, s, x * y);
            AddWideMul(syy64, s, y * y);
        }

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            assert(src || mask);

            const size_t A = svcntb();
            const size_t B = A > 181 ? 181 : AlignLo(181, A);
            const svbool_t body = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            const svuint8_t _index = svdup_n_u8(index);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            const svuint8_t addA = svdup_n_u8((uint8_t)A);
            const svuint8_t col0 = svindex_u8(0, 1);

            svuint64_t n64 = svdup_n_u64(0);
            svuint64_t s64 = svdup_n_u64(0);
            svuint64_t sx64 = svdup_n_u64(0);
            svuint64_t sy64 = svdup_n_u64(0);
            svuint64_t sxx64 = svdup_n_u64(0);
            svuint64_t sxy64 = svdup_n_u64(0);
            svuint64_t syy64 = svdup_n_u64(0);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    size_t colE = Simd::Min(colB + B, width);
                    svuint32_t n0 = svdup_n_u32(0), n1 = svdup_n_u32(0);
                    svuint32_t s0 = svdup_n_u32(0), s1 = svdup_n_u32(0);
                    svuint32_t sx0 = svdup_n_u32(0), sx1 = svdup_n_u32(0);
                    svuint32_t sxx0 = svdup_n_u32(0), sxx1 = svdup_n_u32(0);
                    svuint8_t colIdx = col0;
                    size_t col = colB;

                    if (mask == NULL)
                    {
                        for (; col + 2 * A <= colE; col += 2 * A)
                        {
                            GetObjectMoments8(svld1_u8(body, src + col), one, colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                            GetObjectMoments8(svld1_u8(body, src + col + A), one, colIdx, one, n1, s1, sx1, sxx1);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        for (; col + A <= colE; col += A)
                        {
                            GetObjectMoments8(svld1_u8(body, src + col), one, colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        if (col < colE)
                        {
                            svbool_t tail = svwhilelt_b8(col, colE);
                            GetObjectMoments8(svld1_u8(tail, src + col), svsel_u8(tail, one, zero), colIdx, one, n0, s0, sx0, sxx0);
                        }
                    }
                    else if (src == NULL)
                    {
                        for (; col + 2 * A <= colE; col += 2 * A)
                        {
                            svbool_t equal0 = svcmpeq_u8(body, svld1_u8(body, mask + col), _index);
                            svuint8_t src0 = svand_u8_z(equal0, one, one);
                            GetObjectMoments8(src0, src0, colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                            svbool_t equal1 = svcmpeq_u8(body, svld1_u8(body, mask + col + A), _index);
                            svuint8_t src1 = svand_u8_z(equal1, one, one);
                            GetObjectMoments8(src1, src1, colIdx, one, n1, s1, sx1, sxx1);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        for (; col + A <= colE; col += A)
                        {
                            svbool_t equal = svcmpeq_u8(body, svld1_u8(body, mask + col), _index);
                            svuint8_t _src = svand_u8_z(equal, one, one);
                            GetObjectMoments8(_src, _src, colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        if (col < colE)
                        {
                            svbool_t tail = svwhilelt_b8(col, colE);
                            svbool_t equal = svcmpeq_u8(tail, svld1_u8(tail, mask + col), _index);
                            svuint8_t _src = svand_u8_z(equal, one, one);
                            GetObjectMoments8(_src, _src, colIdx, one, n0, s0, sx0, sxx0);
                        }
                    }
                    else
                    {
                        for (; col + 2 * A <= colE; col += 2 * A)
                        {
                            svbool_t equal0 = svcmpeq_u8(body, svld1_u8(body, mask + col), _index);
                            GetObjectMoments8(svsel_u8(equal0, svld1_u8(body, src + col), zero), svand_u8_z(equal0, one, one), colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                            svbool_t equal1 = svcmpeq_u8(body, svld1_u8(body, mask + col + A), _index);
                            GetObjectMoments8(svsel_u8(equal1, svld1_u8(body, src + col + A), zero), svand_u8_z(equal1, one, one), colIdx, one, n1, s1, sx1, sxx1);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        for (; col + A <= colE; col += A)
                        {
                            svbool_t equal = svcmpeq_u8(body, svld1_u8(body, mask + col), _index);
                            GetObjectMoments8(svsel_u8(equal, svld1_u8(body, src + col), zero), svand_u8_z(equal, one, one), colIdx, one, n0, s0, sx0, sxx0);
                            colIdx = svadd_u8_x(body, colIdx, addA);
                        }
                        if (col < colE)
                        {
                            svbool_t tail = svwhilelt_b8(col, colE);
                            svbool_t equal = svcmpeq_u8(tail, svld1_u8(tail, mask + col), _index);
                            GetObjectMoments8(svsel_u8(equal, svld1_u8(tail, src + col), zero), svand_u8_z(equal, one, one), colIdx, one, n0, s0, sx0, sxx0);
                        }
                    }

                    n0 = svadd_u32_x(body32, n0, n1);
                    s0 = svadd_u32_x(body32, s0, s1);
                    sx0 = svadd_u32_x(body32, sx0, sx1);
                    sxx0 = svadd_u32_x(body32, sxx0, sxx1);
                    AddBlockMoments(n0, s0, sx0, sxx0, colB, row, n64, s64, sx64, sy64, sxx64, sxy64, syy64);

                    colB = colE;
                }
                if (src)
                    src += srcStride;
                if (mask)
                    mask += maskStride;
            }

            const svbool_t mask64 = svptrue_b64();
            *n = svaddv_u64(mask64, n64);
            *s = svaddv_u64(mask64, s64);
            *sx = svaddv_u64(mask64, sx64);
            *sy = svaddv_u64(mask64, sy64);
            *sxx = svaddv_u64(mask64, sxx64);
            *sxy = svaddv_u64(mask64, sxy64);
            *syy = svaddv_u64(mask64, syy64);
        }

        void GetMoments(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t* area, uint64_t* x, uint64_t* y, uint64_t* xx, uint64_t* xy, uint64_t* yy)
        {
            uint64_t stub;
            GetObjectMoments(NULL, 0, width, height, mask, stride, index, &stub, area, x, y, xx, xy, yy);
        }
    }
#endif
}
