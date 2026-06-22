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
        SIMD_INLINE void GetObjectMoments16(const svuint16_t& src, const svuint16_t& col, size_t offset, size_t end, svuint32_t& sx, svuint32_t& sxx)
        {
            const size_t F = svcntw();

            svbool_t lo = svwhilelt_b32(offset, end);
            svuint32_t srcLo = svunpklo_u32(src);
            svuint32_t colLo = svunpklo_u32(col);
            sx = svmla_u32_m(lo, sx, srcLo, colLo);
            sxx = svmla_u32_m(lo, sxx, srcLo, svmul_u32_x(lo, colLo, colLo));

            svbool_t hi = svwhilelt_b32(offset + F, end);
            svuint32_t srcHi = svunpkhi_u32(src);
            svuint32_t colHi = svunpkhi_u32(col);
            sx = svmla_u32_m(hi, sx, srcHi, colHi);
            sxx = svmla_u32_m(hi, sxx, srcHi, svmul_u32_x(hi, colHi, colHi));
        }

        SIMD_INLINE void GetObjectMoments8(const svuint8_t& src, const svuint8_t& count, size_t offset, size_t end, const svuint8_t& one,
            svuint32_t& n, svuint32_t& s, svuint32_t& sx, svuint32_t& sxx)
        {
            const size_t HA = svcnth();

            n = svdot_u32(n, count, one);
            s = svdot_u32(s, src, one);

            svuint16_t srcLo = svunpklo_u16(src);
            svuint16_t colLo = svindex_u16((uint16_t)offset, 1);
            GetObjectMoments16(srcLo, colLo, offset, end, sx, sxx);

            svuint16_t srcHi = svunpkhi_u16(src);
            svuint16_t colHi = svindex_u16((uint16_t)(offset + HA), 1);
            GetObjectMoments16(srcHi, colHi, offset + HA, end, sx, sxx);
        }

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            assert(src || mask);

            *n = 0;
            *s = 0;
            *sx = 0;
            *sy = 0;
            *sxx = 0;
            *sxy = 0;
            *syy = 0;

            const size_t A = svcntb();
            const size_t B = 181;
            const svuint8_t _index = svdup_n_u8(index);
            const svuint8_t one = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    size_t colE = Simd::Min(colB + B, width);
                    svuint32_t _n = svdup_n_u32(0);
                    svuint32_t _s = svdup_n_u32(0);
                    svuint32_t _sx = svdup_n_u32(0);
                    svuint32_t _sxx = svdup_n_u32(0);

                    if (mask == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            svbool_t pg = svwhilelt_b8(col, colE);
                            svuint8_t _src = svld1_u8(pg, src + col);
                            svuint8_t _count = svand_u8_z(pg, one, one);
                            GetObjectMoments8(_src, _count, col - colB, colE - colB, one, _n, _s, _sx, _sxx);
                        }
                    }
                    else if (src == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            svbool_t pg = svwhilelt_b8(col, colE);
                            svbool_t equal = svcmpeq_u8(pg, svld1_u8(pg, mask + col), _index);
                            svuint8_t _src = svand_u8_z(equal, one, one);
                            GetObjectMoments8(_src, _src, col - colB, colE - colB, one, _n, _s, _sx, _sxx);
                        }
                    }
                    else
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            svbool_t pg = svwhilelt_b8(col, colE);
                            svbool_t equal = svcmpeq_u8(pg, svld1_u8(pg, mask + col), _index);
                            svuint8_t _src = svsel_u8(equal, svld1_u8(pg, src + col), zero);
                            svuint8_t _count = svand_u8_z(equal, one, one);
                            GetObjectMoments8(_src, _count, col - colB, colE - colB, one, _n, _s, _sx, _sxx);
                        }
                    }

                    uint64_t _sn = svaddv_u32(svptrue_b32(), _n);
                    uint64_t _ss = svaddv_u32(svptrue_b32(), _s);
                    uint64_t _ssx = svaddv_u32(svptrue_b32(), _sx);
                    uint64_t _ssxx = svaddv_u32(svptrue_b32(), _sxx);
                    uint64_t _x = colB;
                    uint64_t _y = row;

                    *n += _sn;
                    *s += _ss;

                    *sx += _ssx + _ss * _x;
                    *sy += _ss * _y;

                    *sxx += _ssxx + _ssx * _x * 2 + _ss * _x * _x;
                    *sxy += _ssx * _y + _ss * _x * _y;
                    *syy += _ss * _y * _y;

                    colB = colE;
                }
                if (src)
                    src += srcStride;
                if (mask)
                    mask += maskStride;
            }
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
