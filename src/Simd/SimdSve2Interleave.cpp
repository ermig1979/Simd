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
        SIMD_INLINE bool InitInterleaveBgrIndex(uint8_t index[3][3][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t part = 0; part < 3; ++part)
            {
                for (size_t channel = 0; channel < 3; ++channel)
                {
                    for (size_t i = 0; i < A; ++i)
                    {
                        size_t dst = part * A + i;
                        index[part][channel][i] = dst % 3 == channel ? (uint8_t)(dst / 3) : 0xFF;
                    }
                }
            }
            return true;
        }

        SIMD_INLINE bool InitInterleaveBgraIndex(uint8_t index[4][4][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t part = 0; part < 4; ++part)
            {
                for (size_t channel = 0; channel < 4; ++channel)
                {
                    for (size_t i = 0; i < A; ++i)
                    {
                        size_t dst = part * A + i;
                        index[part][channel][i] = dst % 4 == channel ? (uint8_t)(dst / 4) : 0xFF;
                    }
                }
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t INTERLEAVE_BGR_INDEX[3][3][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool INTERLEAVE_BGR_INDEX_INITED = InitInterleaveBgrIndex(INTERLEAVE_BGR_INDEX);

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t INTERLEAVE_BGRA_INDEX[4][4][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool INTERLEAVE_BGRA_INDEX_INITED = InitInterleaveBgraIndex(INTERLEAVE_BGRA_INDEX);

        SIMD_INLINE void InterleaveUv(const uint8_t* u, const uint8_t* v, uint8_t* uv,
            const svbool_t& load, const svbool_t& store0, const svbool_t& store1)
        {
            svuint8_t _u = svld1_u8(load, u);
            svuint8_t _v = svld1_u8(load, v);
            svst1_u8(store0, uv, svzip1_u8(_u, _v));
            svst1_u8(store1, uv + svcntb(), svzip2_u8(_u, _v));
        }

        void InterleaveUv(const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* uv, size_t uvStride)
        {
            size_t A = svcntb(), A2 = A * 2;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 2;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A2)
                    InterleaveUv(u + col, v + col, uv + offset, body, body, body);
                if (widthA < width)
                    InterleaveUv(u + col, v + col, uv + offset, tail, tail0, tail1);
                u += uStride;
                v += vStride;
                uv += uvStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t InterleaveBgr(const svuint8_t& b, const svuint8_t& g, const svuint8_t& r,
            const svuint8_t& indexB, const svuint8_t& indexG, const svuint8_t& indexR, const svbool_t& mask)
        {
            return svorr_u8_x(mask, svorr_u8_x(mask, svtbl_u8(b, indexB), svtbl_u8(g, indexG)), svtbl_u8(r, indexR));
        }

        SIMD_INLINE void InterleaveBgr(const uint8_t* b, const uint8_t* g, const uint8_t* r, uint8_t* bgr, size_t A,
            const svuint8_t& indexB0, const svuint8_t& indexG0, const svuint8_t& indexR0,
            const svuint8_t& indexB1, const svuint8_t& indexG1, const svuint8_t& indexR1,
            const svuint8_t& indexB2, const svuint8_t& indexG2, const svuint8_t& indexR2,
            const svbool_t& load, const svbool_t& store0, const svbool_t& store1, const svbool_t& store2)
        {
            svuint8_t _b = svld1_u8(load, b);
            svuint8_t _g = svld1_u8(load, g);
            svuint8_t _r = svld1_u8(load, r);
            svst1_u8(store0, bgr + 0 * A, InterleaveBgr(_b, _g, _r, indexB0, indexG0, indexR0, store0));
            svst1_u8(store1, bgr + 1 * A, InterleaveBgr(_b, _g, _r, indexB1, indexG1, indexR1, store1));
            svst1_u8(store2, bgr + 2 * A, InterleaveBgr(_b, _g, _r, indexB2, indexG2, indexR2, store2));
        }

        void InterleaveBgr(const uint8_t* b, size_t bStride, const uint8_t* g, size_t gStride, const uint8_t* r, size_t rStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            size_t A = svcntb(), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 3;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            const svbool_t tail2 = svwhilelt_b8(size_t(2) * A, tailSize);
            const svuint8_t indexB0 = svld1_u8(body, INTERLEAVE_BGR_INDEX[0][0]);
            const svuint8_t indexG0 = svld1_u8(body, INTERLEAVE_BGR_INDEX[0][1]);
            const svuint8_t indexR0 = svld1_u8(body, INTERLEAVE_BGR_INDEX[0][2]);
            const svuint8_t indexB1 = svld1_u8(body, INTERLEAVE_BGR_INDEX[1][0]);
            const svuint8_t indexG1 = svld1_u8(body, INTERLEAVE_BGR_INDEX[1][1]);
            const svuint8_t indexR1 = svld1_u8(body, INTERLEAVE_BGR_INDEX[1][2]);
            const svuint8_t indexB2 = svld1_u8(body, INTERLEAVE_BGR_INDEX[2][0]);
            const svuint8_t indexG2 = svld1_u8(body, INTERLEAVE_BGR_INDEX[2][1]);
            const svuint8_t indexR2 = svld1_u8(body, INTERLEAVE_BGR_INDEX[2][2]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    InterleaveBgr(b + col, g + col, r + col, bgr + offset, A,
                        indexB0, indexG0, indexR0, indexB1, indexG1, indexR1, indexB2, indexG2, indexR2, body, body, body, body);
                if (widthA < width)
                    InterleaveBgr(b + col, g + col, r + col, bgr + offset, A,
                        indexB0, indexG0, indexR0, indexB1, indexG1, indexR1, indexB2, indexG2, indexR2, tail, tail0, tail1, tail2);
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t InterleaveBgra(const svuint8_t& b, const svuint8_t& g, const svuint8_t& r, const svuint8_t& a,
            const svuint8_t& indexB, const svuint8_t& indexG, const svuint8_t& indexR, const svuint8_t& indexA, const svbool_t& mask)
        {
            return svorr_u8_x(mask, svorr_u8_x(mask, svtbl_u8(b, indexB), svtbl_u8(g, indexG)),
                svorr_u8_x(mask, svtbl_u8(r, indexR), svtbl_u8(a, indexA)));
        }

        SIMD_INLINE void InterleaveBgra(const uint8_t* b, const uint8_t* g, const uint8_t* r, const uint8_t* a, uint8_t* bgra, size_t A,
            const svuint8_t& indexB0, const svuint8_t& indexG0, const svuint8_t& indexR0, const svuint8_t& indexA0,
            const svuint8_t& indexB1, const svuint8_t& indexG1, const svuint8_t& indexR1, const svuint8_t& indexA1,
            const svuint8_t& indexB2, const svuint8_t& indexG2, const svuint8_t& indexR2, const svuint8_t& indexA2,
            const svuint8_t& indexB3, const svuint8_t& indexG3, const svuint8_t& indexR3, const svuint8_t& indexA3,
            const svbool_t& load, const svbool_t& store0, const svbool_t& store1, const svbool_t& store2, const svbool_t& store3)
        {
            svuint8_t _b = svld1_u8(load, b);
            svuint8_t _g = svld1_u8(load, g);
            svuint8_t _r = svld1_u8(load, r);
            svuint8_t _a = svld1_u8(load, a);
            svst1_u8(store0, bgra + 0 * A, InterleaveBgra(_b, _g, _r, _a, indexB0, indexG0, indexR0, indexA0, store0));
            svst1_u8(store1, bgra + 1 * A, InterleaveBgra(_b, _g, _r, _a, indexB1, indexG1, indexR1, indexA1, store1));
            svst1_u8(store2, bgra + 2 * A, InterleaveBgra(_b, _g, _r, _a, indexB2, indexG2, indexR2, indexA2, store2));
            svst1_u8(store3, bgra + 3 * A, InterleaveBgra(_b, _g, _r, _a, indexB3, indexG3, indexR3, indexA3, store3));
        }

        void InterleaveBgra(const uint8_t* b, size_t bStride, const uint8_t* g, size_t gStride, const uint8_t* r, size_t rStride, const uint8_t* a, size_t aStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            size_t A = svcntb(), A4 = A * 4;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 4;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            const svbool_t tail2 = svwhilelt_b8(size_t(2) * A, tailSize);
            const svbool_t tail3 = svwhilelt_b8(size_t(3) * A, tailSize);
            const svuint8_t indexB0 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[0][0]);
            const svuint8_t indexG0 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[0][1]);
            const svuint8_t indexR0 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[0][2]);
            const svuint8_t indexA0 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[0][3]);
            const svuint8_t indexB1 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[1][0]);
            const svuint8_t indexG1 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[1][1]);
            const svuint8_t indexR1 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[1][2]);
            const svuint8_t indexA1 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[1][3]);
            const svuint8_t indexB2 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[2][0]);
            const svuint8_t indexG2 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[2][1]);
            const svuint8_t indexR2 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[2][2]);
            const svuint8_t indexA2 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[2][3]);
            const svuint8_t indexB3 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[3][0]);
            const svuint8_t indexG3 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[3][1]);
            const svuint8_t indexR3 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[3][2]);
            const svuint8_t indexA3 = svld1_u8(body, INTERLEAVE_BGRA_INDEX[3][3]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    InterleaveBgra(b + col, g + col, r + col, a + col, bgra + offset, A,
                        indexB0, indexG0, indexR0, indexA0, indexB1, indexG1, indexR1, indexA1,
                        indexB2, indexG2, indexR2, indexA2, indexB3, indexG3, indexR3, indexA3,
                        body, body, body, body, body);
                if (widthA < width)
                    InterleaveBgra(b + col, g + col, r + col, a + col, bgra + offset, A,
                        indexB0, indexG0, indexR0, indexA0, indexB1, indexG1, indexR1, indexA1,
                        indexB2, indexG2, indexR2, indexA2, indexB3, indexG3, indexR3, indexA3,
                        tail, tail0, tail1, tail2, tail3);
                b += bStride;
                g += gStride;
                r += rStride;
                a += aStride;
                bgra += bgraStride;
            }
        }
    }
#endif
}
