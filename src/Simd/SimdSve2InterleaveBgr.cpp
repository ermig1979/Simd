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
        SIMD_INLINE bool InitInterleaveBgrIndex(uint8_t index[3][2][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t part = 0; part < 3; ++part)
            {
                for (size_t i = 0; i < A; ++i)
                {
                    size_t dst = part * A + i;
                    size_t channel = dst % 3;
                    size_t pixel = dst / 3;
                    if (channel == 0)
                        index[part][0][i] = (uint8_t)pixel;
                    else if (channel == 1)
                        index[part][0][i] = (uint8_t)(A + pixel);
                    else
                        index[part][0][i] = 0xFF;
                    if (channel == 2)
                        index[part][1][i] = (uint8_t)(A + pixel);
                    else
                        index[part][1][i] = (uint8_t)i;
                }
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t INTERLEAVE_BGR_INDEX[3][2][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool INTERLEAVE_BGR_INDEX_INITED = InitInterleaveBgrIndex(INTERLEAVE_BGR_INDEX);

        SIMD_INLINE svuint8_t InterleaveBgr(svuint8_t b, svuint8_t g, svuint8_t r,
            svuint8_t indexBG, svuint8_t indexBGR)
        {
            svuint8_t bgPlaced = svtbl2_u8(svcreate2_u8(b, g), indexBG);
            return svtbl2_u8(svcreate2_u8(bgPlaced, r), indexBGR);
        }

        SIMD_INLINE void InterleaveBgr(const uint8_t* b, const uint8_t* g, const uint8_t* r, uint8_t* bgr, size_t A,
            svuint8_t indexBG0, svuint8_t indexBGR0,
            svuint8_t indexBG1, svuint8_t indexBGR1,
            svuint8_t indexBG2, svuint8_t indexBGR2,
            svbool_t load, svbool_t store0, svbool_t store1, svbool_t store2)
        {
            svuint8_t _b = svld1_u8(load, b);
            svuint8_t _g = svld1_u8(load, g);
            svuint8_t _r = svld1_u8(load, r);
            svst1_u8(store0, bgr + 0 * A, InterleaveBgr(_b, _g, _r, indexBG0, indexBGR0));
            svst1_u8(store1, bgr + 1 * A, InterleaveBgr(_b, _g, _r, indexBG1, indexBGR1));
            svst1_u8(store2, bgr + 2 * A, InterleaveBgr(_b, _g, _r, indexBG2, indexBGR2));
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
            const svuint8_t indexBG0 = svld1_u8(body, INTERLEAVE_BGR_INDEX[0][0]);
            const svuint8_t indexBGR0 = svld1_u8(body, INTERLEAVE_BGR_INDEX[0][1]);
            const svuint8_t indexBG1 = svld1_u8(body, INTERLEAVE_BGR_INDEX[1][0]);
            const svuint8_t indexBGR1 = svld1_u8(body, INTERLEAVE_BGR_INDEX[1][1]);
            const svuint8_t indexBG2 = svld1_u8(body, INTERLEAVE_BGR_INDEX[2][0]);
            const svuint8_t indexBGR2 = svld1_u8(body, INTERLEAVE_BGR_INDEX[2][1]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    InterleaveBgr(b + col, g + col, r + col, bgr + offset, A,
                        indexBG0, indexBGR0, indexBG1, indexBGR1, indexBG2, indexBGR2, body, body, body, body);
                if (widthA < width)
                    InterleaveBgr(b + col, g + col, r + col, bgr + offset, A,
                        indexBG0, indexBGR0, indexBG1, indexBGR1, indexBG2, indexBGR2, tail, tail0, tail1, tail2);
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }
    }
#endif
}
