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
        SIMD_INLINE bool InitDeinterleaveBgrIndex(uint8_t index[3][2][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t channel = 0; channel < 3; ++channel)
            {
                for (size_t i = 0; i < A; ++i)
                {
                    size_t src = 3 * i + channel;
                    index[channel][0][i] = src < A ? (uint8_t)src : 0xFF;
                    index[channel][1][i] = src >= A ? (uint8_t)(src - A) : 0xFF;
                }
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t DEINTERLEAVE_BGR_INDEX[3][2][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool DEINTERLEAVE_BGR_INDEX_INITED = InitDeinterleaveBgrIndex(DEINTERLEAVE_BGR_INDEX);

        template<int B, int G, int R> SIMD_INLINE void DeinterleaveBgr(const uint8_t* bgr, size_t A,
            const svbool_t& load0, const svbool_t& load1, const svbool_t& load2, const svbool_t& store,
            const svuint8_t& indexB0, const svuint8_t& indexB1, const svuint8_t& indexG0,
            const svuint8_t& indexG1, const svuint8_t& indexR0, const svuint8_t& indexR1,
            uint8_t* b, uint8_t* g, uint8_t* r)
        {
            svuint8_t bgr0 = svld1_u8(load0, bgr + 0 * A);
            svuint8_t bgr1 = svld1_u8(load1, bgr + 1 * A);
            svuint8_t bgr2 = svld1_u8(load2, bgr + 2 * A);
            svuint8x2_t bgr12 = svcreate2_u8(bgr1, bgr2);

            if (B)
                svst1_u8(store, b, svorr_u8_x(store, svtbl_u8(bgr0, indexB0), svtbl2_u8(bgr12, indexB1)));
            if (G)
                svst1_u8(store, g, svorr_u8_x(store, svtbl_u8(bgr0, indexG0), svtbl2_u8(bgr12, indexG1)));
            if (R)
                svst1_u8(store, r, svorr_u8_x(store, svtbl_u8(bgr0, indexR0), svtbl2_u8(bgr12, indexR1)));
        }

        template<int B, int G, int R> void DeinterleaveBgr(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            size_t tailSize = (width - widthA) * 3;
            const svbool_t tail0 = svwhilelt_b8(size_t(0) * A, tailSize);
            const svbool_t tail1 = svwhilelt_b8(size_t(1) * A, tailSize);
            const svbool_t tail2 = svwhilelt_b8(size_t(2) * A, tailSize);
            const svuint8_t indexB0 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[0][0]);
            const svuint8_t indexB1 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[0][1]);
            const svuint8_t indexG0 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[1][0]);
            const svuint8_t indexG1 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[1][1]);
            const svuint8_t indexR0 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[2][0]);
            const svuint8_t indexR1 = svld1_u8(body, DEINTERLEAVE_BGR_INDEX[2][1]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    DeinterleaveBgr<B, G, R>(bgr + offset, A, body, body, body, body, indexB0, indexB1, indexG0, indexG1, indexR0, indexR1,
                        B ? b + col : NULL, G ? g + col : NULL, R ? r + col : NULL);
                if (widthA < width)
                    DeinterleaveBgr<B, G, R>(bgr + offset, A, tail0, tail1, tail2, tail, indexB0, indexB1, indexG0, indexG1, indexR0, indexR1,
                        B ? b + col : NULL, G ? g + col : NULL, R ? r + col : NULL);
                bgr += bgrStride;
                if (B) b += bStride;
                if (G) g += gStride;
                if (R) r += rStride;
            }
        }

        void DeinterleaveBgr(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride)
        {
            if (b && g && r)
                DeinterleaveBgr<1, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && g)
                DeinterleaveBgr<1, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b && r)
                DeinterleaveBgr<1, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g && r)
                DeinterleaveBgr<0, 1, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (b)
                DeinterleaveBgr<1, 0, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (g)
                DeinterleaveBgr<0, 1, 0>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else if (r)
                DeinterleaveBgr<0, 0, 1>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
        }
    }
#endif
}
