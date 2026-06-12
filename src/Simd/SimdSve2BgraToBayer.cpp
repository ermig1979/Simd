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
        enum BgraToBayerIndexType
        {
            BgraToBayerGr,
            BgraToBayerBg,
            BgraToBayerGb,
            BgraToBayerRg,
        };

        SIMD_INLINE bool InitBgraToBayerIndex(uint8_t index[4][2][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                size_t offset = 4 * i, half = 2 * A;
                size_t bgra[4];
                bgra[BgraToBayerGr] = offset + (i & 1 ? 2 : 1);
                bgra[BgraToBayerBg] = offset + (i & 1 ? 1 : 0);
                bgra[BgraToBayerGb] = offset + (i & 1 ? 0 : 1);
                bgra[BgraToBayerRg] = offset + (i & 1 ? 1 : 2);
                for (size_t k = 0; k < 4; ++k)
                {
                    index[k][0][i] = bgra[k] < half ? (uint8_t)bgra[k] : 0xFF;
                    index[k][1][i] = bgra[k] >= half ? (uint8_t)(bgra[k] - half) : 0xFF;
                }
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BGRA_TO_BAYER_INDEX[4][2][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BGRA_TO_BAYER_INDEX_INITED = InitBgraToBayerIndex(BGRA_TO_BAYER_INDEX);

        SIMD_INLINE void BgraToBayer(const uint8_t* bgra, uint8_t* bayer, size_t A,
            const svuint8_t& index0, const svuint8_t& index1, const svbool_t& mask)
        {
            svuint8_t bgra0 = svld1_u8(mask, bgra + 0 * A);
            svuint8_t bgra1 = svld1_u8(mask, bgra + 1 * A);
            svuint8_t bgra2 = svld1_u8(mask, bgra + 2 * A);
            svuint8_t bgra3 = svld1_u8(mask, bgra + 3 * A);
            svuint8_t bayer0 = svtbl2_u8(svcreate2_u8(bgra0, bgra1), index0);
            svuint8_t bayer1 = svtbl2_u8(svcreate2_u8(bgra2, bgra3), index1);
            svst1_u8(mask, bayer, svorr_u8_x(mask, bayer0, bayer1));
        }

        template <int c0, int c1> SIMD_INLINE void BgraToBayerTail(const uint8_t* bgra, uint8_t* bayer, const svbool_t& mask, const svbool_t& even)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svst1_u8(mask, bayer, svsel_u8(even, svget4(_bgra, c0), svget4(_bgra, c1)));
        }

        template <BgraToBayerIndexType i0, BgraToBayerIndexType i1, int c00, int c01, int c10, int c11> void BgraToBayer(const uint8_t* bgra, size_t width, size_t height,
            size_t bgraStride, uint8_t* bayer, size_t bayerStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            size_t A = svlen(svuint8_t()), A4 = A * 4;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svbool_t even = svcmpeq_n_u8(body, svand_n_u8_x(body, svindex_u8(0, 1), 1), 0);
            const svuint8_t index00 = svld1_u8(body, BGRA_TO_BAYER_INDEX[i0][0]);
            const svuint8_t index01 = svld1_u8(body, BGRA_TO_BAYER_INDEX[i0][1]);
            const svuint8_t index10 = svld1_u8(body, BGRA_TO_BAYER_INDEX[i1][0]);
            const svuint8_t index11 = svld1_u8(body, BGRA_TO_BAYER_INDEX[i1][1]);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    BgraToBayer(bgra + offset, bayer + col, A, index00, index01, body);
                if (widthA < width)
                    BgraToBayerTail<c00, c01>(bgra + offset, bayer + col, tail, even);
                bgra += bgraStride;
                bayer += bayerStride;

                col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A4)
                    BgraToBayer(bgra + offset, bayer + col, A, index10, index11, body);
                if (widthA < width)
                    BgraToBayerTail<c10, c11>(bgra + offset, bayer + col, tail, even);
                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        void BgraToBayer(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bayer,
            size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<BgraToBayerGr, BgraToBayerBg, 1, 2, 0, 1>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<BgraToBayerGb, BgraToBayerRg, 1, 0, 2, 1>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<BgraToBayerRg, BgraToBayerGb, 2, 1, 1, 0>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<BgraToBayerBg, BgraToBayerGr, 0, 1, 1, 2>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
