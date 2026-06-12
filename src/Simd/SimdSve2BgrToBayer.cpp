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
        enum BgrToBayerIndexType
        {
            BgrToBayerGr,
            BgrToBayerBg,
            BgrToBayerGb,
            BgrToBayerRg,
        };

        SIMD_INLINE bool InitBgrToBayerIndex(uint8_t index[4][2][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                size_t offset = 3 * i, half = 2 * A;
                size_t bgr[4];
                bgr[BgrToBayerGr] = offset + (i & 1 ? 2 : 1);
                bgr[BgrToBayerBg] = offset + (i & 1 ? 1 : 0);
                bgr[BgrToBayerGb] = offset + (i & 1 ? 0 : 1);
                bgr[BgrToBayerRg] = offset + (i & 1 ? 1 : 2);
                for (size_t k = 0; k < 4; ++k)
                {
                    index[k][0][i] = bgr[k] < half ? (uint8_t)bgr[k] : 0xFF;
                    index[k][1][i] = bgr[k] >= half ? (uint8_t)(bgr[k] - half) : 0xFF;
                }
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BGR_TO_BAYER_INDEX[4][2][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BGR_TO_BAYER_INDEX_INITED = InitBgrToBayerIndex(BGR_TO_BAYER_INDEX);

        SIMD_INLINE void BgrToBayer(const uint8_t* bgr, uint8_t* bayer, size_t A,
            const svuint8_t& index0, const svuint8_t& index1, const svbool_t& mask)
        {
            svuint8_t bgr0 = svld1_u8(mask, bgr + 0 * A);
            svuint8_t bgr1 = svld1_u8(mask, bgr + 1 * A);
            svuint8_t bgr2 = svld1_u8(mask, bgr + 2 * A);
            svuint8_t bayer0 = svtbl2_u8(svcreate2_u8(bgr0, bgr1), index0);
            svuint8_t bayer1 = svtbl_u8(bgr2, index1);
            svst1_u8(mask, bayer, svorr_u8_x(mask, bayer0, bayer1));
        }

        template <int c0, int c1> SIMD_INLINE void BgrToBayerTail(const uint8_t* bgr, uint8_t* bayer, const svbool_t& mask, const svbool_t& even)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svst1_u8(mask, bayer, svsel_u8(even, svget3(_bgr, c0), svget3(_bgr, c1)));
        }

        template <BgrToBayerIndexType i0, BgrToBayerIndexType i1, int c00, int c01, int c10, int c11> void BgrToBayer(const uint8_t* bgr, size_t width, size_t height,
            size_t bgrStride, uint8_t* bayer, size_t bayerStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            size_t A = svlen(svuint8_t()), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svbool_t even = svcmpeq_n_u8(body, svand_n_u8_x(body, svindex_u8(0, 1), 1), 0);
            const svuint8_t index00 = svld1_u8(body, BGR_TO_BAYER_INDEX[i0][0]);
            const svuint8_t index01 = svld1_u8(body, BGR_TO_BAYER_INDEX[i0][1]);
            const svuint8_t index10 = svld1_u8(body, BGR_TO_BAYER_INDEX[i1][0]);
            const svuint8_t index11 = svld1_u8(body, BGR_TO_BAYER_INDEX[i1][1]);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToBayer(bgr + offset, bayer + col, A, index00, index01, body);
                if (widthA < width)
                    BgrToBayerTail<c00, c01>(bgr + offset, bayer + col, tail, even);
                bgr += bgrStride;
                bayer += bayerStride;

                col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToBayer(bgr + offset, bayer + col, A, index10, index11, body);
                if (widthA < width)
                    BgrToBayerTail<c10, c11>(bgr + offset, bayer + col, tail, even);
                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer,
            size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<BgrToBayerGr, BgrToBayerBg, 1, 2, 0, 1>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<BgrToBayerGb, BgrToBayerRg, 1, 0, 2, 1>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<BgrToBayerRg, BgrToBayerGb, 2, 1, 1, 0>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<BgrToBayerBg, BgrToBayerGr, 0, 1, 1, 2>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
