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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint8_t LbpBit(const uint8_t* src, const svuint8_t& threshold, uint8_t bit, const svbool_t& mask)
        {
            return svsel_u8(Compare8u<SimdCompareGreaterOrEqual>(mask, svld1_u8(mask, src), threshold), svdup_n_u8(bit), svdup_n_u8(0));
        }

        SIMD_INLINE void LbpEstimate(const uint8_t* src, size_t stride, uint8_t* dst, const svbool_t& mask)
        {
            const svuint8_t threshold = svld1_u8(mask, src);
            svuint8_t lbp = LbpBit(src - stride - 1, threshold, 0x01, mask);
            lbp = svorr_u8_x(mask, lbp, LbpBit(src - stride, threshold, 0x02, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src - stride + 1, threshold, 0x04, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src + 1, threshold, 0x08, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src + stride + 1, threshold, 0x10, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src + stride, threshold, 0x20, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src + stride - 1, threshold, 0x40, mask));
            lbp = svorr_u8_x(mask, lbp, LbpBit(src - 1, threshold, 0x80, mask));
            svst1_u8(mask, dst, lbp);
        }

        void LbpEstimate(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(width >= 3);

            const size_t A = svcntb();
            const size_t alignedWidth = AlignLo(width - 2, A) + 1;
            const svbool_t body = svptrue_b8();

            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                dst[0] = 0;
                for (size_t col = 1; col < alignedWidth; col += A)
                    LbpEstimate(src + col, srcStride, dst + col, body);
                if (alignedWidth != width - 1)
                    LbpEstimate(src + alignedWidth, srcStride, dst + alignedWidth, svwhilelt_b8(alignedWidth, width - 1));
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }
    }
#endif
}
