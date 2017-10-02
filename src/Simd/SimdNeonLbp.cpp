/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void LbpEstimate(const uint8_t * src, ptrdiff_t stride, uint8_t * dst)
        {
            uint8x16_t threshold = Load<false>(src);
            uint8x16_t lbp = K8_00;
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<align>(src - 1 - stride), threshold), K8_01));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<false>(src - stride), threshold), K8_02));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<false>(src + 1 - stride), threshold), K8_04));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<false>(src + 1), threshold), K8_08));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<false>(src + 1 + stride), threshold), K8_10));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<false>(src + stride), threshold), K8_20));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<align>(src - 1 + stride), threshold), K8_40));
            lbp = vorrq_u8(lbp, vandq_u8(vcgeq_u8(Load<align>(src - 1), threshold), K8_80));
            Store<false>(dst, lbp);
        }

        template <bool align> void LbpEstimate(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A + 2);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width - 2, A) + 1;

            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                dst[0] = 0;
                for (size_t col = 1; col < alignedWidth; col += A)
                    LbpEstimate<align>(src + col, srcStride, dst + col);
                if (alignedWidth != width - 1)
                    LbpEstimate<false>(src + width - 1 - A, srcStride, dst + width - 1 - A);
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }

        void LbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                LbpEstimate<true>(src, srcStride, width, height, dst, dstStride);
            else
                LbpEstimate<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
