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
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> void LbpEstimate(const uint8_t * src, ptrdiff_t stride, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i threshold = Load<false, mask>(src, tail);
            __m512i lbp = _mm512_setzero_si512();
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<align, mask>(src - 1 - stride, tail)), threshold), (char)0x01));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<false, mask>(src - stride, tail)), threshold), (char)0x02));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<false, mask>(src + 1 - stride, tail)), threshold), (char)0x04));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<false, mask>(src + 1, tail)), threshold), (char)0x08));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<false, mask>(src + 1 + stride, tail)), threshold), (char)0x10));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<false, mask>(src + stride, tail)), threshold), (char)0x20));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<align, mask>(src - 1 + stride, tail)), threshold), (char)0x40));
            lbp = _mm512_or_si512(lbp, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask((Load<align, mask>(src - 1, tail)), threshold), (char)0x80));
            Store<false, mask>(dst, lbp, tail);
        }

        template <bool align> void LbpEstimate(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width >= 2);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width - 2, A) + 1;
            __mmask64 tailMask = Aligned(width - alignedWidth);

            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                dst[0] = 0;
                size_t col = 1;
                for (; col < alignedWidth; col += A)
                    LbpEstimate<align, false>(src + col, srcStride, dst + col);
                if (col < width)
                    LbpEstimate<align, false>(src + col, srcStride, dst + col, tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
