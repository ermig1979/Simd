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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool align, bool mask> SIMD_INLINE void StretchGray2x2(const uint8_t * src, uint8_t * dst0, uint8_t * dst1, const __mmask64 * tails)
        {
            __m512i _src = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(src, tails[2])));
            __m512i lo = _mm512_unpacklo_epi8(_src, _src);
            __m512i hi = _mm512_unpackhi_epi8(_src, _src);
            Store<align, mask>(dst0 + 0, lo, tails[0]);
            Store<align, mask>(dst0 + A, hi, tails[1]);
            Store<align, mask>(dst1 + 0, lo, tails[0]);
            Store<align, mask>(dst1 + A, hi, tails[1]);
        }

        template <bool align> void StretchGray2x2(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth * 2 == dstWidth && srcHeight * 2 == dstHeight);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(srcWidth, A);
            __mmask64 tailMasks[3];
            for (size_t c = 0; c < 2; ++c)
                tailMasks[c] = TailMask64((srcWidth - alignedWidth) * 2 - A*c);
            tailMasks[2] = TailMask64(srcWidth - alignedWidth);

            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                uint8_t * dst1 = dst + dstStride;
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedWidth; srcOffset += A, dstOffset += DA)
                    StretchGray2x2<align, false>(src + srcOffset, dst + dstOffset, dst1 + dstOffset, tailMasks);
                if (srcOffset < srcWidth)
                    StretchGray2x2<align, true>(src + srcOffset, dst + dstOffset, dst1 + dstOffset, tailMasks);
                src += srcStride;
                dst += 2 * dstStride;
            }
        }

        void StretchGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                StretchGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                StretchGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
