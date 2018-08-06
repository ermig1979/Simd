/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
        SIMD_INLINE __m512i Reduce16(const __m512i & s0, const __m512i & s1)
        {
            return _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(s0, K8_01), _mm512_maddubs_epi16(s1, K8_01)), K16_0002), 2);
        }

        SIMD_INLINE __m512i Reduce8(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11)
        {
            return _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(Reduce16(s00, s10), Reduce16(s01, s11)));
        }

        template <bool align, bool mask> SIMD_INLINE void ReduceGray2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst, const __mmask64 * tails)
        {
            const __m512i s00 = Load<align, mask>(src0 + 0, tails[0]);
            const __m512i s01 = Load<align, mask>(src0 + A, tails[1]);
            const __m512i s10 = Load<align, mask>(src1 + 0, tails[0]);
            const __m512i s11 = Load<align, mask>(src1 + A, tails[1]);
            Store<align, mask>(dst, Reduce8(s00, s01, s10, s11), tails[2]);
        }

        template <bool align> void ReduceGray2x2(
            const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride) && Aligned(dstWidth));
            }

            size_t alignedWidth = AlignLo(srcWidth, DA);
            __mmask64 tailMasks[3];
            for (size_t c = 0; c < 2; ++c)
                tailMasks[c] = TailMask64(srcWidth - alignedWidth - A*c);
            tailMasks[2] = TailMask64((srcWidth - alignedWidth) / 2);
            size_t evenWidth = AlignLo(srcWidth, 2);

            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedWidth; srcOffset += DA, dstOffset += A)
                    ReduceGray2x2<align, false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                if (srcOffset < srcWidth)
                {
                    ReduceGray2x2<align, true>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                    if (evenWidth != srcWidth)
                        dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        void ReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcWidth) && Aligned(srcStride) && Aligned(dst) && Aligned(dstWidth) && Aligned(dstStride))
                ReduceGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
