/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
#ifdef SIMD_MADDUBS_ERROR
        SIMD_INLINE __m256i Average8(const __m256i & s00, const __m256i & s01, const __m256i & s10, const __m256i & s11)
        {
            __m256i lo = Average16(
                _mm256_and_si256(s00, K16_00FF),
                _mm256_and_si256(_mm256_srli_si256(s00, 1), K16_00FF),
                _mm256_and_si256(s10, K16_00FF),
                _mm256_and_si256(_mm256_srli_si256(s10, 1), K16_00FF));
            __m256i hi = Average16(
                _mm256_and_si256(s01, K16_00FF),
                _mm256_and_si256(_mm256_srli_si256(s01, 1), K16_00FF),
                _mm256_and_si256(s11, K16_00FF),
                _mm256_and_si256(_mm256_srli_si256(s11, 1), K16_00FF));
            return PackI16ToU8(lo, hi);
        }
#else
        SIMD_INLINE __m256i Average16(const __m256i & s0, const __m256i & s1)
        {
            return _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(s0, K8_01), _mm256_maddubs_epi16(s1, K8_01)), K16_0002), 2);
        }

        SIMD_INLINE __m256i Average8(const __m256i & s00, const __m256i & s01, const __m256i & s10, const __m256i & s11)
        {
            return PackI16ToU8(Average16(s00, s10), Average16(s01, s11));
        }
#endif

        template <bool align> void ReduceGray2x2(
            const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride) && Aligned(dstWidth));
            }

            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenWidth = AlignLo(srcWidth, 2);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedWidth; srcOffset += DA, dstOffset += A)
                {
                    Store<align>((__m256i*)(dst + dstOffset), Average8(
                        Load<align>((__m256i*)(src0 + srcOffset)), Load<align>((__m256i*)(src0 + srcOffset + A)),
                        Load<align>((__m256i*)(src1 + srcOffset)), Load<align>((__m256i*)(src1 + srcOffset + A))));
                }
                if (alignedWidth != srcWidth)
                {
                    dstOffset = dstWidth - A - (evenWidth != srcWidth ? 1 : 0);
                    srcOffset = evenWidth - DA;
                    Store<align>((__m256i*)(dst + dstOffset), Average8(
                        Load<align>((__m256i*)(src0 + srcOffset)), Load<align>((__m256i*)(src0 + srcOffset + A)),
                        Load<align>((__m256i*)(src1 + srcOffset)), Load<align>((__m256i*)(src1 + srcOffset + A))));
                    if (evenWidth != srcWidth)
                    {
                        dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                    }
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
#endif// SIMD_AVX2_ENABLE
}
