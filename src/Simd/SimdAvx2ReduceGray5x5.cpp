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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool compensation> SIMD_INLINE __m256i DivideBy256(__m256i value);

        template <> SIMD_INLINE __m256i DivideBy256<true>(__m256i value)
        {
            return _mm256_srli_epi16(_mm256_add_epi16(value, K16_0080), 8);
        }

        template <> SIMD_INLINE __m256i DivideBy256<false>(__m256i value)
        {
            return _mm256_srli_epi16(value, 8);
        }

        SIMD_INLINE __m256i LoadUnpacked(const void * src)
        {
            return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)src));
        }

        SIMD_INLINE void FirstRow5x5(__m256i src, uint16_t * in0, uint16_t * in1, size_t offset)
        {
            _mm256_storeu_si256((__m256i*)(in0 + offset), src);
            _mm256_storeu_si256((__m256i*)(in1 + offset), _mm256_mullo_epi16(src, K16_0005));
        }

        SIMD_INLINE void FirstRow5x5(const uint8_t * src, uint16_t * in0, uint16_t * in1, size_t offset)
        {
            FirstRow5x5(LoadUnpacked(src + offset), in0, in1, offset);
            offset += HA;
            FirstRow5x5(LoadUnpacked(src + offset), in0, in1, offset);
        }

        SIMD_INLINE void MainRowY5x5(__m256i odd, __m256i even, uint16_t * in0, uint16_t * in1, uint16_t * out0, uint16_t * out1, uint16_t * dst, size_t offset)
        {
            __m256i cp = _mm256_mullo_epi16(odd, K16_0004);
            __m256i c0 = _mm256_loadu_si256((__m256i*)(in0 + offset));
            __m256i c1 = _mm256_loadu_si256((__m256i*)(in1 + offset));
            _mm256_storeu_si256((__m256i*)(dst + offset), _mm256_add_epi16(even, _mm256_add_epi16(c1, _mm256_add_epi16(cp, _mm256_mullo_epi16(c0, K16_0006)))));
            _mm256_storeu_si256((__m256i*)(out1 + offset), _mm256_add_epi16(c0, cp));
            _mm256_storeu_si256((__m256i*)(out0 + offset), even);
        }

        SIMD_INLINE void MainRowY5x5(const uint8_t * odd, const uint8_t * even, uint16_t * in0, uint16_t * in1, uint16_t * out0, uint16_t * out1, uint16_t * dst, size_t offset)
        {
            MainRowY5x5(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), in0, in1, out0, out1, dst, offset);
            offset += HA;
            MainRowY5x5(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), in0, in1, out0, out1, dst, offset);
        }

        template <bool compensation> SIMD_INLINE __m256i MainRowX5x5(uint16_t * dst)
        {
            __m256i t0 = _mm256_loadu_si256((__m256i*)(dst - 2));
            __m256i t1 = _mm256_loadu_si256((__m256i*)(dst - 1));
            __m256i t2 = _mm256_loadu_si256((__m256i*)dst);
            __m256i t3 = _mm256_loadu_si256((__m256i*)(dst + 1));
            __m256i t4 = _mm256_loadu_si256((__m256i*)(dst + 2));
            t2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_mullo_epi16(t2, K16_0006), _mm256_mullo_epi16(_mm256_add_epi16(t1, t3), K16_0004)), _mm256_add_epi16(t0, t4));
            return DivideBy256<compensation>(t2);
        }

        template <bool compensation> SIMD_INLINE __m256i MainRowX5x5(uint16_t * src, size_t offset)
        {
            const __m256i lo = MainRowX5x5<compensation>(src + offset);
            const __m256i hi = MainRowX5x5<compensation>(src + offset + HA);
            return _mm256_and_si256(PackI16ToU8(lo, hi), K16_00FF);
        }

        template <bool compensation> SIMD_INLINE void MainRowX5x5(uint16_t * src, size_t offset, uint8_t * dst)
        {
            __m256i lo = MainRowX5x5<compensation>(src, offset);
            __m256i hi = MainRowX5x5<compensation>(src, offset + A);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(lo, hi));
        }

        template <bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);

            size_t alignedWidth = Simd::AlignLo(srcWidth, DA);
            size_t bufferDstTail = Simd::AlignHi(srcWidth - DA, 2);

            const size_t bufferWidth = Simd::AlignHi(srcWidth, A);
            Array16u buffer(5 * bufferWidth + A);
            uint16_t * in0 = buffer.data;
            uint16_t * in1 = in0 + bufferWidth;
            uint16_t * out0 = in1 + bufferWidth;
            uint16_t * out1 = out0 + bufferWidth;
            uint16_t * sum = out1 + bufferWidth + HA;

            for (size_t col = 0; col < alignedWidth; col += A)
                FirstRow5x5(src, in0, in1, col);
            if (alignedWidth != srcWidth)
            {
                FirstRow5x5(src, in0, in1, srcWidth - DA);
                FirstRow5x5(src, in0, in1, srcWidth - A);
            }
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t * odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t * even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t col = 0; col < alignedWidth; col += A)
                    MainRowY5x5(odd, even, in0, in1, out0, out1, sum, col);
                if (alignedWidth != srcWidth)
                {
                    MainRowY5x5(odd, even, in0, in1, out0, out1, sum, srcWidth - DA);
                    MainRowY5x5(odd, even, in0, in1, out0, out1, sum, srcWidth - A);
                }

                Swap(in0, out0);
                Swap(in1, out1);

                sum[-2] = sum[0];
                sum[-1] = sum[0];
                sum[srcWidth] = sum[srcWidth - 1];
                sum[srcWidth + 1] = sum[srcWidth - 1];

                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += DA, dstCol += A)
                    MainRowX5x5<compensation>(sum, srcCol, dst + dstCol);
                if (alignedWidth != srcWidth)
                    MainRowX5x5<compensation>(sum, bufferDstTail, dst + dstWidth - A);
            }
        }

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
