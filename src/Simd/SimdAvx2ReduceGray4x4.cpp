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
        SIMD_INLINE __m256i DivideBy64(__m256i value)
        {
            return _mm256_srli_epi16(_mm256_add_epi16(value, K16_0020), 6);
        }

        SIMD_INLINE __m256i BinomialSum16(const __m256i & a, const __m256i & b, const __m256i & c, const __m256i & d)
        {
            return _mm256_add_epi16(_mm256_add_epi16(a, d), _mm256_mullo_epi16(_mm256_add_epi16(b, c), K16_0003));
        }

        const __m256i K8_01_03 = SIMD_MM256_SET2_EPI8(1, 3);
        const __m256i K8_03_01 = SIMD_MM256_SET2_EPI8(3, 1);

        SIMD_INLINE __m256i BinomialSum16(const __m256i & ab, const __m256i & cd)
        {
#ifdef SIMD_MADDUBS_ERROR
            return _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_or_si256(K_ZERO, ab), K8_01_03), _mm256_maddubs_epi16(_mm256_or_si256(K_ZERO, cd), K8_03_01));
#else
            return _mm256_add_epi16(_mm256_maddubs_epi16(ab, K8_01_03), _mm256_maddubs_epi16(cd, K8_03_01));
#endif        
        }

        SIMD_INLINE __m256i ReduceColNose(const uint8_t * src)
        {
            const __m256i t2 = _mm256_loadu_si256((__m256i*)(src + 1));
            return BinomialSum16(LoadBeforeFirst<false, 1>(src), t2);
        }

        SIMD_INLINE __m256i ReduceColBody(const uint8_t * src)
        {
            const __m256i t0 = _mm256_loadu_si256((__m256i*)(src - 1));
            const __m256i t2 = _mm256_loadu_si256((__m256i*)(src + 1));
            return BinomialSum16(t0, t2);
        }

        template <bool even> SIMD_INLINE __m256i ReduceColTail(const uint8_t * src);

        template <> SIMD_INLINE __m256i ReduceColTail<true>(const uint8_t * src)
        {
            const __m256i t0 = _mm256_loadu_si256((__m256i*)(src - 1));
            const __m256i t2 = LoadAfterLast<false, 1>(src);
            return BinomialSum16(t0, t2);
        }

        template <> SIMD_INLINE __m256i ReduceColTail<false>(const uint8_t * src)
        {
            const __m256i t0 = _mm256_loadu_si256((__m256i*)(src - 1));
            __m256i t1, t2;
            LoadAfterLast<false, 1>(src - 1, t1, t2);
            return BinomialSum16(t0, t2);
        }

        SIMD_INLINE __m256i ReduceRow16(const uint16_t * src0, const uint16_t * src1, const uint16_t * src2, const uint16_t * src3, size_t offset)
        {
            return _mm256_and_si256(DivideBy64(BinomialSum16(
                _mm256_loadu_si256((__m256i*)(src0 + offset)), _mm256_loadu_si256((__m256i*)(src1 + offset)),
                _mm256_loadu_si256((__m256i*)(src2 + offset)), _mm256_loadu_si256((__m256i*)(src3 + offset)))), K16_00FF);
        }

        SIMD_INLINE __m256i ReduceRow8(const uint16_t * src0, const uint16_t * src1, const uint16_t * src2, const uint16_t * src3, size_t offset)
        {
            __m256i lo = ReduceRow16(src0, src1, src2, src3, offset);
            __m256i hi = ReduceRow16(src0, src1, src2, src3, offset + HA);
            return PackI16ToU8(lo, hi);
        }

        template <bool even> void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > DA);

            size_t alignedDstWidth = Simd::AlignLo(dstWidth, A);
            size_t srcTail = Simd::AlignHi(srcWidth - A, 2);

            const size_t bufferWidth = Simd::AlignHi(dstWidth, A);
            Array16u buffer(4 * bufferWidth);
            uint16_t * src0 = buffer.data;
            uint16_t * src1 = src0 + bufferWidth;
            uint16_t * src2 = src1 + bufferWidth;
            uint16_t * src3 = src2 + bufferWidth;

            __m256i tmp = ReduceColNose(src);
            _mm256_storeu_si256((__m256i*)src0, tmp);
            _mm256_storeu_si256((__m256i*)src1, tmp);
            size_t srcCol = A, dstCol = HA;
            for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
            {
                tmp = ReduceColBody(src + srcCol);
                _mm256_storeu_si256((__m256i*)(src0 + dstCol), tmp);
                _mm256_storeu_si256((__m256i*)(src1 + dstCol), tmp);
            }
            tmp = ReduceColTail<even>(src + srcTail);
            _mm256_storeu_si256((__m256i*)(src0 + dstWidth - HA), tmp);
            _mm256_storeu_si256((__m256i*)(src1 + dstWidth - HA), tmp);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t * s2 = src + srcStride*(row + 1);
                const uint8_t * s3 = s2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    s2 = src + srcStride*(srcHeight - 1);
                    s3 = s2;
                }

                _mm256_storeu_si256((__m256i*)src2, ReduceColNose(s2));
                _mm256_storeu_si256((__m256i*)src3, ReduceColNose(s3));
                size_t srcCol = A, dstCol = HA;
                for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                {
                    _mm256_storeu_si256((__m256i*)(src2 + dstCol), ReduceColBody(s2 + srcCol));
                    _mm256_storeu_si256((__m256i*)(src3 + dstCol), ReduceColBody(s3 + srcCol));
                }
                _mm256_storeu_si256((__m256i*)(src2 + dstWidth - HA), ReduceColTail<even>(s2 + srcTail));
                _mm256_storeu_si256((__m256i*)(src3 + dstWidth - HA), ReduceColTail<even>(s3 + srcTail));

                _mm256_storeu_si256((__m256i*)dst, ReduceRow8(src0, src1, src2, src3, 0));
                for (size_t col = A; col < alignedDstWidth; col += A)
                    _mm256_storeu_si256((__m256i*)(dst + col), ReduceRow8(src0, src1, src2, src3, col));

                if (alignedDstWidth != dstWidth)
                    _mm256_storeu_si256((__m256i*)(dst + dstWidth - A), ReduceRow8(src0, src1, src2, src3, dstWidth - A));

                Swap(src0, src2);
                Swap(src1, src3);
            }
        }

        void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(srcWidth, 2))
                ReduceGray4x4<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray4x4<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
