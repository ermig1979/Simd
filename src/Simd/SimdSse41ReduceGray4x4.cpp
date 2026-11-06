/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i DivideBy64(__m128i value)
        {
            return _mm_srli_epi16(_mm_add_epi16(value, K16_0020), 6);
        }

        const __m128i K8_01_03 = SIMD_MM_SET2_EPI8(1, 3);
        const __m128i K8_03_01 = SIMD_MM_SET2_EPI8(3, 1);

        SIMD_INLINE __m128i BinomialSum16(const __m128i & ab, const __m128i & cd)
        {
            return _mm_add_epi16(_mm_maddubs_epi16(ab, K8_01_03), _mm_maddubs_epi16(cd, K8_03_01));
        }

        SIMD_INLINE __m128i ReduceColNose(const uint8_t *src)
        {
            const __m128i t1 = _mm_loadu_si128((__m128i*)src);
            const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
            return BinomialSum16(LoadBeforeFirst<1>(t1), t2);
        }

        SIMD_INLINE __m128i ReduceColBody(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
            return BinomialSum16(t0, t2);
        }

        template <bool even> SIMD_INLINE __m128i ReduceColTail(const uint8_t *src);

        template <> SIMD_INLINE __m128i ReduceColTail<true>(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            const __m128i t1 = _mm_loadu_si128((__m128i*)src);
            return BinomialSum16(t0, LoadAfterLast<1>(t1));
        }

        template <> SIMD_INLINE __m128i ReduceColTail<false>(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            return BinomialSum16(t0, LoadAfterLast<1>(LoadAfterLast<1>(t0)));
        }

        template <bool align> SIMD_INLINE __m128i ReduceRow(const uint16_t * src0, const uint16_t * src1, const uint16_t * src2, const uint16_t * src3, size_t offset)
        {
            return _mm_packus_epi16(_mm_and_si128(DivideBy64(BinomialSum16(
                Load<align>((__m128i*)(src0 + offset)), Load<align>((__m128i*)(src1 + offset)),
                Load<align>((__m128i*)(src2 + offset)), Load<align>((__m128i*)(src3 + offset)))), K16_00FF), K_ZERO);
        }

        template <bool even> void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > A);

            size_t alignedDstWidth = Simd::AlignLo(dstWidth, HA);
            size_t srcTail = Simd::AlignHi(srcWidth - A, 2);

            const size_t bufferWidth = Simd::AlignHi(dstWidth, A);
            Array16u buffer(4 * bufferWidth);
            uint16_t * src0 = buffer.data;
            uint16_t * src1 = src0 + bufferWidth;
            uint16_t * src2 = src1 + bufferWidth;
            uint16_t * src3 = src2 + bufferWidth;

            __m128i tmp = ReduceColNose(src);
            Store<true>((__m128i*)src0, tmp);
            Store<true>((__m128i*)src1, tmp);
            size_t srcCol = A, dstCol = HA;
            for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
            {
                tmp = ReduceColBody(src + srcCol);
                Store<true>((__m128i*)(src0 + dstCol), tmp);
                Store<true>((__m128i*)(src1 + dstCol), tmp);
            }
            tmp = ReduceColTail<even>(src + srcTail);
            Store<false>((__m128i*)(src0 + dstWidth - HA), tmp);
            Store<false>((__m128i*)(src1 + dstWidth - HA), tmp);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t * s2 = src + srcStride*(row + 1);
                const uint8_t * s3 = s2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    s2 = src + srcStride*(srcHeight - 1);
                    s3 = s2;
                }

                Store<true>((__m128i*)src2, ReduceColNose(s2));
                Store<true>((__m128i*)src3, ReduceColNose(s3));
                size_t srcCol = A, dstCol = HA;
                for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                {
                    Store<true>((__m128i*)(src2 + dstCol), ReduceColBody(s2 + srcCol));
                    Store<true>((__m128i*)(src3 + dstCol), ReduceColBody(s3 + srcCol));
                }
                Store<false>((__m128i*)(src2 + dstWidth - HA), ReduceColTail<even>(s2 + srcTail));
                Store<false>((__m128i*)(src3 + dstWidth - HA), ReduceColTail<even>(s3 + srcTail));

                for (size_t col = 0; col < alignedDstWidth; col += HA)
                    _mm_storel_epi64((__m128i*)(dst + col), ReduceRow<true>(src0, src1, src2, src3, col));

                if (alignedDstWidth != dstWidth)
                    _mm_storel_epi64((__m128i*)(dst + dstWidth - HA), ReduceRow<false>(src0, src1, src2, src3, dstWidth - HA));

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
#endif
}
