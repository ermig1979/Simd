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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 4 * width);
                    src0 = (uint16_t*)_p;
                    src1 = src0 + width;
                    src2 = src1 + width;
                    src3 = src2 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * src0;
                uint16_t * src1;
                uint16_t * src2;
                uint16_t * src3;
            private:
                void * _p;
            };
        }

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

        template <bool align> SIMD_INLINE __m256i ReduceRow16(const Buffer & buffer, size_t offset)
        {
            return _mm256_and_si256(DivideBy64(BinomialSum16(
                Load<align>((__m256i*)(buffer.src0 + offset)), Load<align>((__m256i*)(buffer.src1 + offset)),
                Load<align>((__m256i*)(buffer.src2 + offset)), Load<align>((__m256i*)(buffer.src3 + offset)))), K16_00FF);
        }

        template <bool align> SIMD_INLINE __m256i ReduceRow8(const Buffer & buffer, size_t offset)
        {
            __m256i lo = ReduceRow16<align>(buffer, offset);
            __m256i hi = ReduceRow16<align>(buffer, offset + HA);
            return PackI16ToU8(lo, hi);
        }

        template <bool even> void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > DA);

            size_t alignedDstWidth = Simd::AlignLo(dstWidth, A);
            size_t srcTail = Simd::AlignHi(srcWidth - A, 2);

            Buffer buffer(Simd::AlignHi(dstWidth, A));

            __m256i tmp = ReduceColNose(src);
            Store<true>((__m256i*)buffer.src0, tmp);
            Store<true>((__m256i*)buffer.src1, tmp);
            size_t srcCol = A, dstCol = HA;
            for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
            {
                tmp = ReduceColBody(src + srcCol);
                Store<true>((__m256i*)(buffer.src0 + dstCol), tmp);
                Store<true>((__m256i*)(buffer.src1 + dstCol), tmp);
            }
            tmp = ReduceColTail<even>(src + srcTail);
            Store<false>((__m256i*)(buffer.src0 + dstWidth - HA), tmp);
            Store<false>((__m256i*)(buffer.src1 + dstWidth - HA), tmp);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                const uint8_t *src3 = src2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    src2 = src + srcStride*(srcHeight - 1);
                    src3 = src2;
                }

                Store<true>((__m256i*)buffer.src2, ReduceColNose(src2));
                Store<true>((__m256i*)buffer.src3, ReduceColNose(src3));
                size_t srcCol = A, dstCol = HA;
                for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                {
                    Store<true>((__m256i*)(buffer.src2 + dstCol), ReduceColBody(src2 + srcCol));
                    Store<true>((__m256i*)(buffer.src3 + dstCol), ReduceColBody(src3 + srcCol));
                }
                Store<false>((__m256i*)(buffer.src2 + dstWidth - HA), ReduceColTail<even>(src2 + srcTail));
                Store<false>((__m256i*)(buffer.src3 + dstWidth - HA), ReduceColTail<even>(src3 + srcTail));

                Store<false>((__m256i*)dst, ReduceRow8<true>(buffer, 0));
                for (size_t col = A; col < alignedDstWidth; col += A)
                    Store<false>((__m256i*)(dst + col), ReduceRow8<true>(buffer, col));

                if (alignedDstWidth != dstWidth)
                    Store<false>((__m256i*)(dst + dstWidth - A), ReduceRow8<false>(buffer, dstWidth - A));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src1, buffer.src3);
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
