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
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
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

        SIMD_INLINE __m128i DivideBy64(__m128i value)
        {
            return _mm_srli_epi16(_mm_add_epi16(value, K16_0020), 6);
        }

        SIMD_INLINE __m128i ReduceColNose(const uint8_t *src)
        {
            const __m128i t1 = _mm_loadu_si128((__m128i*)src);
            const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
            return BinomialSum16(
                _mm_and_si128(LoadBeforeFirst<1>(t1), K16_00FF),
                _mm_and_si128(t1, K16_00FF),
                _mm_and_si128(t2, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
        }

        SIMD_INLINE __m128i ReduceColBody(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
            return BinomialSum16(
                _mm_and_si128(t0, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t0, 1), K16_00FF),
                _mm_and_si128(t2, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
        }

        template <bool even> SIMD_INLINE __m128i ReduceColTail(const uint8_t *src);

        template <> SIMD_INLINE __m128i ReduceColTail<true>(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            const __m128i t1 = _mm_loadu_si128((__m128i*)src);
            const __m128i t2 = LoadAfterLast<1>(t1);
            return BinomialSum16(
                _mm_and_si128(t0, K16_00FF),
                _mm_and_si128(t1, K16_00FF),
                _mm_and_si128(t2, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
        }

        template <> SIMD_INLINE __m128i ReduceColTail<false>(const uint8_t *src)
        {
            const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
            const __m128i t1 = LoadAfterLast<1>(t0);
            const __m128i t2 = LoadAfterLast<1>(t1);
            return BinomialSum16(
                _mm_and_si128(t0, K16_00FF),
                _mm_and_si128(t1, K16_00FF),
                _mm_and_si128(t2, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
        }

        template <bool align> SIMD_INLINE __m128i ReduceRow(const Buffer & buffer, size_t offset)
        {
            return _mm_packus_epi16(_mm_and_si128(DivideBy64(BinomialSum16(
                Load<align>((__m128i*)(buffer.src0 + offset)), Load<align>((__m128i*)(buffer.src1 + offset)),
                Load<align>((__m128i*)(buffer.src2 + offset)), Load<align>((__m128i*)(buffer.src3 + offset)))), K16_00FF), K_ZERO);
        }

        template <bool even> void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > A);

            size_t alignedDstWidth = Simd::AlignLo(dstWidth, HA);
            size_t srcTail = Simd::AlignHi(srcWidth - A, 2);

            Buffer buffer(Simd::AlignHi(dstWidth, A));

            __m128i tmp = ReduceColNose(src);
            Store<true>((__m128i*)buffer.src0, tmp);
            Store<true>((__m128i*)buffer.src1, tmp);
            size_t srcCol = A, dstCol = HA;
            for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
            {
                tmp = ReduceColBody(src + srcCol);
                Store<true>((__m128i*)(buffer.src0 + dstCol), tmp);
                Store<true>((__m128i*)(buffer.src1 + dstCol), tmp);
            }
            tmp = ReduceColTail<even>(src + srcTail);
            Store<false>((__m128i*)(buffer.src0 + dstWidth - HA), tmp);
            Store<false>((__m128i*)(buffer.src1 + dstWidth - HA), tmp);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                const uint8_t *src3 = src2 + srcStride;
                if (row >= srcHeight - 2)
                {
                    src2 = src + srcStride*(srcHeight - 1);
                    src3 = src2;
                }

                Store<true>((__m128i*)buffer.src2, ReduceColNose(src2));
                Store<true>((__m128i*)buffer.src3, ReduceColNose(src3));
                size_t srcCol = A, dstCol = HA;
                for (; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
                {
                    Store<true>((__m128i*)(buffer.src2 + dstCol), ReduceColBody(src2 + srcCol));
                    Store<true>((__m128i*)(buffer.src3 + dstCol), ReduceColBody(src3 + srcCol));
                }
                Store<false>((__m128i*)(buffer.src2 + dstWidth - HA), ReduceColTail<even>(src2 + srcTail));
                Store<false>((__m128i*)(buffer.src3 + dstWidth - HA), ReduceColTail<even>(src3 + srcTail));

                for (size_t col = 0; col < alignedDstWidth; col += HA)
                    _mm_storel_epi64((__m128i*)(dst + col), ReduceRow<true>(buffer, col));

                if (alignedDstWidth != dstWidth)
                    _mm_storel_epi64((__m128i*)(dst + dstWidth - HA), ReduceRow<false>(buffer, dstWidth - HA));

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
#endif// SIMD_SSE2_ENABLE
}
