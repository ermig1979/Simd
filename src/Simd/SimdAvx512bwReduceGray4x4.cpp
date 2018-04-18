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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i DivideBy64(__m512i value)
        {
            return _mm512_srli_epi16(_mm512_add_epi16(value, K16_0020), 6);
        }

        SIMD_INLINE __m512i BinomialSum16(const __m512i & a, const __m512i & b, const __m512i & c, const __m512i & d)
        {
            return _mm512_add_epi16(_mm512_add_epi16(a, d), _mm512_mullo_epi16(_mm512_add_epi16(b, c), K16_0003));
        }

        const __m512i K8_01_03 = SIMD_MM512_SET2_EPI8(1, 3);
        const __m512i K8_03_01 = SIMD_MM512_SET2_EPI8(3, 1);

        SIMD_INLINE __m512i BinomialSum8(const __m512i & ab, const __m512i & cd)
        {
            return _mm512_add_epi16(_mm512_maddubs_epi16(ab, K8_01_03), _mm512_maddubs_epi16(cd, K8_03_01));
        }

        SIMD_INLINE __m512i ReduceColNose(const uint8_t * src)
        {
            return BinomialSum8(LoadBeforeFirst<1>(src), Load<false>(src + 1));
        }

        SIMD_INLINE void ReduceColNose(const uint8_t * s[4], __m512i a[4])
        {
            a[0] = ReduceColNose(s[0]);
            a[1] = ReduceColNose(s[1]);
            a[2] = ReduceColNose(s[2]);
            a[3] = ReduceColNose(s[3]);
        }

        SIMD_INLINE __m512i ReduceColBody(const uint8_t * src)
        {
            return BinomialSum8(Load<false>(src - 1), Load<false>(src + 1));
        }

        SIMD_INLINE void ReduceColBody(const uint8_t * s[4], size_t offset, __m512i a[4])
        {
            a[0] = ReduceColBody(s[0] + offset);
            a[1] = ReduceColBody(s[1] + offset);
            a[2] = ReduceColBody(s[2] + offset);
            a[3] = ReduceColBody(s[3] + offset);
        }

        template <bool even> SIMD_INLINE __m512i ReduceColTail(const uint8_t * src);

        template <> SIMD_INLINE __m512i ReduceColTail<true>(const uint8_t * src)
        {
            return BinomialSum8(Load<false>(src - 1), LoadAfterLast<1>(src));
        }

        template <> SIMD_INLINE __m512i ReduceColTail<false>(const uint8_t * src)
        {
            return BinomialSum8(Load<false>(src - 1), LoadAfterLast2<1>(src - 1));
        }

        template <bool even> SIMD_INLINE void ReduceColTail(const uint8_t * s[4], size_t offset, __m512i a[4])
        {
            a[0] = ReduceColTail<even>(s[0] + offset);
            a[1] = ReduceColTail<even>(s[1] + offset);
            a[2] = ReduceColTail<even>(s[2] + offset);
            a[3] = ReduceColTail<even>(s[3] + offset);
        }

        SIMD_INLINE __m512i ReduceRow(const __m512i lo[4], const __m512i hi[4])
        {
            return _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(
                DivideBy64(BinomialSum16(lo[0], lo[1], lo[2], lo[3])),
                DivideBy64(BinomialSum16(hi[0], hi[1], hi[2], hi[3]))));
        }

        template <bool even> void ReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth > DA);

            size_t bodyWidth = AlignLo(srcWidth, DA);
            size_t srcTail = Simd::AlignHi(srcWidth - DA, 2);

            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
            {
                const uint8_t * s[4];
                s[1] = src + srcStride*row;
                s[0] = s[1] - (row ? srcStride : 0);
                s[2] = s[1] + (row < srcHeight - 1 ? srcStride : 0);
                s[3] = s[2] + (row < srcHeight - 2 ? srcStride : 0);

                __m512i lo[4], hi[4];
                ReduceColNose(s, lo);
                ReduceColBody(s, A, hi);
                Store<false>(dst, ReduceRow(lo, hi));
                for (size_t srcCol = DA, dstCol = A; srcCol < bodyWidth; srcCol += DA, dstCol += A)
                {
                    ReduceColBody(s, srcCol + 0, lo);
                    ReduceColBody(s, srcCol + A, hi);
                    Store<false>(dst + dstCol, ReduceRow(lo, hi));
                }
                ReduceColBody(s, srcTail + 0, lo);
                ReduceColTail<even>(s, srcTail + A, hi);
                Store<false>(dst + dstWidth - A, ReduceRow(lo, hi));
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
#endif// SIMD_AVX512BW_ENABLE
}
