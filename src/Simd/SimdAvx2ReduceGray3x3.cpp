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
        template <bool compensation> SIMD_INLINE __m256i DivideBy16(__m256i value);

        template <> SIMD_INLINE __m256i DivideBy16<true>(__m256i value)
        {
            return _mm256_srli_epi16(_mm256_add_epi16(value, K16_0008), 4);
        }

        template <> SIMD_INLINE __m256i DivideBy16<false>(__m256i value)
        {
            return _mm256_srli_epi16(value, 4);
        }

        const __m256i K16_0102 = SIMD_MM256_SET1_EPI16(0x0102);

        SIMD_INLINE __m256i BinomialSum8(const __m256i & s01, const __m256i & s12)
        {
#ifdef SIMD_MADDUBS_ERROR
            return BinomialSum16(_mm256_and_si256(s01, K16_00FF), _mm256_and_si256(s12, K16_00FF), _mm256_and_si256(_mm256_srli_si256(s12, 1), K16_00FF));
#else
            return _mm256_add_epi16(_mm256_and_si256(s01, K16_00FF), _mm256_maddubs_epi16(s12, K16_0102));
#endif
        }

        template<bool align> SIMD_INLINE __m256i ReduceColNose(const uint8_t * p)
        {
            return BinomialSum8(LoadBeforeFirst<align, 1>(p), Load<align>((__m256i*)p));
        }

        template<bool align> SIMD_INLINE void ReduceColNose(const uint8_t * s[3], __m256i a[3])
        {
            a[0] = ReduceColNose<align>(s[0]);
            a[1] = ReduceColNose<align>(s[1]);
            a[2] = ReduceColNose<align>(s[2]);
        }

        template<bool align> SIMD_INLINE __m256i ReduceColBody(const uint8_t * p)
        {
            return BinomialSum8(Load<false>((__m256i*)(p - 1)), Load<align>((__m256i*)p));
        }

        template<bool align> SIMD_INLINE void ReduceColBody(const uint8_t * s[3], size_t offset, __m256i a[3])
        {
            a[0] = ReduceColBody<align>(s[0] + offset);
            a[1] = ReduceColBody<align>(s[1] + offset);
            a[2] = ReduceColBody<align>(s[2] + offset);
        }

        template <bool compensation> SIMD_INLINE __m256i ReduceRow(const __m256i lo[3], const __m256i hi[3])
        {
            return PackI16ToU8(
                DivideBy16<compensation>(BinomialSum16(lo[0], lo[1], lo[2])),
                DivideBy16<compensation>(BinomialSum16(hi[0], hi[1], hi[2])));
        }

        template<bool align, bool compensation> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth >= DA && (srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            if (align)
                assert(Aligned(src) && Aligned(srcStride));

            size_t lastOddCol = srcWidth - AlignLo(srcWidth, 2);
            size_t bodyWidth = AlignLo(srcWidth, DA);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t * s[3];
                s[1] = src;
                s[0] = s[1] - (row ? srcStride : 0);
                s[2] = s[1] + (row != srcHeight - 1 ? srcStride : 0);

                __m256i lo[3], hi[3];
                ReduceColNose<align>(s, lo);
                ReduceColBody<align>(s, A, hi);
                Store<false>((__m256i*)dst, ReduceRow<compensation>(lo, hi));

                for (size_t srcCol = DA, dstCol = A; srcCol < bodyWidth; srcCol += DA, dstCol += A)
                {
                    ReduceColBody<align>(s, srcCol, lo);
                    ReduceColBody<align>(s, srcCol + A, hi);
                    Store<false>((__m256i*)(dst + dstCol), ReduceRow<compensation>(lo, hi));
                }

                if (bodyWidth != srcWidth)
                {
                    size_t srcCol = srcWidth - DA - lastOddCol;
                    size_t dstCol = dstWidth - A - lastOddCol;
                    ReduceColBody<false>(s, srcCol, lo);
                    ReduceColBody<false>(s, srcCol + A, hi);
                    Store<false>((__m256i*)(dst + dstCol), ReduceRow<compensation>(lo, hi));
                    if (lastOddCol)
                        dst[dstWidth - 1] = Base::GaussianBlur3x3<compensation>(s[0] + srcWidth, s[1] + srcWidth, s[2] + srcWidth, -2, -1, -1);
                }
            }
        }

        template<bool align> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray3x3<align, true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray3x3<align, false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }

        void ReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (Aligned(src) && Aligned(srcStride))
                ReduceGray3x3<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
            else
                ReduceGray3x3<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
