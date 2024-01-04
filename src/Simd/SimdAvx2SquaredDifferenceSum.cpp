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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i SquaredDifference(__m256i a, __m256i b)
        {
            const __m256i lo = SubUnpackedU8<0>(a, b);
            const __m256i hi = SubUnpackedU8<1>(a, b);
            return _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
        }

        template <bool align> void SquaredDifferenceSum(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);
            if (align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            for (size_t row = 0; row < height; ++row)
            {
                __m256i rowSum = _mm256_setzero_si256();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i a_ = Load<align>((__m256i*)(a + col));
                    const __m256i b_ = Load<align>((__m256i*)(b + col));
                    rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
                }
                if (width - bodyWidth)
                {
                    const __m256i a_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(a + width - A)));
                    const __m256i b_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(b + width - A)));
                    rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
                }
                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> void SquaredDifferenceSumMasked(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);
            if (align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            __m256i index_ = _mm256_set1_epi8(index);
            for (size_t row = 0; row < height; ++row)
            {
                __m256i rowSum = _mm256_setzero_si256();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i mask_ = LoadMaskI8<align>((__m256i*)(mask + col), index_);
                    const __m256i a_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(a + col)));
                    const __m256i b_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(b + col)));
                    rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
                }
                if (width - bodyWidth)
                {
                    const __m256i mask_ = _mm256_and_si256(tailMask, LoadMaskI8<false>((__m256i*)(mask + width - A), index_));
                    const __m256i a_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(a + width - A)));
                    const __m256i b_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(b + width - A)));
                    rowSum = _mm256_add_epi32(rowSum, SquaredDifference(a_, b_));
                }
                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void SquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                SquaredDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                SquaredDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float* a, const float* b, size_t offset, __m256& sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            __m256 _d = _mm256_sub_ps(_a, _b);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_d, _d));
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float* a, const float* b, size_t size, float* sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 8);
            size_t fullAlignedSize = AlignLo(size, 32);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += 32)
                    {
                        SquaredDifferenceSum32f<align>(a, b, i, sums[0]);
                        SquaredDifferenceSum32f<align>(a, b, i + 8, sums[1]);
                        SquaredDifferenceSum32f<align>(a, b, i + 16, sums[2]);
                        SquaredDifferenceSum32f<align>(a, b, i + 24, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += 8)
                    SquaredDifferenceSum32f<align>(a, b, i, sums[0]);
                *sum += Avx::ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceSum32f(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum32f<false>(a, b, size, sum);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float* a, const float* b, size_t offset, __m256& sum, __m256& correction)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            __m256 _d = _mm256_sub_ps(_a, _b);
            __m256 term = _mm256_sub_ps(_mm256_mul_ps(_d, _d), correction);
            __m256 temp = _mm256_add_ps(sum, term);
            correction = _mm256_sub_ps(_mm256_sub_ps(temp, sum), term);
            sum = temp;
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float* a, const float* b, size_t size, float* sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 8);
            size_t fullAlignedSize = AlignLo(size, 32);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                __m256 corrections[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += 32)
                    {
                        SquaredDifferenceKahanSum32f<align>(a, b, i, sums[0], corrections[0]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 8, sums[1], corrections[1]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 16, sums[2], corrections[2]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 24, sums[3], corrections[3]);
                    }
                }
                for (; i < partialAlignedSize; i += 8)
                    SquaredDifferenceKahanSum32f<align>(a, b, i, sums[0], corrections[0]);
                *sum += Avx::ExtractSum(_mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
            }
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceKahanSum32f(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceKahanSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceKahanSum32f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
