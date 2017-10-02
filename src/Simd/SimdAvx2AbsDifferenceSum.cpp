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
#include "Simd/SimdLoad.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> void AbsDifferenceSum(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i a_ = Load<align>((__m256i*)(a + col));
                    const __m256i b_ = Load<align>((__m256i*)(b + col));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
                }
                if (width - bodyWidth)
                {
                    const __m256i a_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(a + width - A)));
                    const __m256i b_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(b + width - A)));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
                }
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        template <bool align> void AbsDifferenceSumMasked(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
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
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i mask_ = LoadMaskI8<align>((__m256i*)(mask + col), index_);
                    const __m256i a_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(a + col)));
                    const __m256i b_ = _mm256_and_si256(mask_, Load<align>((__m256i*)(b + col)));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
                }
                if (width - bodyWidth)
                {
                    const __m256i mask_ = _mm256_and_si256(tailMask, LoadMaskI8<false>((__m256i*)(mask + width - A), index_));
                    const __m256i a_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(a + width - A)));
                    const __m256i b_ = _mm256_and_si256(mask_, Load<false>((__m256i*)(b + width - A)));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(a_, b_), fullSum);
                }
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        void AbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                AbsDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                AbsDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <bool align> void AbsDifferenceSums3(__m256i current, const uint8_t * background, __m256i sums[3])
        {
            sums[0] = _mm256_add_epi64(sums[0], _mm256_sad_epu8(current, Load<align>((__m256i*)(background - 1))));
            sums[1] = _mm256_add_epi64(sums[1], _mm256_sad_epu8(current, Load<false>((__m256i*)(background))));
            sums[2] = _mm256_add_epi64(sums[2], _mm256_sad_epu8(current, Load<false>((__m256i*)(background + 1))));
        }

        template <bool align> void AbsDifferenceSums3x3(__m256i current, const uint8_t * background, size_t stride, __m256i sums[9])
        {
            AbsDifferenceSums3<align>(current, background - stride, sums + 0);
            AbsDifferenceSums3<align>(current, background, sums + 3);
            AbsDifferenceSums3<align>(current, background + stride, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3Masked(__m256i current, const uint8_t * background, __m256i mask, __m256i sums[3])
        {
            sums[0] = _mm256_add_epi64(sums[0], _mm256_sad_epu8(current, _mm256_and_si256(mask, Load<align>((__m256i*)(background - 1)))));
            sums[1] = _mm256_add_epi64(sums[1], _mm256_sad_epu8(current, _mm256_and_si256(mask, Load<false>((__m256i*)(background)))));
            sums[2] = _mm256_add_epi64(sums[2], _mm256_sad_epu8(current, _mm256_and_si256(mask, Load<false>((__m256i*)(background + 1)))));
        }

        template <bool align> void AbsDifferenceSums3x3Masked(__m256i current, const uint8_t * background, size_t stride, __m256i mask, __m256i sums[9])
        {
            AbsDifferenceSums3Masked<align>(current, background - stride, mask, sums + 0);
            AbsDifferenceSums3Masked<align>(current, background, mask, sums + 3);
            AbsDifferenceSums3Masked<align>(current, background + stride, mask, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride,
            const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width >= A + 2);
            if (align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);

            __m256i fullSums[9];
            for (size_t i = 0; i < 9; ++i)
                fullSums[i] = _mm256_setzero_si256();

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i _current = Load<false>((__m256i*)(current + col));
                    AbsDifferenceSums3x3<align>(_current, background + col, backgroundStride, fullSums);
                }
                if (width - bodyWidth)
                {
                    const __m256i _current = _mm256_and_si256(tailMask, Load<false>((__m256i*)(current + width - A)));
                    AbsDifferenceSums3x3Masked<false>(_current, background + width - A, backgroundStride, tailMask, fullSums);
                }
                current += currentStride;
                background += backgroundStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum<uint64_t>(fullSums[i]);
        }

        void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            size_t width, size_t height, uint64_t * sums)
        {
            if (Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3<true>(current, currentStride, background, backgroundStride, width, height, sums);
            else
                AbsDifferenceSums3x3<false>(current, currentStride, background, backgroundStride, width, height, sums);
        }

        template <bool align> void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width >= A + 2);
            if (align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;
            mask += 1 + maskStride;

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i _index = _mm256_set1_epi8(index);

            __m256i fullSums[9];
            for (size_t i = 0; i < 9; ++i)
                fullSums[i] = _mm256_setzero_si256();

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i _mask = LoadMaskI8<false>((__m256i*)(mask + col), _index);
                    const __m256i _current = _mm256_and_si256(Load<false>((__m256i*)(current + col)), _mask);
                    AbsDifferenceSums3x3Masked<align>(_current, background + col, backgroundStride, _mask, fullSums);
                }
                if (width - bodyWidth)
                {
                    const __m256i _mask = _mm256_and_si256(LoadMaskI8<false>((__m256i*)(mask + width - A), _index), tailMask);
                    const __m256i _current = _mm256_and_si256(_mask, Load<false>((__m256i*)(current + width - A)));
                    AbsDifferenceSums3x3Masked<false>(_current, background + width - A, backgroundStride, _mask, fullSums);
                }
                current += currentStride;
                background += backgroundStride;
                mask += maskStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum<uint64_t>(fullSums[i]);
        }

        void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            if (Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3Masked<true>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
            else
                AbsDifferenceSums3x3Masked<false>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
