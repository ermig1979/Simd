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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i FeatureDifference(__m256i value, __m256i lo, __m256i hi)
        {
            return _mm256_max_epu8(_mm256_subs_epu8(value, hi), _mm256_subs_epu8(lo, value));
        }

        SIMD_INLINE __m256i ShiftedWeightedSquare16(__m256i difference, __m256i weight)
        {
            return _mm256_mulhi_epu16(_mm256_mullo_epi16(difference, difference), weight);
        }

        SIMD_INLINE __m256i ShiftedWeightedSquare8(__m256i difference, __m256i weight)
        {
            const __m256i lo = ShiftedWeightedSquare16(_mm256_unpacklo_epi8(difference, K_ZERO), weight);
            const __m256i hi = ShiftedWeightedSquare16(_mm256_unpackhi_epi8(difference, K_ZERO), weight);
            return _mm256_packus_epi16(lo, hi);
        }

        template <bool align> SIMD_INLINE void AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi,
            uint8_t * difference, size_t offset, __m256i weight, __m256i mask)
        {
            const __m256i _value = Load<align>((__m256i*)(value + offset));
            const __m256i _lo = Load<align>((__m256i*)(lo + offset));
            const __m256i _hi = Load<align>((__m256i*)(hi + offset));
            __m256i _difference = Load<align>((__m256i*)(difference + offset));

            const __m256i featureDifference = FeatureDifference(_value, _lo, _hi);
            const __m256i inc = _mm256_and_si256(mask, ShiftedWeightedSquare8(featureDifference, weight));
            Store<align>((__m256i*)(difference + offset), _mm256_adds_epu8(_difference, inc));
        }

        template <bool align> void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
            uint16_t weight, uint8_t * difference, size_t differenceStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
                assert(Aligned(difference) && Aligned(differenceStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            __m256i _weight = _mm256_set1_epi16((short)weight);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddFeatureDifference<align>(value, lo, hi, difference, col, _weight, K_INV_ZERO);
                if (alignedWidth != width)
                    AddFeatureDifference<false>(value, lo, hi, difference, width - A, _weight, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                difference += differenceStride;
            }
        }

        void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
            uint16_t weight, uint8_t * difference, size_t differenceStride)
        {
            if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) &&
                Aligned(hi) && Aligned(hiStride) && Aligned(difference) && Aligned(differenceStride))
                AddFeatureDifference<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
            else
                AddFeatureDifference<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
