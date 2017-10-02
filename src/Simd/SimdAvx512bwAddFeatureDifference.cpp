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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i FeatureDifference(__m512i value, __m512i lo, __m512i hi)
        {
            return _mm512_max_epu8(_mm512_subs_epu8(value, hi), _mm512_subs_epu8(lo, value));
        }

        SIMD_INLINE __m512i ShiftedWeightedSquare16(__m512i difference, __m512i weight)
        {
            return _mm512_mulhi_epu16(_mm512_mullo_epi16(difference, difference), weight);
        }

        SIMD_INLINE __m512i ShiftedWeightedSquare8(__m512i difference, __m512i weight)
        {
            const __m512i lo = ShiftedWeightedSquare16(UnpackU8<0>(difference), weight);
            const __m512i hi = ShiftedWeightedSquare16(UnpackU8<1>(difference), weight);
            return _mm512_packus_epi16(lo, hi);
        }

        template <bool align, bool mask> SIMD_INLINE void AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi,
            uint8_t * difference, size_t offset, __m512i weight, __mmask64 m = -1)
        {
            const __m512i _value = Load<align, mask>(value + offset, m);
            const __m512i _lo = Load<align, mask>(lo + offset, m);
            const __m512i _hi = Load<align, mask>(hi + offset, m);
            __m512i _difference = Load<align, mask>(difference + offset, m);

            const __m512i featureDifference = FeatureDifference(_value, _lo, _hi);
            const __m512i inc = ShiftedWeightedSquare8(featureDifference, weight);
            Store<align, mask>(difference + offset, _mm512_adds_epu8(_difference, inc), m);
        }

        template <bool align> void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
            uint16_t weight, uint8_t * difference, size_t differenceStride)
        {
            if (align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
                assert(Aligned(difference) && Aligned(differenceStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
            __m512i _weight = _mm512_set1_epi16((short)weight);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    AddFeatureDifference<align, false>(value, lo, hi, difference, col, _weight);
                if (col < width)
                    AddFeatureDifference<align, true>(value, lo, hi, difference, col, _weight, tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
