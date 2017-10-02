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

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint8x16_t FeatureDifference(uint8x16_t value, uint8x16_t lo, uint8x16_t hi)
        {
            return vmaxq_u8(vqsubq_u8(value, hi), vqsubq_u8(lo, value));
        }

        SIMD_INLINE uint16x8_t ShiftedWeightedSquare(uint8x8_t difference, uint16x4_t weight)
        {
            uint16x8_t square = vmull_u8(difference, difference);
            uint16x4_t lo = vshrn_n_u32(vmull_u16(Half<0>(square), weight), 16);
            uint16x4_t hi = vshrn_n_u32(vmull_u16(Half<1>(square), weight), 16);
            return vcombine_u16(lo, hi);
        }

        SIMD_INLINE uint8x16_t ShiftedWeightedSquare(uint8x16_t difference, uint16x4_t weight)
        {
            const uint16x8_t lo = ShiftedWeightedSquare(Half<0>(difference), weight);
            const uint16x8_t hi = ShiftedWeightedSquare(Half<1>(difference), weight);
            return PackSaturatedU16(lo, hi);
        }

        template <bool align> SIMD_INLINE void AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi,
            uint8_t * difference, size_t offset, uint16x4_t weight, uint8x16_t mask)
        {
            const uint8x16_t _value = Load<align>(value + offset);
            const uint8x16_t _lo = Load<align>(lo + offset);
            const uint8x16_t _hi = Load<align>(hi + offset);
            uint8x16_t _difference = Load<align>(difference + offset);

            const uint8x16_t featureDifference = FeatureDifference(_value, _lo, _hi);
            const uint8x16_t inc = vandq_u8(mask, ShiftedWeightedSquare(featureDifference, weight));
            Store<align>(difference + offset, vqaddq_u8(_difference, inc));
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
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            uint16x4_t _weight = vdup_n_u16(weight);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddFeatureDifference<align>(value, lo, hi, difference, col, _weight, K8_FF);
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
#endif// SIMD_NEON_ENABLE
}
