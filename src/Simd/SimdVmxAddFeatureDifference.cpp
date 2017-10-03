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
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        SIMD_INLINE v128_u8 FeatureDifference(v128_u8 value, v128_u8 lo, v128_u8 hi)
        {
            return vec_max(vec_subs(value, hi), vec_subs(lo, value));
        }

        SIMD_INLINE v128_u16 ShiftedWeightedSquare16(v128_u16 difference, v128_u16 weight)
        {
            return MulHiU16(vec_mladd(difference, difference, K16_0000), weight);
        }

        SIMD_INLINE v128_u8 ShiftedWeightedSquare8(v128_u8 difference, v128_u16 weight)
        {
            const v128_u16 lo = ShiftedWeightedSquare16(UnpackU8<0>(difference), weight);
            const v128_u16 hi = ShiftedWeightedSquare16(UnpackU8<1>(difference), weight);
            return vec_packsu(lo, hi);
        }

        template <bool align> SIMD_INLINE v128_u8 FeatureDifferenceInc(const uint8_t * value, const uint8_t * lo, const uint8_t * hi, size_t offset, v128_u16 weight)
        {
            const v128_u8 _value = Load<align>(value + offset);
            const v128_u8 _lo = Load<align>(lo + offset);
            const v128_u8 _hi = Load<align>(hi + offset);
            return ShiftedWeightedSquare8(FeatureDifference(_value, _lo, _hi), weight);
        }

        template <bool align> SIMD_INLINE void AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi, size_t offset, v128_u16 weight, uint8_t * difference)
        {
            const v128_u8 _increase = FeatureDifferenceInc<align>(value, lo, hi, offset, weight);
            const v128_u8 _difference = Load<align>(difference + offset);
            Store<align>(difference + offset, vec_adds(_difference, _increase));
        }

        SIMD_INLINE void AddFeatureDifferenceMasked(const uint8_t * value, const uint8_t * lo, const uint8_t * hi, size_t offset, v128_u16 weight, v128_u8 mask, uint8_t * difference)
        {
            const v128_u8 _increase = FeatureDifferenceInc<false>(value, lo, hi, offset, weight);
            const v128_u8 _difference = Load<false>(difference + offset);
            Store<false>(difference + offset, vec_adds(_difference, vec_and(_increase, mask)));
        }

        template <bool align> SIMD_INLINE v128_u8 AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi, const uint8_t * difference, size_t offset, v128_u16 weight)
        {
            const v128_u8 _increase = FeatureDifferenceInc<align>(value, lo, hi, offset, weight);
            const v128_u8 _difference = Load<align>(difference + offset);
            return vec_adds(_difference, _increase);
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
            size_t fullAlignedWidth = AlignLo(width, QA);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            const v128_u16 _weight = SIMD_VEC_SET1_EPI16(weight);

            for (size_t row = 0; row < height; ++row)
            {
                if (align)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                    {
                        AddFeatureDifference<align>(value, lo, hi, col, _weight, difference);
                        AddFeatureDifference<align>(value, lo, hi, col + A, _weight, difference);
                        AddFeatureDifference<align>(value, lo, hi, col + 2 * A, _weight, difference);
                        AddFeatureDifference<align>(value, lo, hi, col + 3 * A, _weight, difference);
                    }
                    for (; col < alignedWidth; col += A)
                        AddFeatureDifference<align>(value, lo, hi, col, _weight, difference);
                }
                else
                {
                    Storer<align> _difference(difference);
                    _difference.First(AddFeatureDifference<align>(value, lo, hi, difference, 0, _weight));
                    for (size_t col = A; col < alignedWidth; col += A)
                        _difference.Next(AddFeatureDifference<align>(value, lo, hi, difference, col, _weight));
                    Flush(_difference);
                }
                if (alignedWidth != width)
                    AddFeatureDifferenceMasked(value, lo, hi, width - A, _weight, tailMask, difference);
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
#endif// SIMD_VMX_ENABLE
}
