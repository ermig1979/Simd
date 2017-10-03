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
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template<bool increment> v128_s16 InterferenceChange(v128_s16 statistic, v128_s16 value, v128_s16 saturation);

        template<> SIMD_INLINE v128_s16 InterferenceChange<true>(v128_s16 statistic, v128_s16 value, v128_s16 saturation)
        {
            return vec_min(vec_add(statistic, value), saturation);
        }

        template<> SIMD_INLINE v128_s16 InterferenceChange<false>(v128_s16 statistic, v128_s16 value, v128_s16 saturation)
        {
            return vec_max(vec_sub(statistic, value), saturation);
        }

        template<bool align, bool first, bool increment>
        SIMD_INLINE void InterferenceChange(const Loader<align> & statisticSrc, v128_s16 value, v128_s16 saturation, Storer<align> & statisticDst)
        {
            const v128_s16 statistic = (v128_s16)Load<align, first>(statisticSrc);
            Store<align, first>(statisticDst, InterferenceChange<increment>(statistic, value, saturation));
        }

        template <bool align, bool increment> void InterferenceChange(int16_t * statistic, size_t stride, size_t width, size_t height, uint8_t value, int16_t saturation)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(statistic) && Aligned(stride, HA));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            v128_s16 tailMask = (v128_s16)ShiftLeft(K16_FFFF, HA - width + alignedWidth);

            v128_s16 _value = SetI16(value);
            v128_s16 _saturation = SetI16(saturation);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> statisticSrc(statistic);
                Storer<align> statisticDst(statistic);
                InterferenceChange<align, true, increment>(statisticSrc, _value, _saturation, statisticDst);
                for (size_t col = HA; col < alignedWidth; col += HA)
                    InterferenceChange<align, false, increment>(statisticSrc, _value, _saturation, statisticDst);
                Flush(statisticDst);
                if (alignedWidth != width)
                {
                    Loader<false> statisticSrc(statistic + width - HA);
                    Storer<false> statisticDst(statistic + width - HA);
                    InterferenceChange<false, true, increment>(statisticSrc, vec_and(_value, tailMask), _saturation, statisticDst);
                    Flush(statisticDst);
                }
                statistic += stride;
            }
        }

        void InterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
        {
            assert(Aligned(stride, 2));

            if (Aligned(statistic) && Aligned(stride))
                InterferenceChange<true, true>((int16_t*)statistic, stride / 2, width, height, increment, saturation);
            else
                InterferenceChange<false, true>((int16_t*)statistic, stride / 2, width, height, increment, saturation);
        }

        void InterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation)
        {
            assert(Aligned(stride, 2));

            if (Aligned(statistic) && Aligned(stride))
                InterferenceChange<true, false>((int16_t*)statistic, stride / 2, width, height, decrement, saturation);
            else
                InterferenceChange<false, false>((int16_t*)statistic, stride / 2, width, height, decrement, saturation);
        }

        template<bool align, bool first, bool increment>
        SIMD_INLINE void InterferenceChangeMasked(const Loader<align> & statisticSrc, v128_s16 value, v128_s16 saturation,
            const Loader<align> & maskSrc, v128_u8 index, v128_u8 tailMask, Storer<align> & statisticDst)
        {
            v128_u8 mask = vec_and(vec_cmpeq(Load<align, first>(maskSrc), index), tailMask);
            InterferenceChange<align, first, increment>(statisticSrc, vec_and(value, (v128_s16)UnpackLoU8(mask, mask)), saturation, statisticDst);
            InterferenceChange<align, false, increment>(statisticSrc, vec_and(value, (v128_s16)UnpackHiU8(mask, mask)), saturation, statisticDst);
        }

        template <bool align, bool increment> void InterferenceChangeMasked(int16_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t value, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(statistic) && Aligned(statisticStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            v128_s16 _value = SetI16(value);
            v128_s16 _saturation = SetI16(saturation);
            v128_u8 _index = SetU8(index);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> statisticSrc(statistic), maskSrc(mask);
                Storer<align> statisticDst(statistic);
                InterferenceChangeMasked<align, true, increment>(statisticSrc, _value, _saturation, maskSrc, _index, K8_FF, statisticDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    InterferenceChangeMasked<align, false, increment>(statisticSrc, _value, _saturation, maskSrc, _index, K8_FF, statisticDst);
                Flush(statisticDst);
                if (alignedWidth != width)
                {
                    Loader<false> statisticSrc(statistic + width - A), maskSrc(mask + width - A);
                    Storer<false> statisticDst(statistic + width - A);
                    InterferenceChangeMasked<false, true, increment>(statisticSrc, _value, _saturation, maskSrc, _index, tailMask, statisticDst);
                    Flush(statisticDst);
                }
                statistic += statisticStride;
                mask += maskStride;
            }
        }

        void InterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(Aligned(statisticStride, 2));

            if (Aligned(statistic) && Aligned(statisticStride) && Aligned(mask) && Aligned(maskStride))
                InterferenceChangeMasked<true, true>((int16_t*)statistic, statisticStride / 2, width, height, increment, saturation, mask, maskStride, index);
            else
                InterferenceChangeMasked<false, true>((int16_t*)statistic, statisticStride / 2, width, height, increment, saturation, mask, maskStride, index);
        }

        void InterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
            uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
        {
            assert(Aligned(statisticStride, 2));

            if (Aligned(statistic) && Aligned(statisticStride) && Aligned(mask) && Aligned(maskStride))
                InterferenceChangeMasked<true, false>((int16_t*)statistic, statisticStride / 2, width, height, decrement, saturation, mask, maskStride, index);
            else
                InterferenceChangeMasked<false, false>((int16_t*)statistic, statisticStride / 2, width, height, decrement, saturation, mask, maskStride, index);
        }
    }
#endif// SIMD_VMX_ENABLE
}
