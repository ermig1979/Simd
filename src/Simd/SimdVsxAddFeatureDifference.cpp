/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
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
            const v128_u16 lo = ShiftedWeightedSquare16(UnpackLoU8(difference), weight);
            const v128_u16 hi = ShiftedWeightedSquare16(UnpackHiU8(difference), weight);
            return vec_packsu(lo, hi);
        }

        template <bool align, bool first> 
        SIMD_INLINE void AddFeatureDifference(const Loader<align> & value, const Loader<align> & lo, const Loader<align> & hi, 
            const Loader<align> & differenceSrc, v128_u16 weight, v128_u8 mask, Storer<align> & differenceDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _lo = Load<align, first>(lo);
            const v128_u8 _hi = Load<align, first>(hi);
            v128_u8 _difference = Load<align, first>(differenceSrc);

            const v128_u8 featureDifference = FeatureDifference(_value, _lo, _hi);
            const v128_u8 inc = vec_and(mask, ShiftedWeightedSquare8(featureDifference, weight));
            Store<align, first>(differenceDst, vec_adds(_difference, inc));
        }

        template <bool align> void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height, 
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
            uint16_t weight, uint8_t * difference, size_t differenceStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
                assert(Aligned(difference) && Aligned(differenceStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            const v128_u16 _weight = SIMD_VEC_SET1_EPI16(weight);

            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _lo(lo), _hi(hi), _differenceSrc(difference);
                Storer<align> _differenceDst(difference);
                AddFeatureDifference<align, true>(_value, _lo, _hi, _differenceSrc, _weight, K8_FF, _differenceDst);
                for(size_t col = A; col < alignedWidth; col += A)
                    AddFeatureDifference<align, false>(_value, _lo, _hi, _differenceSrc, _weight, K8_FF, _differenceDst);
                Flush(_differenceDst);

                if(alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _lo(lo + width - A), _hi(hi + width - A), _differenceSrc(difference + width - A);
                    Storer<false> _differenceDst(difference + width - A);
                    AddFeatureDifference<false, true>(_value, _lo, _hi, _differenceSrc, _weight, tailMask, _differenceDst);
                    Flush(_differenceDst);
                }
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
            if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && 
                Aligned(hi) && Aligned(hiStride) && Aligned(difference) && Aligned(differenceStride))
                AddFeatureDifference<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
            else
                AddFeatureDifference<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}