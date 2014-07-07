/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool align, bool first> 
        SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const Loader<align> & value, const Loader<align> & backgroundSrc, v128_u8 mask, Storer<align> & backgroundDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            const v128_u8 inc = vec_and(mask, vec_cmpgt(_value, _background));
            Store<align, first>(backgroundDst, vec_adds(_background, inc));
        }

        template <bool align> void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _backgroundSrc(background);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundGrowRangeSlow<align, true>(_value, _backgroundSrc, K8_01, _backgroundDst);
                for(size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeSlow<align, false>(_value, _backgroundSrc, K8_01, _backgroundDst);
                _backgroundDst.Flush();

                if(alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _backgroundSrc(background + width - A);
                    Storer<false> _backgroundDst(background + width - A);
                    EdgeBackgroundGrowRangeSlow<false, true>(_value, _backgroundSrc, tailMask, _backgroundDst);
                    _backgroundDst.Flush();
                }

                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundGrowRangeSlow<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundGrowRangeSlow<false>(value, valueStride, width, height, background, backgroundStride);
        }

        template <bool align, bool first> 
        SIMD_INLINE void EdgeBackgroundGrowRangeFast(const Loader<align> & value, const Loader<align> & backgroundSrc, Storer<align> & backgroundDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            Store<align, first>(backgroundDst, vec_max(_background, _value));
        }

        template <bool align> void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _backgroundSrc(background);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundGrowRangeFast<align, true>(_value, _backgroundSrc, _backgroundDst);
                for(size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeFast<align, false>(_value, _backgroundSrc, _backgroundDst);
                _backgroundDst.Flush();

                if(alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _backgroundSrc(background + width - A);
                    Storer<false> _backgroundDst(background + width - A);
                    EdgeBackgroundGrowRangeFast<false, true>(_value, _backgroundSrc, _backgroundDst);
                    _backgroundDst.Flush();
                }

                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundGrowRangeFast<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundGrowRangeFast<false>(value, valueStride, width, height, background, backgroundStride);
        }

        template <bool align, bool first> 
        SIMD_INLINE void EdgeBackgroundShiftRange(const Loader<align> & value, const Loader<align> & backgroundSrc, const v128_u8 & mask, Storer<align> & backgroundDst)
        {
            const v128_u8 _value = Load<align, first>(value);
            const v128_u8 _background = Load<align, first>(backgroundSrc);
            Store<align, first>(backgroundDst, vec_sel(_background, _value, mask));
        }

        template <bool align> void EdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _value(value), _backgroundSrc(background);
                Storer<align> _backgroundDst(background);
                EdgeBackgroundShiftRange<align, true>(_value, _backgroundSrc, K8_FF, _backgroundDst);
                for(size_t col = A; col < alignedWidth; col += A)
                    EdgeBackgroundShiftRange<align, false>(_value, _backgroundSrc, K8_FF, _backgroundDst);
                _backgroundDst.Flush();

                if(alignedWidth != width)
                {
                    Loader<false> _value(value + width - A), _backgroundSrc(background + width - A);
                    Storer<false> _backgroundDst(background + width - A);
                    EdgeBackgroundShiftRange<false, true>(_value, _backgroundSrc, tailMask, _backgroundDst);
                    _backgroundDst.Flush();
                }

                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * background, size_t backgroundStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundShiftRange<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundShiftRange<false>(value, valueStride, width, height, background, backgroundStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}