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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
	namespace Neon
	{
        template <bool align, SimdCompareType compareType> 
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

#ifdef __GNUC__
			uint8x16_t _value = SIMD_VEC_SET1_EPI8(value);
			uint8x16_t _positive = SIMD_VEC_SET1_EPI8(positive);
			uint8x16_t _negative = SIMD_VEC_SET1_EPI8(negative);
#else
			uint8x16_t _value = vld1q_dup_u8(&value);
			uint8x16_t _positive = vld1q_dup_u8(&positive);
			uint8x16_t _negative = vld1q_dup_u8(&negative);
#endif

            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const uint8x16_t mask = Compare8u<compareType>(Load<align>(src + col), _value);
                    Store<align>(dst + col, vbslq_u8(mask, _positive, _negative));
                }
                if(alignedWidth != width)
                {
                    const uint8x16_t mask = Compare8u<compareType>(Load<false>(src + width - A), _value);
                    Store<false>(dst + width - A, vbslq_u8(mask, _positive, _negative));
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <SimdCompareType compareType> 
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Binarization<true, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            else
                Binarization<false, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
        }

        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType)
        {
            switch(compareType)
            {
            case SimdCompareEqual: 
                return Binarization<SimdCompareEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareNotEqual: 
                return Binarization<SimdCompareNotEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreater: 
                return Binarization<SimdCompareGreater>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual: 
                return Binarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesser: 
                return Binarization<SimdCompareLesser>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual: 
                return Binarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default: 
                assert(0);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}