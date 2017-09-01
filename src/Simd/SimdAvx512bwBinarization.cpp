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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
		template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void Binarization(const uint8_t * src, 
			const __m512i & value, const __m512i & positive, const __m512i & negative, uint8_t * dst, __mmask64 m = -1)
		{
			__mmask64 mm = Compare8u<compareType>(Load<align, mask>(src, m), value);
			Store<align, mask>(dst, _mm512_mask_blend_epi8(mm, negative, positive), m);
		}

		template <bool align, SimdCompareType compareType> SIMD_INLINE void Binarization4(const uint8_t * src,
			const __m512i & value, const __m512i & positive, const __m512i & negative, uint8_t * dst)
		{
			Store<align>(dst + 0 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 0 * A), value), negative, positive));
			Store<align>(dst + 1 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 1 * A), value), negative, positive));
			Store<align>(dst + 2 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 2 * A), value), negative, positive));
			Store<align>(dst + 3 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 3 * A), value), negative, positive));
		}

        template <bool align, SimdCompareType compareType> 
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
			size_t fullAlignedWidth = Simd::AlignLo(width, QA);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i _value = _mm512_set1_epi8(value);
            __m512i _positive = _mm512_set1_epi8(positive);
            __m512i _negative = _mm512_set1_epi8(negative);
            for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
				for (; col < fullAlignedWidth; col += QA)
					Binarization4<align, compareType>(src + col, _value, _positive, _negative, dst + col);
				for (; col < alignedWidth; col += A)
					Binarization<align, false, compareType>(src + col, _value, _positive, _negative, dst + col);
				if (col < width)
					Binarization<align, true, compareType>(src + col, _value, _positive, _negative, dst + col, tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
