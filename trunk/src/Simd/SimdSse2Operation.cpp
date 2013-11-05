/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSse2.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <SimdOperationType type> SIMD_INLINE __m128i Operation(const __m128i & a, const __m128i & b);

		template <> SIMD_INLINE __m128i Operation<SimdOperationAverage>(const __m128i & a, const __m128i & b)
		{
			return _mm_avg_epu8(a, b);
		}

		template <> SIMD_INLINE __m128i Operation<SimdOperationAnd>(const __m128i & a, const __m128i & b)
		{
			return _mm_and_si128(a, b);
		}

		template <> SIMD_INLINE __m128i Operation<SimdOperationMaximum>(const __m128i & a, const __m128i & b)
		{
			return _mm_max_epu8(a, b);
		}

        template <> SIMD_INLINE __m128i Operation<SimdOperationSaturatedSubtraction>(const __m128i & a, const __m128i & b)
        {
            return _mm_subs_epu8(a, b);
        }

		template <bool align, SimdOperationType type> void Operation(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
		{
			assert(width*channelCount >= A);
			if(align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

			size_t size = channelCount*width;
			size_t alignedSize = Simd::AlignLo(size, A);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t offset = 0; offset < alignedSize; offset += A)
				{
					const __m128i a_ = Load<align>((__m128i*)(a + offset));
					const __m128i b_ = Load<align>((__m128i*)(b + offset));
					Store<align>((__m128i*)(dst + offset), Operation<type>(a_, b_));
				}
				if(alignedSize != size)
				{
					const __m128i a_ = Load<false>((__m128i*)(a + size - A));
					const __m128i b_ = Load<false>((__m128i*)(b + size - A));
					Store<false>((__m128i*)(dst + size - A), Operation<type>(a_, b_));
				}
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		template <bool align> void Operation(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationType type)
		{
			switch(type)
			{
			case SimdOperationAverage:
				return Operation<align, SimdOperationAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case SimdOperationAnd:
				return Operation<align, SimdOperationAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case SimdOperationMaximum:
				return Operation<align, SimdOperationMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationSaturatedSubtraction:
                return Operation<align, SimdOperationSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			default:
				assert(0);
			}
		}

		void Operation(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationType type)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
				Operation<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
			else
				Operation<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
		}

        template <bool align> SIMD_INLINE void VectorProduct(const __m128i & vertical, const uint8_t * horizontal, uint8_t * dst)
        {
            __m128i _horizontal = Load<align>((__m128i*)horizontal);
            __m128i lo = DivideI16By255(_mm_mullo_epi16(vertical, _mm_unpacklo_epi8(_horizontal, K_ZERO)));
            __m128i hi = DivideI16By255(_mm_mullo_epi16(vertical, _mm_unpackhi_epi8(_horizontal, K_ZERO)));
            Store<align>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        } 

        template <bool align> void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            for(size_t row = 0; row < height; ++row)
            {
                __m128i _vertical = _mm_set1_epi16(vertical[row]);
                for(size_t col = 0; col < alignedWidth; col += A)
                    VectorProduct<align>(_vertical, horizontal + col, dst + col);
                if(alignedWidth != width)
                    VectorProduct<false>(_vertical, horizontal + width - A, dst + width - A);
                dst += stride;
            }
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if(Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                VectorProduct<true>(vertical, horizontal, dst, stride, width, height);
            else
                VectorProduct<false>(vertical, horizontal, dst, stride, width, height);
        }
	}
#endif// SIMD_SSE2_ENABLE
}