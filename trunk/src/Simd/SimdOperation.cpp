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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdOperation.h"

namespace Simd
{
	namespace Base
	{
		template <OperationType type> SIMD_INLINE uchar Operation(const uchar & a, const uchar & b);

		template <> SIMD_INLINE uchar Operation<OperationAverage>(const uchar & a, const uchar & b)
		{
			return Average(a, b);
		}

		template <> SIMD_INLINE uchar Operation<OperationAnd>(const uchar & a, const uchar & b)
		{
			return  a & b;
		}

		template <> SIMD_INLINE uchar Operation<OperationMaximum>(const uchar & a, const uchar & b)
		{
			return  MaxU8(a, b);
		}

        template <> SIMD_INLINE uchar Operation<OperationSaturatedSubtraction>(const uchar & a, const uchar & b)
        {
            return  SaturatedSubtractionU8(a, b);
        }

		template <OperationType type> void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
		{
			size_t size = width*channelCount;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t offset = 0; offset < size; ++offset)
					dst[offset] = Operation<type>(a[offset], b[offset]);
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, OperationType type)
		{
			switch(type)
			{
			case OperationAverage:
				return Operation<OperationAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case OperationAnd:
				return Operation<OperationAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case OperationMaximum:
				return Operation<OperationMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case OperationSaturatedSubtraction:
                return Operation<OperationSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			default:
				assert(0);
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <OperationType type> SIMD_INLINE __m128i Operation(const __m128i & a, const __m128i & b);

		template <> SIMD_INLINE __m128i Operation<OperationAverage>(const __m128i & a, const __m128i & b)
		{
			return _mm_avg_epu8(a, b);
		}

		template <> SIMD_INLINE __m128i Operation<OperationAnd>(const __m128i & a, const __m128i & b)
		{
			return _mm_and_si128(a, b);
		}

		template <> SIMD_INLINE __m128i Operation<OperationMaximum>(const __m128i & a, const __m128i & b)
		{
			return _mm_max_epu8(a, b);
		}

        template <> SIMD_INLINE __m128i Operation<OperationSaturatedSubtraction>(const __m128i & a, const __m128i & b)
        {
            return _mm_subs_epu8(a, b);
        }

		template <bool align, OperationType type> void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
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

		template <bool align> void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, OperationType type)
		{
			switch(type)
			{
			case OperationAverage:
				return Operation<align, OperationAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case OperationAnd:
				return Operation<align, OperationAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case OperationMaximum:
				return Operation<align, OperationMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case OperationSaturatedSubtraction:
                return Operation<align, OperationSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			default:
				assert(0);
			}
		}

		void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, OperationType type)
		{
			if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
				Operation<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
			else
				Operation<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
		size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, OperationType type)
	{
#ifdef SIMD_AVX2_ENABLE
		if(Avx2::Enable && width*channelCount >= Avx2::A)
			Avx2::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
		else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width*channelCount >= Sse2::A)
			Sse2::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
		else
#endif// SIMD_SSE2_ENABLE
			Base::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
	}
}