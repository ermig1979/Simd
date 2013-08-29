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
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template <OperationType type> SIMD_INLINE __m256i Operation(const __m256i & a, const __m256i & b);

		template <> SIMD_INLINE __m256i Operation<OperationAverage>(const __m256i & a, const __m256i & b)
		{
			return _mm256_avg_epu8(a, b);
		}

		template <> SIMD_INLINE __m256i Operation<OperationAnd>(const __m256i & a, const __m256i & b)
		{
			return _mm256_and_si256(a, b);
		}

		template <> SIMD_INLINE __m256i Operation<OperationMaximum>(const __m256i & a, const __m256i & b)
		{
			return _mm256_max_epu8(a, b);
		}

        template <> SIMD_INLINE __m256i Operation<OperationSaturatedSubtraction>(const __m256i & a, const __m256i & b)
        {
            return _mm256_subs_epu8(a, b);
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
					const __m256i a_ = Load<align>((__m256i*)(a + offset));
					const __m256i b_ = Load<align>((__m256i*)(b + offset));
					Store<align>((__m256i*)(dst + offset), Operation<type>(a_, b_));
				}
				if(alignedSize != size)
				{
					const __m256i a_ = Load<false>((__m256i*)(a + size - A));
					const __m256i b_ = Load<false>((__m256i*)(b + size - A));
					Store<false>((__m256i*)(dst + size - A), Operation<type>(a_, b_));
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
#endif// SIMD_AVX2_ENABLE
}