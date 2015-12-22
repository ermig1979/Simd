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
#include "Simd/SimdExtract.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
	namespace Neon
	{
        template <bool align, SimdCompareType compareType> 
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;

#ifdef __GNUC__
			uint8x16_t _value = SIMD_VEC_SET1_EPI8(value);
#else
			uint8x16_t _value = vld1q_dup_u8(&value);
#endif
			uint32x4_t _count = K32_00000000;
            for(size_t row = 0; row < height; ++row)
            {
				uint16x8_t rowSum = K16_0000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint8x16_t blockSum = K8_00;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t mask = Compare8u<compareType>(Load<align>(src + col), _value);
						blockSum = vaddq_u8(blockSum, vandq_u8(mask, K8_01));
					}
					rowSum = vaddq_u16(rowSum, HorizontalSum(blockSum));
				}
                if(alignedWidth != width)
                {
                    const uint8x16_t mask = vandq_u8(Compare8u<compareType>(Load<false>(src + width - A), _value), tailMask);
					rowSum = vaddq_u16(rowSum, HorizontalSum(vandq_u8(mask, K8_01)));
                }
				_count = vaddq_u32(_count, HorizontalSum(rowSum));
                src += stride;
            }
            *count = ExtractSum(_count);
        }

        template <SimdCompareType compareType> 
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            if(Aligned(src) && Aligned(stride))
                ConditionalCount8u<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount8u<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch(compareType)
            {
            case SimdCompareEqual: 
                return ConditionalCount8u<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual: 
                return ConditionalCount8u<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater: 
                return ConditionalCount8u<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual: 
                return ConditionalCount8u<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser: 
                return ConditionalCount8u<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual: 
                return ConditionalCount8u<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default: 
                assert(0);
            }
        }

		template <bool align, SimdCompareType compareType>
		void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
		{
			assert(width >= HA);
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, HA);
			uint16x8_t tailMask = (uint16x8_t)ShiftLeft(K8_FF, 2 * (HA - width + alignedWidth));

#ifdef __GNUC__
			int16x8_t _value = SIMD_VEC_SET1_EPI16(value);
#else
			int16x8_t _value = vld1q_dup_s16(&value);
#endif
			uint32x4_t _count = K32_00000000;
			for (size_t row = 0; row < height; ++row)
			{
				const int16_t * s = (const int16_t *)src;
				uint16x8_t rowSum = K16_0000;
				for (size_t col = 0; col < alignedWidth; col += HA)
				{
					const uint16x8_t mask = Compare16i<compareType>(Load<align>(s + col), _value);
					rowSum = vaddq_u16(rowSum, vandq_u16(mask, K16_0001));
				}
				if (alignedWidth != width)
				{
					const uint16x8_t mask = vandq_u16(Compare16i<compareType>(Load<false>(s + width - HA), _value), tailMask);
					rowSum = vaddq_u16(rowSum, vandq_u16(mask, K16_0001));
				}
				_count = vaddq_u32(_count, HorizontalSum(rowSum));
				src += stride;
			}
			*count = ExtractSum(_count);
		}

		template <SimdCompareType compareType>
		void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
		{
			if (Aligned(src) && Aligned(stride))
				ConditionalCount16i<true, compareType>(src, stride, width, height, value, count);
			else
				ConditionalCount16i<false, compareType>(src, stride, width, height, value, count);
		}

		void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
			int16_t value, SimdCompareType compareType, uint32_t * count)
		{
			switch (compareType)
			{
			case SimdCompareEqual:
				return ConditionalCount16i<SimdCompareEqual>(src, stride, width, height, value, count);
			case SimdCompareNotEqual:
				return ConditionalCount16i<SimdCompareNotEqual>(src, stride, width, height, value, count);
			case SimdCompareGreater:
				return ConditionalCount16i<SimdCompareGreater>(src, stride, width, height, value, count);
			case SimdCompareGreaterOrEqual:
				return ConditionalCount16i<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
			case SimdCompareLesser:
				return ConditionalCount16i<SimdCompareLesser>(src, stride, width, height, value, count);
			case SimdCompareLesserOrEqual:
				return ConditionalCount16i<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
			default:
				assert(0);
			}
		}

		template <bool align, SimdCompareType compareType>
		void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;

#ifdef __GNUC__
			uint8x16_t _value = SIMD_VEC_SET1_EPI8(value);
#else
			uint8x16_t _value = vld1q_dup_u8(&value);
#endif
			uint64x2_t _sum = {0, 0};
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						const uint8x16_t _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
						blockSum = vaddq_u16(blockSum, HorizontalSum(vandq_u8(_mask, _src)));
					}
					rowSum = vaddq_u32(rowSum, HorizontalSum(blockSum));
				}
				if (alignedWidth != width)
				{
					const uint8x16_t _src = Load<false>(src + width - A);
					const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<false>(mask + width - A), _value), tailMask);
					rowSum = vaddq_u32(rowSum, HorizontalSum(HorizontalSum(vandq_u8(_mask, _src))));
				}
				_sum = vaddq_u64(_sum, HorizontalSum(rowSum));
				src += srcStride;
				mask += maskStride;
			}
			*sum = ExtractSum(_sum);
		}

		template <SimdCompareType compareType>
		void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
		{
			if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
				ConditionalSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
			else
				ConditionalSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
		}

		void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
		{
			switch (compareType)
			{
			case SimdCompareEqual:
				return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
			case SimdCompareNotEqual:
				return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
			case SimdCompareGreater:
				return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
			case SimdCompareGreaterOrEqual:
				return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
			case SimdCompareLesser:
				return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
			case SimdCompareLesserOrEqual:
				return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
			default:
				assert(0);
			}
		}
	}
#endif// SIMD_SSE2_ENABLE
}