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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
#ifdef SIMD_X64_ENABLE
		template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u(const uint8_t * src, __m512i value, uint64_t * counts, __mmask64 tail = -1)
		{
			const __m512i _src = Load<align, mask>(src, tail);
			__mmask64 bits = Compare8u<compareType>(_src, value);
			counts[0] += _mm_popcnt_u64(bits&tail);
		}

		template <bool align, SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u4(const uint8_t * src, __m512i value, uint64_t * counts)
		{
			counts[0] += _mm_popcnt_u64(Compare8u<compareType>(Load<align>(src + 0 * A), value));
			counts[1] += _mm_popcnt_u64(Compare8u<compareType>(Load<align>(src + 1 * A), value));
			counts[2] += _mm_popcnt_u64(Compare8u<compareType>(Load<align>(src + 2 * A), value));
			counts[3] += _mm_popcnt_u64(Compare8u<compareType>(Load<align>(src + 3 * A), value));
		}

		template <bool align, SimdCompareType compareType> void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
		{
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			size_t fullAlignedWidth = Simd::AlignLo(width, QA);
			__mmask64 tailMask = TailMask64(width - alignedWidth);

			__m512i _value = _mm512_set1_epi8(value);
			uint64_t counts[4] = {0, 0, 0, 0};
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < fullAlignedWidth; col += QA)
					ConditionalCount8u4<align, compareType>(src + col, _value, counts);
				for (; col < alignedWidth; col += A)
					ConditionalCount8u<align, false, compareType>(src + col, _value, counts);
				if (col < width)
					ConditionalCount8u<align, true, compareType>(src + col, _value, counts, tailMask);
				src += stride;
			}
			*count = (uint32_t)(counts[0] + counts[1] + counts[2] + counts[3]);
		}
#else
		template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u(const uint8_t * src, __m512i value, uint32_t * counts, __mmask64 tail = -1)
		{
			const __m512i _src = Load<align, mask>(src, tail);
			union Mask
			{
				__mmask32 m32[2];
				__mmask64 m64[1];
			} bits;
			bits.m64[0] = Compare8u<compareType>(_src, value)&tail;
			counts[0] += _mm_popcnt_u32(bits.m32[0]) + _mm_popcnt_u32(bits.m32[1]);
		}

		template <bool align, SimdCompareType compareType> void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
		{
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);

			__m512i _value = _mm512_set1_epi8(value);
			uint32_t counts[1] = { 0 };
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < alignedWidth; col += A)
					ConditionalCount8u<align, false, compareType>(src + col, _value, counts);
				if (col < width)
					ConditionalCount8u<align, true, compareType>(src + col, _value, counts, tailMask);
				src += stride;
			}
			*count = counts[0];
		}
#endif//SIMD_X64_ENABLE

		template <SimdCompareType compareType> void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
		{
			if (Aligned(src) && Aligned(stride))
				ConditionalCount8u<true, compareType>(src, stride, width, height, value, count);
			else
				ConditionalCount8u<false, compareType>(src, stride, width, height, value, count);
		}

		void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t * count)
		{
			switch (compareType)
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

		template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount16i(const uint8_t * src, __m512i value, uint32_t * counts, __mmask32 tail = -1)
		{
			const __m512i _src = Load<align, mask>((int16_t*)src, tail);
			__mmask32 bits = Compare16i<compareType>(_src, value);
			counts[0] += _mm_popcnt_u32(bits&tail);
		}

		template <bool align, SimdCompareType compareType> SIMD_INLINE void ConditionalCount16i4(const uint8_t * src, __m512i value, uint32_t * counts)
		{
			counts[0] += _mm_popcnt_u32(Compare16i<compareType>(Load<align>(src + 0 * A), value));
			counts[1] += _mm_popcnt_u32(Compare16i<compareType>(Load<align>(src + 1 * A), value));
			counts[2] += _mm_popcnt_u32(Compare16i<compareType>(Load<align>(src + 2 * A), value));
			counts[3] += _mm_popcnt_u32(Compare16i<compareType>(Load<align>(src + 3 * A), value));
		}

		template <bool align, SimdCompareType compareType> void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
		{
			if (align)
				assert(Aligned(src) && Aligned(stride));

			width *= 2;
			size_t alignedWidth = Simd::AlignLo(width, A);
			size_t fullAlignedWidth = Simd::AlignLo(width, QA);
			__mmask32 tailMask = TailMask32((width - alignedWidth)/2);

			__m512i _value = _mm512_set1_epi16(value);
			uint32_t counts[4] = { 0, 0, 0, 0 };
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < fullAlignedWidth; col += QA)
					ConditionalCount16i4<align, compareType>(src + col, _value, counts);
				for (; col < alignedWidth; col += A)
					ConditionalCount16i<align, false, compareType>(src + col, _value, counts);
				if (col < width)
					ConditionalCount16i<align, true, compareType>(src + col, _value, counts, tailMask);
				src += stride;
			}
			*count = counts[0] + counts[1] + counts[2] + counts[3];
		}

		template <SimdCompareType compareType> void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
		{
			if (Aligned(src) && Aligned(stride))
				ConditionalCount16i<true, compareType>(src, stride, width, height, value, count);
			else
				ConditionalCount16i<false, compareType>(src, stride, width, height, value, count);
		}

		void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t * count)
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
	}
#endif// SIMD_AVX512BW_ENABLE
}
