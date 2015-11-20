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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
		template <SimdOperationBinary16iType type> SIMD_INLINE int16x8_t OperationBinary16i(const int16x8_t & a, const int16x8_t & b);

		template <> SIMD_INLINE int16x8_t OperationBinary16i<SimdOperationBinary16iAddition>(const int16x8_t & a, const int16x8_t & b)
		{
			return vaddq_s16(a, b);
		}

		template <> SIMD_INLINE int16x8_t OperationBinary16i<SimdOperationBinary16iSubtraction>(const int16x8_t & a, const int16x8_t & b)
		{
			return vsubq_s16(a, b);
		}

		template <bool align, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			size_t width, size_t height, uint8_t * dst, size_t dstStride)
		{
			assert(width*sizeof(uint16_t) >= A);
			if (align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

			size_t size = width*sizeof(int16_t);
			size_t alignedSize = Simd::AlignLo(size, A);
			for (size_t row = 0; row < height; ++row)
			{
				for (size_t offset = 0; offset < alignedSize; offset += A)
				{
					const int16x8_t a_ = (int16x8_t)Load<align>(a + offset);
					const int16x8_t b_ = (int16x8_t)Load<align>(b + offset);
					Store<align>(dst + offset, (uint8x16_t)OperationBinary16i<type>(a_, b_));
				}
				if (alignedSize != size)
				{
					const int16x8_t a_ = (int16x8_t)Load<false>(a + size - A);
					const int16x8_t b_ = (int16x8_t)Load<false>(b + size - A);
					Store<false>(dst + size - A, (uint8x16_t)OperationBinary16i<type>(a_, b_));
				}
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		template <bool align> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
		{
			switch (type)
			{
			case SimdOperationBinary16iAddition:
				return OperationBinary16i<align, SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
			case SimdOperationBinary16iSubtraction:
				return OperationBinary16i<align, SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
			default:
				assert(0);
			}
		}

		void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
		{
			if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
				OperationBinary16i<true>(a, aStride, b, bStride, width, height, dst, dstStride, type);
			else
				OperationBinary16i<false>(a, aStride, b, bStride, width, height, dst, dstStride, type);
		}
    }
#endif// SIMD_NEON_ENABLE
}