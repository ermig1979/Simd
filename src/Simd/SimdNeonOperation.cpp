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
		template <SimdOperationBinary16iType type> SIMD_INLINE int16_t OperationBinary16i(const int16_t & a, const int16_t & b);

		template <> SIMD_INLINE int16_t OperationBinary16i<SimdOperationBinary16iAddition>(const int16_t & a, const int16_t & b)
		{
			return a + b;
		}

		template <> SIMD_INLINE int16_t OperationBinary16i<SimdOperationBinary16iSubtraction>(const int16_t & a, const int16_t & b)
		{
			return a - b;
		}

		template <SimdOperationBinary16iType type> void OperationBinary16i(const int16_t * a, size_t aStride, const int16_t * b, size_t bStride,
			size_t width, size_t height, int16_t * dst, size_t dstStride)
		{
			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0; col < width; ++col)
					dst[col] = OperationBinary16i<type>(a[col], b[col]);
				a += aStride;
				b += bStride;
				dst += dstStride;
			}
		}

		void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
			size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
		{
			assert(aStride%sizeof(int16_t) == 0 && bStride%sizeof(int16_t) == 0 && dstStride%sizeof(int16_t) == 0);

			switch (type)
			{
			case SimdOperationBinary16iAddition:
				return OperationBinary16i<SimdOperationBinary16iAddition>(
					(const int16_t*)a, aStride / sizeof(int16_t), (const int16_t*)b, bStride / sizeof(int16_t), width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
			case SimdOperationBinary16iSubtraction:
				return OperationBinary16i<SimdOperationBinary16iSubtraction>(
					(const int16_t*)a, aStride / sizeof(int16_t), (const int16_t*)b, bStride / sizeof(int16_t), width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
			default:
				assert(0);
			}
		}
    }
#endif// SIMD_NEON_ENABLE
}