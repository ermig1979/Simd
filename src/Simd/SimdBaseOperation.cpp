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
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"

namespace Simd
{
	namespace Base
	{
		template <SimdOperationType type> SIMD_INLINE uchar Operation(const uchar & a, const uchar & b);

		template <> SIMD_INLINE uchar Operation<SimdOperationAverage>(const uchar & a, const uchar & b)
		{
			return Average(a, b);
		}

		template <> SIMD_INLINE uchar Operation<SimdOperationAnd>(const uchar & a, const uchar & b)
		{
			return  a & b;
		}

		template <> SIMD_INLINE uchar Operation<SimdOperationMaximum>(const uchar & a, const uchar & b)
		{
			return  MaxU8(a, b);
		}

        template <> SIMD_INLINE uchar Operation<SimdOperationSaturatedSubtraction>(const uchar & a, const uchar & b)
        {
            return  SaturatedSubtractionU8(a, b);
        }

		template <SimdOperationType type> void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
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
			size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, SimdOperationType type)
		{
			switch(type)
			{
			case SimdOperationAverage:
				return Operation<SimdOperationAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case SimdOperationAnd:
				return Operation<SimdOperationAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			case SimdOperationMaximum:
				return Operation<SimdOperationMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationSaturatedSubtraction:
                return Operation<SimdOperationSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
			default:
				assert(0);
			}
		}

        void VectorProduct(const uchar * vertical, const uchar * horizontal, uchar * dst, size_t stride, size_t width, size_t height)
        {
            for(size_t row = 0; row < height; ++row)
            {
                int _vertical = vertical[row];
                for(size_t col = 0; col < width; ++col)
                    dst[col] = DivideBy255(_vertical * horizontal[col]);
                dst += stride;
            }
        }
	}
}