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
#include "Simd/SimdMath.h"
#include "Simd/SimdAbsGradientSaturatedSum.h"

namespace Simd
{
	namespace Base
	{
		void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
		{
			memset(dst, 0, width);
			src += srcStride;
			dst += dstStride;
			for (size_t row = 2; row < height; ++row)
			{
				dst[0] = 0;
				for (size_t col = 1; col < width - 1; ++col)
				{
					const int dy = AbsDifferenceU8(src[col - srcStride], src[col + srcStride]);
					const int dx = AbsDifferenceU8(src[col - 1], src[col + 1]);
					dst[col] = MinU8(dx + dy, 0xFF);
				}
				dst[width - 1] = 0;

				src += srcStride;
				dst += dstStride;
			}
			memset(dst, 0, width);
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template<bool align> SIMD_INLINE __m128i AbsGradientSaturatedSum(const uchar * src, size_t stride)
		{
			const __m128i s10 = Load<false>((__m128i*)(src - 1));
			const __m128i s12 = Load<false>((__m128i*)(src + 1));
			const __m128i s01 = Load<align>((__m128i*)(src - stride));
			const __m128i s21 = Load<align>((__m128i*)(src + stride));
			const __m128i dx = AbsDifferenceU8(s10, s12);
			const __m128i dy = AbsDifferenceU8(s01, s21);
			return _mm_adds_epu8(dx, dy);
		}

		template<bool align> void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
		{
			size_t alignedWidth = AlignLo(width, A);
			memset(dst, 0, width);
			src += srcStride;
			dst += dstStride;
			for (size_t row = 2; row < height; ++row)
			{
				for (size_t col = 0; col < alignedWidth; col += A)
					Store<align>((__m128i*)(dst + col), AbsGradientSaturatedSum<align>(src + col, srcStride));
				if(width != alignedWidth)
					Store<false>((__m128i*)(dst + width - A), AbsGradientSaturatedSum<false>(src + width - A, srcStride));

				dst[0] = 0;
				dst[width - 1] = 0;

				src += srcStride;
				dst += dstStride;
			}
			memset(dst, 0, width);
		}

		void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
		{
			if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
				AbsGradientSaturatedSum<true>(src, srcStride, width, height, dst, dstStride);
			else
				AbsGradientSaturatedSum<false>(src, srcStride, width, height, dst, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
		else
#endif//SIMD_SSE2_ENABLE
			Base::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
	}

	void AbsGradientSaturatedSum(const View & src, View & dst)
	{
		assert(src.width == dst.width && src.height == dst.height && src.format == dst.format);
		assert(src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

		AbsGradientSaturatedSum(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
	}
}