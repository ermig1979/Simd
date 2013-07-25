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
#include "Simd/SimdInit.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAbsSecondDerivativeHistogram.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
		namespace
		{
			struct Buffer
			{
				Buffer(size_t size)
				{
					_p = Allocate(sizeof(uchar)*size);
					p = (uchar*)_p;
				}

				~Buffer()
				{
					Free(_p);
				}

				uchar * p;
			private:
				void *_p;
			};
		}

		template <bool srcAlign, bool stepAlign>
		SIMD_INLINE __m256i AbsSecondDerivative(const uchar * src, ptrdiff_t step)
		{
			const __m256i s0 = Load<srcAlign && stepAlign>((__m256i*)(src - step));
			const __m256i s1 = Load<srcAlign>((__m256i*)src);
			const __m256i s2 = Load<srcAlign && stepAlign>((__m256i*)(src + step));
			return AbsDifferenceU8(_mm256_avg_epu8(s0, s2), s1);
		}

		template <bool align>
		SIMD_INLINE void AbsSecondDerivative(const uchar * src, ptrdiff_t colStep, ptrdiff_t rowStep, uchar * dst)
		{
			const __m256i sdX = AbsSecondDerivative<align, false>(src, colStep);
			const __m256i sdY = AbsSecondDerivative<align, true>(src, rowStep);
			Store<align>((__m256i*)dst, _mm256_max_epu8(sdY, sdX));
		}

		template<bool align> void AbsSecondDerivativeHistogram(const uchar *src, size_t width, size_t height, size_t stride,
			size_t step, size_t indent, uint * histogram)
		{
			memset(histogram, 0, sizeof(uint)*HISTOGRAM_SIZE);

			Buffer buffer(stride);
			buffer.p += indent;
			src += indent*(stride + 1);
			height -= 2*indent;
			width -= 2*indent;

			ptrdiff_t bodyStart = (uchar*)AlignHi(buffer.p, A) - buffer.p;
			ptrdiff_t bodyEnd = bodyStart + AlignLo(width - bodyStart, A);
			size_t rowStep = step*stride;
			for(size_t row = 0; row < height; ++row)
			{
				if(bodyStart)
					AbsSecondDerivative<false>(src, step, rowStep, buffer.p);
				for(ptrdiff_t col = bodyStart; col < bodyEnd; col += A)
					AbsSecondDerivative<align>(src + col, step, rowStep, buffer.p + col);
				if(width != (size_t)bodyEnd)
					AbsSecondDerivative<false>(src + width - A, step, rowStep, buffer.p + width - A);

				for(size_t i = 0; i < width; ++i)
					++histogram[buffer.p[i]];

				src += stride;
			}
		}

		void AbsSecondDerivativeHistogram(const uchar *src, size_t width, size_t height, size_t stride,
			size_t step, size_t indent, uint * histogram)
		{
			assert(width > 2*indent && height > 2*indent && indent >= step && width > A + 2*indent);

			if(Aligned(src) && Aligned(stride))
				AbsSecondDerivativeHistogram<true>(src, width, height, stride, step, indent, histogram);
			else
				AbsSecondDerivativeHistogram<false>(src, width, height, stride, step, indent, histogram);
		}
	}
#endif// SIMD_AVX2_ENABLE
}
