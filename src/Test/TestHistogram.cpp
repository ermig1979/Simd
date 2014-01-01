/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/Test.h"

namespace Test
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(
				const uint8_t *src, size_t width, size_t height, size_t stride,
				size_t step, size_t indent, uint32_t * histogram);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, size_t step, size_t indent, uint32_t * histogram) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride,
					step, indent, histogram);
			}
		};
	}

#define FUNC(function) Func(function, #function)

	bool HistogramTest(int width, int height, int step, int indent, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description
			<< " [" << width << ", " << height << "] (" << step << ", " << indent << ")." << std::endl;

		View s(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		Histogram h1 = {0}, h2 = {0};

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, step, indent, h1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, step, indent, h2));

		result = result && Compare(h1, h2, 0, true, 10);

		return result;
	}

	bool AbsSecondDerivativeHistogramTest()
	{
		bool result = true;

		result = result && HistogramTest(W, H, 1, 16, FUNC(Simd::Base::AbsSecondDerivativeHistogram), FUNC(SimdAbsSecondDerivativeHistogram));
		result = result && HistogramTest(W + 1, H - 1, 2, 16, FUNC(Simd::Base::AbsSecondDerivativeHistogram), FUNC(SimdAbsSecondDerivativeHistogram));
		result = result && HistogramTest(W, H, 3, 8, FUNC(Simd::Base::AbsSecondDerivativeHistogram), FUNC(SimdAbsSecondDerivativeHistogram));
		result = result && HistogramTest(W - 1, H + 1, 4, 8, FUNC(Simd::Base::AbsSecondDerivativeHistogram), FUNC(SimdAbsSecondDerivativeHistogram));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && HistogramTest(W, H, 1, 16, FUNC(Simd::Sse2::AbsSecondDerivativeHistogram), FUNC(Simd::Avx2::AbsSecondDerivativeHistogram));
            result = result && HistogramTest(W + 1, H - 1, 2, 16, FUNC(Simd::Sse2::AbsSecondDerivativeHistogram), FUNC(Simd::Avx2::AbsSecondDerivativeHistogram));
            result = result && HistogramTest(W, H, 3, 8, FUNC(Simd::Sse2::AbsSecondDerivativeHistogram), FUNC(Simd::Avx2::AbsSecondDerivativeHistogram));
            result = result && HistogramTest(W - 1, H + 1, 4, 8, FUNC(Simd::Sse2::AbsSecondDerivativeHistogram), FUNC(Simd::Avx2::AbsSecondDerivativeHistogram));
        }
#endif 

		return result;
	}
}
