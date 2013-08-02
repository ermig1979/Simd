/*
* Simd Library Tests.
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
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/Test.h"

namespace Test
{
	namespace
	{
		struct Func1
		{
			typedef void (*FuncPtr)(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
				uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
			}
		};
	}

#define FUNC1(function) Func1(function, #function)

	namespace
	{
		struct Func2
		{
			typedef void (*FuncPtr)(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
				uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool correction);

			FuncPtr func;
			std::string description;
			bool correction;

			Func2(const FuncPtr & f, const std::string & d, bool c) : func(f), description(d + (c ? "<1>" : "<0>")), correction(c) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, correction);
			}
		};
	}

#define FUNC2(function, correction) Func2(function, #function, correction)

	template <class Func1, class Func2>
	bool ReduceGrayTest(int width, int height, const Func1 & f1, const Func2 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		const int reducedWidth = (width + 1)/2;
		const int reducedHeight = (height + 1)/2;

		View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));
		View d2(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 64);

		return result;
	}

	bool ReduceGray2x2Test()
	{
		bool result = true;

        result = result && ReduceGrayTest(W, H, FUNC1(Simd::Base::ReduceGray2x2), FUNC1(Simd::ReduceGray2x2));
		result = result && ReduceGrayTest(W + 2, H, FUNC1(Simd::Base::ReduceGray2x2), FUNC1(Simd::ReduceGray2x2));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC1(Simd::Base::ReduceGray2x2), FUNC1(Simd::ReduceGray2x2));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && ReduceGrayTest(W, H, FUNC1(Simd::Sse2::ReduceGray2x2), FUNC1(Simd::Avx2::ReduceGray2x2));
            result = result && ReduceGrayTest(W + 2, H, FUNC1(Simd::Sse2::ReduceGray2x2), FUNC1(Simd::Avx2::ReduceGray2x2));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC1(Simd::Sse2::ReduceGray2x2), FUNC1(Simd::Avx2::ReduceGray2x2));
        }
#endif 

		return result;
	}

	bool ReduceGray3x3Test()
	{
		bool result = true;

		result = result && ReduceGrayTest(W, H, FUNC2(Simd::Base::ReduceGray3x3, true), FUNC2(Simd::ReduceGray3x3, true));
		result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Base::ReduceGray3x3, true), FUNC2(Simd::ReduceGray3x3, true));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Base::ReduceGray3x3, true), FUNC2(Simd::ReduceGray3x3, true));

		result = result && ReduceGrayTest(W, H, FUNC2(Simd::Base::ReduceGray3x3, false), FUNC2(Simd::ReduceGray3x3, false));
		result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Base::ReduceGray3x3, false), FUNC2(Simd::ReduceGray3x3, false));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Base::ReduceGray3x3, false), FUNC2(Simd::ReduceGray3x3, false));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && ReduceGrayTest(W, H, FUNC2(Simd::Sse2::ReduceGray3x3, true), FUNC2(Simd::Avx2::ReduceGray3x3, true));
            result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Sse2::ReduceGray3x3, true), FUNC2(Simd::Avx2::ReduceGray3x3, true));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Sse2::ReduceGray3x3, true), FUNC2(Simd::Avx2::ReduceGray3x3, true));

            result = result && ReduceGrayTest(W, H, FUNC2(Simd::Sse2::ReduceGray3x3, false), FUNC2(Simd::Avx2::ReduceGray3x3, false));
            result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Sse2::ReduceGray3x3, false), FUNC2(Simd::Avx2::ReduceGray3x3, false));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Sse2::ReduceGray3x3, false), FUNC2(Simd::Avx2::ReduceGray3x3, false));
        }
#endif 

		return result;
	}

	bool ReduceGray4x4Test()
	{
		bool result = true;

		result = result && ReduceGrayTest(W, H, FUNC1(Simd::Base::ReduceGray4x4), FUNC1(Simd::ReduceGray4x4));
		result = result && ReduceGrayTest(W + 2, H, FUNC1(Simd::Base::ReduceGray4x4), FUNC1(Simd::ReduceGray4x4));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC1(Simd::Base::ReduceGray4x4), FUNC1(Simd::ReduceGray4x4));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && ReduceGrayTest(W, H, FUNC1(Simd::Sse2::ReduceGray4x4), FUNC1(Simd::Avx2::ReduceGray4x4));
            result = result && ReduceGrayTest(W + 2, H, FUNC1(Simd::Sse2::ReduceGray4x4), FUNC1(Simd::Avx2::ReduceGray4x4));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC1(Simd::Sse2::ReduceGray4x4), FUNC1(Simd::Avx2::ReduceGray4x4));
        }
#endif 

		return result;
	}

	bool ReduceGray5x5Test()
	{
		bool result = true;

		result = result && ReduceGrayTest(W, H, FUNC2(Simd::Base::ReduceGray5x5, false), FUNC2(Simd::ReduceGray5x5, false));
		result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Base::ReduceGray5x5, false), FUNC2(Simd::ReduceGray5x5, false));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Base::ReduceGray5x5, false), FUNC2(Simd::ReduceGray5x5, false));

		result = result && ReduceGrayTest(W, H, FUNC2(Simd::Base::ReduceGray5x5, true), FUNC2(Simd::ReduceGray5x5, true));
		result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Base::ReduceGray5x5, true), FUNC2(Simd::ReduceGray5x5, true));
		result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Base::ReduceGray5x5, true), FUNC2(Simd::ReduceGray5x5, true));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && ReduceGrayTest(W, H, FUNC2(Simd::Sse2::ReduceGray5x5, false), FUNC2(Simd::Avx2::ReduceGray5x5, false));
            result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Sse2::ReduceGray5x5, false), FUNC2(Simd::Avx2::ReduceGray5x5, false));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Sse2::ReduceGray5x5, false), FUNC2(Simd::Avx2::ReduceGray5x5, false));

            result = result && ReduceGrayTest(W, H, FUNC2(Simd::Sse2::ReduceGray5x5, true), FUNC2(Simd::Avx2::ReduceGray5x5, true));
            result = result && ReduceGrayTest(W + 2, H, FUNC2(Simd::Sse2::ReduceGray5x5, true), FUNC2(Simd::Avx2::ReduceGray5x5, true));
            result = result && ReduceGrayTest(W - 1, H + 1, FUNC2(Simd::Sse2::ReduceGray5x5, true), FUNC2(Simd::Avx2::ReduceGray5x5, true));
        }
#endif 

		return result;
	}
}