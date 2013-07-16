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

namespace Test
{
	namespace
	{
		struct Func1
		{
			typedef void (*FuncPtr)(const uchar * a, size_t aStride, const uchar * b, size_t bStride,
				size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, View::SizeOf(a.format), dst.data, dst.stride);
			}
		};
	}

#define ARGS1(format, width, height, function1, function2) \
	format, width, height, \
	Func1(function1.func, function1.description + ColorDescription(format)), \
	Func1(function2.func, function2.description + ColorDescription(format))

#define ARGS2(function1, function2) \
	Func1(function1, std::string(#function1)), Func1(function2, std::string(#function2))

	bool OperationTest(View::Format format, int width, int height, const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool OperationTest(const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		result = result && OperationTest(ARGS1(View::Gray8, W, H, f1, f2));
		result = result && OperationTest(ARGS1(View::Gray8, W + 1, H - 1, f1, f2));

		result = result && OperationTest(ARGS1(View::Uv16, W, H, f1, f2));
		result = result && OperationTest(ARGS1(View::Uv16, W + 1, H - 1, f1, f2));

		result = result && OperationTest(ARGS1(View::Bgr24, W, H, f1, f2));
		result = result && OperationTest(ARGS1(View::Bgr24, W + 1, H - 1, f1, f2));

		result = result && OperationTest(ARGS1(View::Bgra32, W, H, f1, f2));
		result = result && OperationTest(ARGS1(View::Bgra32, W + 1, H - 1, f1, f2));

		return result;
	}

	bool AverageTest()
	{
		bool result = true;

		result = result && OperationTest(ARGS2(Simd::Base::Average, Simd::Average));

#ifdef SIMD_SSE2_ENABLE
		if(Simd::Sse2::Enable)
			result = result && OperationTest(ARGS2(Simd::Sse2::Average, Simd::Average));
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
		if(Simd::Avx2::Enable)
			result = result && OperationTest(ARGS2(Simd::Avx2::Average, Simd::Average));
#endif//SIMD_AVX2_ENABLE

		return result;
	}

	bool AndTest()
	{
		bool result = true;

		result = result && OperationTest(ARGS2(Simd::Base::And, Simd::And));

		return result;
	}
}
