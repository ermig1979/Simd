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
		struct Func
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, 
				uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uchar value, uchar positive, uchar negative, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride);
			}
		};
	}

#define FUNC(function) Func(function, std::string(#function))

	bool BinarizationTest(int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		uchar value = Random(256);
		uchar positive = Random(256);
		uchar negative = Random(256);

		View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, positive, negative, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, positive, negative, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool GreaterThenBinarizationTest()
	{
		bool result = true;

		result = result && BinarizationTest(W, H, FUNC(Simd::Base::GreaterThenBinarization), FUNC(Simd::GreaterThenBinarization));
		result = result && BinarizationTest(W + 1, H - 1, FUNC(Simd::Base::GreaterThenBinarization), FUNC(Simd::GreaterThenBinarization));

		return result;
	}

	bool LesserThenBinarizationTest()
	{
		bool result = true;

		result = result && BinarizationTest(W, H, FUNC(Simd::Base::LesserThenBinarization), FUNC(Simd::LesserThenBinarization));
		result = result && BinarizationTest(W + 1, H - 1, FUNC(Simd::Base::LesserThenBinarization), FUNC(Simd::LesserThenBinarization));

		return result;
	}

	bool EqualToBinarizationTest()
	{
		bool result = true;

		result = result && BinarizationTest(W, H, FUNC(Simd::Base::EqualToBinarization), FUNC(Simd::EqualToBinarization));
		result = result && BinarizationTest(W + 1, H - 1, FUNC(Simd::Base::EqualToBinarization), FUNC(Simd::EqualToBinarization));

		return result;
	}
}
