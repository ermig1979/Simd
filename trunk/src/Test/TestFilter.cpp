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
		struct ColorFunc
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			ColorFunc(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, View::SizeOf(src.format), dst.data, dst.stride);
			}
		};

		struct GrayFunc
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			GrayFunc(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
			}
		};
	}

#define ARGS(format, width, height, function1, function2) \
	format, width, height, \
	ColorFunc(function1, std::string(#function1) + ColorDescription(format)), \
	ColorFunc(function2, std::string(#function2) + ColorDescription(format))

#define FUNC(function) GrayFunc(function, std::string(#function))

	bool ColorFilterTest(View::Format format, int width, int height, const ColorFunc & f1, const ColorFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool GrayFilterTest(int width, int height, const GrayFunc & f1, const GrayFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool MedianFilterSquare3x3Test()
	{
		bool result = true;

		result = result && ColorFilterTest(ARGS(View::Gray8, W, H, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));
		result = result && ColorFilterTest(ARGS(View::Gray8, W + 2, H - 1, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));

		result = result && ColorFilterTest(ARGS(View::Uv16, W, H, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));
		result = result && ColorFilterTest(ARGS(View::Uv16, W + 2, H - 1, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));

		result = result && ColorFilterTest(ARGS(View::Bgr24, W, H, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));
		result = result && ColorFilterTest(ARGS(View::Bgr24, W + 2, H - 1, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));

		result = result && ColorFilterTest(ARGS(View::Bgra32, W, H, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));
		result = result && ColorFilterTest(ARGS(View::Bgra32, W + 2, H - 1, Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));

		return result;
	}

	bool MedianFilterSquare5x5Test()
	{
		bool result = true;

		result = result && ColorFilterTest(ARGS(View::Gray8, W, H, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));
		result = result && ColorFilterTest(ARGS(View::Gray8, W + 2, H - 1, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));

		result = result && ColorFilterTest(ARGS(View::Uv16, W, H, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));
		result = result && ColorFilterTest(ARGS(View::Uv16, W + 2, H - 1, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));

		result = result && ColorFilterTest(ARGS(View::Bgr24, W, H, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));
		result = result && ColorFilterTest(ARGS(View::Bgr24, W + 2, H - 1, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));

		result = result && ColorFilterTest(ARGS(View::Bgra32, W, H, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));
		result = result && ColorFilterTest(ARGS(View::Bgra32, W + 2, H - 1, Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));

		return result;
	}

	bool GaussianBlur3x3Test()
	{
		bool result = true;

		result = result && ColorFilterTest(ARGS(View::Gray8, W, H, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));
		result = result && ColorFilterTest(ARGS(View::Gray8, W + 1, H - 1, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));

		result = result && ColorFilterTest(ARGS(View::Uv16, W, H, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));
		result = result && ColorFilterTest(ARGS(View::Uv16, W + 1, H - 1, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));

		result = result && ColorFilterTest(ARGS(View::Bgr24, W, H, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));
		result = result && ColorFilterTest(ARGS(View::Bgr24, W + 1, H - 1, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));

		result = result && ColorFilterTest(ARGS(View::Bgra32, W, H, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));
		result = result && ColorFilterTest(ARGS(View::Bgra32, W + 1, H - 1, Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));

		return result;
	}

	bool AbsGradientSaturatedSumTest()
	{
		bool result = true;

		result = result && GrayFilterTest(W, H, FUNC(Simd::Base::AbsGradientSaturatedSum), FUNC(Simd::AbsGradientSaturatedSum));
		result = result && GrayFilterTest(W + 1, H - 1, FUNC(Simd::Base::AbsGradientSaturatedSum), FUNC(Simd::AbsGradientSaturatedSum));

		return result;
	}
}
