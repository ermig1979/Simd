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
			typedef void (*FuncPtr)(
				const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
				uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride,
					dst.data, dst.width, dst.height, dst.stride, View::SizeOf(src.format));
			}
		};
	}

#define ARGS(format, width, height, k, function1, function2) \
	format, width, height, k, \
	Func(function1, std::string(#function1) + ColorDescription(format)), \
	Func(function2, std::string(#function2) + ColorDescription(format))

	bool ResizeTest(View::Format format, int width, int height, double k, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description
			<< " [" << int(width*k) << ", " << int(height*k) << "] -> [" << width << ", " << height << "]." << std::endl;

		View s(int(width*k), int(height*k), format, NULL, (int(k*width)%16 == 0 ? 16 : 1));
		FillRandom(s);

		View d1(width, height, format, NULL, (width%16 == 0 ? 16 : 1));
		View d2(width, height, format, NULL, (width%16 == 0 ? 16 : 1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool ResizeBilinearTest()
	{
		bool result = true;

		result = result && ResizeTest(ARGS(View::Gray8, W, H, 0.9, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));
		result = result && ResizeTest(ARGS(View::Gray8, W - 1, H + 1, 1.1, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));

		result = result && ResizeTest(ARGS(View::Uv16, W, H, 0.9, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));
		result = result && ResizeTest(ARGS(View::Uv16, W - 1, H + 1, 1.1, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));

		result = result && ResizeTest(ARGS(View::Bgr24, W, H, 0.9, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));
		result = result && ResizeTest(ARGS(View::Bgr24, W - 1, H + 1, 1.1, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));

		result = result && ResizeTest(ARGS(View::Bgra32, W, H, 0.9, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));
		result = result && ResizeTest(ARGS(View::Bgra32, W - 1, H + 1, 1.1, Simd::Base::ResizeBilinear, Simd::ResizeBilinear));

		return result;
	}
}
