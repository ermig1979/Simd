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
			typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
				const uint8_t * bkg, size_t bkgStride, double shiftX, double shiftY, 
				size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, const View & bkg, double shiftX, double shiftY, 
				size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, View::PixelSize(src.format), bkg.data, bkg.stride,
					shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst.data, dst.stride);
			}
		};
	}

#define ARGS1(format, width, height, function1, function2) \
	format, width, height, \
	Func(function1.func, function1.description + ColorDescription(format)), \
	Func(function2.func, function2.description + ColorDescription(format))

#define ARGS2(function1, function2) \
    Func(function1, std::string(#function1)), Func(function2, std::string(#function2))

	bool ShiftTest(View::Format format, int width, int height, double dx, double dy, int crop, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << std::setprecision(1) << std::fixed 
			<< "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "],"
			<< " (" << dx << ", " << dy << ", " << crop << ")." << std::endl;

		View s(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(s);
		View b(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool ShiftTest(View::Format format, int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		const double x0 = 7.1, dx = -5.3, y0 = -5.2, dy = 3.7;
		for(int i = 0; i < 4; ++i)
			result = result && ShiftTest(format, width, height, x0 + i*dx, y0 + i*dy, i*3, f1, f2);

		return result;
	}

    bool ShiftBilinearTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ShiftTest(ARGS1(format, W, H, f1, f2));
            result = result && ShiftTest(ARGS1(format, W + 1, H - 1, f1, f2));
            result = result && ShiftTest(ARGS1(format, W - 1, H + 1, f1, f2));
        }

        return result;
    }

	bool ShiftBilinearTest()
	{
		bool result = true;

		result = result && ShiftBilinearTest(ARGS2(Simd::Base::ShiftBilinear, SimdShiftBilinear));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ShiftBilinearTest(ARGS2(Simd::Sse2::ShiftBilinear, Simd::Avx2::ShiftBilinear));
#endif 

		return result;
	}
}
