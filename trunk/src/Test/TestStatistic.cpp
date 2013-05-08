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
			typedef void (*FuncPtr)(const uchar *src, size_t stride, size_t width, size_t height, 
				uchar * min, uchar * max, uchar * average);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uchar * min, uchar * max, uchar * average) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, min, max, average);
			}
		};
	}

#define FUNC(function) Func(function, #function)

	bool GetStatisticTest(int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		uchar min1, max1, average1;
		uchar min2, max2, average2;

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, &min1, &max1, &average1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, &min2, &max2, &average2));

		if(min1 != min2)
		{
			result = false;
			std::cout << "Error min: (" << min1 << " != " << min2 << ")! " << std::endl;
		}
		if(max1 != max2)
		{
			result = false;
			std::cout << "Error max: (" << max1 << " != " << max2 << ")! " << std::endl;
		}
		if(average1 != average2)
		{
			result = false;
			std::cout << "Error average: (" << average1 << " != " << average2 << ")! " << std::endl;
		}
		return result;
	}

	bool GetStatisticTest()
	{
		bool result = true;

		result = result && GetStatisticTest(W, H, FUNC(Simd::Base::GetStatistic), FUNC(Simd::GetStatistic));
		result = result && GetStatisticTest(W + 1, H - 1, FUNC(Simd::Base::GetStatistic), FUNC(Simd::GetStatistic));

		return result;
	}
}
