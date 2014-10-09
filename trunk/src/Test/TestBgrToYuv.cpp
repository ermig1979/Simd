/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar,
*               2014-2014 Antonenka Mikhail.
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
#include "Test/TestData.h"
#include "Test/Test.h"

namespace Test 
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
				uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & bgr, View & y, View & u, View & v) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
			}
		};	
	}

#define FUNC(function) Func(function, #function)

	bool BgrToYuvAutoTest(int width, int height, const Func & f1, const Func & f2, bool is420)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

		View bgr(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
		FillRandom(bgr);

		View y1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		View y2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bgr, y1, u1, v1));
		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bgr, y2, u2, v2));

		result = result && Compare(y1, y2, 0, true, 64);
		result = result && Compare(u1, u2, 0, true, 64);
		result = result && Compare(v1, v2, 0, true, 64);

		return result;
	}

	bool BgrToYuvAutoTest(const Func & f1, const Func & f2, bool is420)
	{
		bool result = true;

        int step = is420 ? E : O;
		
		result = result && BgrToYuvAutoTest(W, H, f1, f2, is420);
		result = result && BgrToYuvAutoTest(W + step, H - step, f1, f2, is420);
		result = result && BgrToYuvAutoTest(W - step, H + step, f1, f2, is420);
		
		return result;
	}

	bool BgrToYuv420pAutoTest()
	{
		bool result = true;

		result = result && BgrToYuvAutoTest(FUNC(Simd::Base::BgrToYuv420p), FUNC(SimdBgrToYuv420p), true);

		return result;
	}

	//-----------------------------------------------------------------------

	bool BgrToYuvDataTest(bool create, int width, int height, const Func & f, bool is420)
	{
		bool result = true;

		Data data(f.description);

		std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

		View bgr(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

		View y1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		View y2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		if(create)
		{
			FillRandom(bgr);

			TEST_SAVE(bgr);

			f.Call(bgr, y1, u1, v1);

			TEST_SAVE(y1);
			TEST_SAVE(u1);
			TEST_SAVE(v1);
		}
		else
		{
			TEST_LOAD(bgr);

			TEST_LOAD(y1);
			TEST_LOAD(u1);
			TEST_LOAD(v1);

			f.Call(bgr, y2, u2, v2);

			TEST_SAVE(y2);
			TEST_SAVE(u2);
			TEST_SAVE(v2);

			result = result && Compare(y1, y2, 0, true, 64);
			result = result && Compare(u1, u2, 0, true, 64);
			result = result && Compare(v1, v2, 0, true, 64);
		}

		return result;
	}

	bool BgrToYuv420pDataTest(bool create)
	{
		bool result = true;

		result = result && BgrToYuvDataTest(create, DW, DH, FUNC(SimdBgrToYuv420p), true);

		return result;
	}
}