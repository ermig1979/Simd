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
		struct Func
		{
			typedef void (*FuncPtr)(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
				size_t width, size_t height, uchar * hue, size_t hueStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const Simd::View & y, const View & u, const View & v, View & hue) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
			}
		};	
	}

#define FUNC(function) Func(function, #function)

	bool YuvToHueTest(int width, int height, const Func & f1, const Func & f2, bool is420)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		const int uvWidth = is420 ? width/2 : width;
		const int uvHeight = is420 ? height/2 : height;

		View y(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(y);
		View u(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		FillRandom(u);
		View v(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		FillRandom(v);

		View hue1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hue2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, hue1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, hue2));

		result = result && Compare(hue1, hue2, 0, true, 64, 256);

		return result;
	}

	bool Yuv444ToHueTest()
	{
		bool result = true;

		result = result && YuvToHueTest(W, H, FUNC(Simd::Base::Yuv444ToHue), FUNC(Simd::Yuv444ToHue), false);
		result = result && YuvToHueTest(W + 1, H - 1, FUNC(Simd::Base::Yuv444ToHue), FUNC(Simd::Yuv444ToHue), false);
        result = result && YuvToHueTest(W - 1, H + 1, FUNC(Simd::Base::Yuv444ToHue), FUNC(Simd::Yuv444ToHue), false);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && YuvToHueTest(W, H, FUNC(Simd::Sse2::Yuv444ToHue), FUNC(Simd::Avx2::Yuv444ToHue), false);
            result = result && YuvToHueTest(W + 1, H - 1, FUNC(Simd::Sse2::Yuv444ToHue), FUNC(Simd::Avx2::Yuv444ToHue), false);
            result = result && YuvToHueTest(W - 1, H + 1, FUNC(Simd::Sse2::Yuv444ToHue), FUNC(Simd::Avx2::Yuv444ToHue), false);
        }
#endif 

		return result;
	}

	bool Yuv420ToHueTest()
	{
		bool result = true;

		result = result && YuvToHueTest(W, H, FUNC(Simd::Base::Yuv420ToHue), FUNC(Simd::Yuv420ToHue), true);
		result = result && YuvToHueTest(W - 2, H + 2, FUNC(Simd::Base::Yuv420ToHue), FUNC(Simd::Yuv420ToHue), true);
        result = result && YuvToHueTest(W + 2, H - 2, FUNC(Simd::Base::Yuv420ToHue), FUNC(Simd::Yuv420ToHue), true);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && YuvToHueTest(W, H, FUNC(Simd::Sse2::Yuv420ToHue), FUNC(Simd::Avx2::Yuv420ToHue), true);
            result = result && YuvToHueTest(W - 2, H + 2, FUNC(Simd::Sse2::Yuv420ToHue), FUNC(Simd::Avx2::Yuv420ToHue), true);
            result = result && YuvToHueTest(W + 2, H - 2, FUNC(Simd::Sse2::Yuv420ToHue), FUNC(Simd::Avx2::Yuv420ToHue), true);
        }
#endif 

		return result;
	}
}