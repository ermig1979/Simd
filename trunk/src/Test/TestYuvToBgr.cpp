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
			typedef void (*FuncPtr)(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
				size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & y, const View & u, const View & v, View & bgr) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
			}
		};	
	}

#define FUNC(function) Func(function, #function)

	bool YuvToBgrAutoTest(int width, int height, const Func & f1, const Func & f2, bool is420)
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

		View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
		View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, bgr1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, bgr2));

		result = result && Compare(bgr1, bgr2, 0, true, 64);

		return result;
	}

	bool Yuv444pToBgrAutoTest()
	{
		bool result = true;

		result = result && YuvToBgrAutoTest(W, H, FUNC(Simd::Base::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);
		result = result && YuvToBgrAutoTest(W + 1, H - 1, FUNC(Simd::Base::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);
        result = result && YuvToBgrAutoTest(W - 1, H + 1, FUNC(Simd::Base::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
        {
            result = result && YuvToBgrAutoTest(W, H, FUNC(Simd::Ssse3::Yuv444pToBgr), FUNC(Simd::Avx2::Yuv444pToBgr), false);
            result = result && YuvToBgrAutoTest(W + 1, H - 1, FUNC(Simd::Ssse3::Yuv444pToBgr), FUNC(Simd::Avx2::Yuv444pToBgr), false);
            result = result && YuvToBgrAutoTest(W - 1, H + 1, FUNC(Simd::Ssse3::Yuv444pToBgr), FUNC(Simd::Avx2::Yuv444pToBgr), false);
        }
#endif 

		return result;
	}

	bool Yuv420pToBgrAutoTest()
	{
		bool result = true;

		result = result && YuvToBgrAutoTest(W, H, FUNC(Simd::Base::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);
        result = result && YuvToBgrAutoTest(W + 2, H - 2, FUNC(Simd::Base::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);
		result = result && YuvToBgrAutoTest(W - 2, H + 2, FUNC(Simd::Base::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
        {
            result = result && YuvToBgrAutoTest(W, H, FUNC(Simd::Ssse3::Yuv420pToBgr), FUNC(Simd::Avx2::Yuv420pToBgr), true);
            result = result && YuvToBgrAutoTest(W + 2, H - 2, FUNC(Simd::Ssse3::Yuv420pToBgr), FUNC(Simd::Avx2::Yuv420pToBgr), true);
            result = result && YuvToBgrAutoTest(W - 2, H + 2, FUNC(Simd::Ssse3::Yuv420pToBgr), FUNC(Simd::Avx2::Yuv420pToBgr), true);
        }
#endif 

		return result;
	}
}