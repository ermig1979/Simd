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
#include "Test/TestData.h"
#include "Test/Test.h"

namespace Test 
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
				size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & y, const View & u, const View & v, View & bgra) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, 0xFF);
			}
		};	
	}

#define FUNC(function) Func(function, #function)

	bool YuvToBgraAutoTest(int width, int height, const Func & f1, const Func & f2, bool is420)
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

		View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
		View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, bgra1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, bgra2));

		result = result && Compare(bgra1, bgra2, 0, true, 64);

		return result;
	}

    bool YuvToBgraAutoTest(const Func & f1, const Func & f2, bool is420)
    {
        bool result = true;

        int step = is420 ? 4 : 3;

        result = result && YuvToBgraAutoTest(W, H, f1, f2, is420);
        result = result && YuvToBgraAutoTest(W + step, H - step, f1, f2, is420);
        result = result && YuvToBgraAutoTest(W - step, H + step, f1, f2, is420);

        return result;
    }

	bool Yuv444pToBgraAutoTest()
	{
		bool result = true;

		result = result && YuvToBgraAutoTest(FUNC(Simd::Base::Yuv444pToBgra), FUNC(SimdYuv444pToBgra), false);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && YuvToBgraAutoTest(FUNC(Simd::Sse2::Yuv444pToBgra), FUNC(SimdYuv444pToBgra), false);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && YuvToBgraAutoTest(FUNC(Simd::Avx2::Yuv444pToBgra), FUNC(SimdYuv444pToBgra), false);
#endif 

//#ifdef SIMD_VSX_ENABLE
//        if(Simd::Vsx::Enable)
//            result = result && YuvToBgraAutoTest(FUNC(Simd::Vsx::Yuv444pToBgra), FUNC(SimdYuv444pToBgra), false);
//#endif 

        return result;
	}

	bool Yuv420pToBgraAutoTest()
	{
		bool result = true;

		result = result && YuvToBgraAutoTest(FUNC(Simd::Base::Yuv420pToBgra), FUNC(SimdYuv420pToBgra), true);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && YuvToBgraAutoTest(FUNC(Simd::Sse2::Yuv420pToBgra), FUNC(SimdYuv420pToBgra), true);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && YuvToBgraAutoTest(FUNC(Simd::Avx2::Yuv420pToBgra), FUNC(SimdYuv420pToBgra), true);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && YuvToBgraAutoTest(FUNC(Simd::Vsx::Yuv420pToBgra), FUNC(SimdYuv420pToBgra), true);
#endif 

        return result;
	}

    //-----------------------------------------------------------------------

    bool YuvToBgraDataTest(bool create, int width, int height, const Func & f, bool is420)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

        View y(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
        View v(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(y);
            FillRandom(u);
            FillRandom(v);

            TEST_SAVE(y);
            TEST_SAVE(u);
            TEST_SAVE(v);

            f.Call(y, u, v, bgra1);

            TEST_SAVE(bgra1);
        }
        else
        {
            TEST_LOAD(y);
            TEST_LOAD(u);
            TEST_LOAD(v);

            TEST_LOAD(bgra1);

            f.Call(y, u, v, bgra2);

            TEST_SAVE(bgra2)
            result = result && Compare(bgra1, bgra2, 0, true, 64);
        }

        return result;
    }

    bool Yuv420pToBgraDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToBgraDataTest(create, DW, DH, FUNC(SimdYuv420pToBgra), true);

        return result;
    }
}