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

    bool YuvToBgrAutoTest(const Func & f1, const Func & f2, bool is420)
    {
        bool result = true;

        int step = is420 ? 4 : 3;

        result = result && YuvToBgrAutoTest(W, H, f1, f2, is420);
        result = result && YuvToBgrAutoTest(W + step, H - step, f1, f2, is420);
        result = result && YuvToBgrAutoTest(W - step, H + step, f1, f2, is420);

        return result;
    }

    bool Yuv444pToBgrAutoTest()
    {
        bool result = true;

        result = result && YuvToBgrAutoTest(FUNC(Simd::Base::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Ssse3::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Avx2::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Vsx::Yuv444pToBgr), FUNC(SimdYuv444pToBgr), false);
#endif 

        return result;
    }

    bool Yuv420pToBgrAutoTest()
    {
        bool result = true;

        result = result && YuvToBgrAutoTest(FUNC(Simd::Base::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Ssse3::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Avx2::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && YuvToBgrAutoTest(FUNC(Simd::Vsx::Yuv420pToBgr), FUNC(SimdYuv420pToBgr), true);
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool YuvToBgrDataTest(bool create, int width, int height, const Func & f, bool is420)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

        View y(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
        View v(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(y);
            FillRandom(u);
            FillRandom(v);

            TEST_SAVE(y);
            TEST_SAVE(u);
            TEST_SAVE(v);

            f.Call(y, u, v, bgr1);

            TEST_SAVE(bgr1);
        }
        else
        {
            TEST_LOAD(y);
            TEST_LOAD(u);
            TEST_LOAD(v);

            TEST_LOAD(bgr1);

            f.Call(y, u, v, bgr2);

            TEST_SAVE(bgr2);

            result = result && Compare(bgr1, bgr2, 0, true, 64);
        }

        return result;
    }

    bool Yuv420pToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToBgrDataTest(create, DW, DH, FUNC(SimdYuv420pToBgr), true);

        return result;
    }

    bool Yuv444pToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToBgrDataTest(create, DW, DH, FUNC(SimdYuv444pToBgr), false);

        return result;
    }
}